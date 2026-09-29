"""
Post-hoc, no-retraining interpolation with a SINGLE PROTOTYPE per label, on
top of the same BertHSLN baseline used throughout this repo.

Unlike PCM/PBR, this method needs access to an already-trained model's
internal sentence representations to build the prototype datastore. Rather
than adding checkpoint save/reload plumbing (never used elsewhere in this
repo), this script trains its own baseline in-process and keeps the best
model in memory the whole time -- train once, then immediately: build
prototypes, grid-search the interpolation hyperparameters on dev, evaluate
on test.

Method (single prototype per label instead of a full kNN datastore -- no
retrieval step):
  1. Train the baseline HSLN model (identical architecture/config to
     baseline_run.py with no context injection).
  2. Datastore: one forward pass over the TRAIN set (labels=None,
     return_sent_repr=True) to get each sentence's post-sentence-LSTM
     representation c_i. Average c_i per gold label -> one prototype P_v
     per rhetorical role v.
  3. Interpolation: for each sentence x_i (dev/test), compute
         p_kNN(l_i=v | x_i) = softmax_v( -d(c_i, P_v) / tau )   (k=1 "neighbour" per label)
         p_final = lambda * p_baseline + (1 - lambda) * p_kNN
     where p_baseline is the trained classifier's own softmax (CRFOutputLayer's
     "logits", i.e. the per-position distribution before CRF decoding).
  4. lambda in [0,1] (step --lambda_step) and tau in --tau_grid are grid-searched
     on dev to maximize macro-F1, then that (lambda, tau) is applied once on
     test for the reported numbers.

Usage:
    python mind_neighbours_run.py --task <task> --seed 1 \
        --tokenized_folder processed-datasets --output_dir output-training-<task>
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from eval import calc_classification_metrics
from models import BertHSLN
from prototype_net.distance import pairwise_dist
from task import pubmed_task
from train import SentenceClassificationTrainer
from utils import ResultWriter, get_device, log, tensor_dict_to_cpu, tensor_dict_to_gpu


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "yes", "1"):
        return True
    if v.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--tokenized_folder", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--mini_data", type=str2bool, default=False)
parser.add_argument("--lambda_step", type=float, default=0.1,
                    help="Grid step for lambda in [0,1] (paper: increments of 0.1).")
parser.add_argument("--tau_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0],
                    help="Softmax temperature candidates for p_kNN (not given explicitly "
                         "in the paper excerpt available here, so swept like lambda).")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

BERT_MODEL = "models/bert-base-uncased"
config = {
    "bert_model": BERT_MODEL,
    "bert_trainable": False,
    "model": BertHSLN.__name__,
    "cacheable_tasks": [],

    "dropout": 0.5,
    "word_lstm_hs": 758,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,

    "lr": 3e-05,
    "lr_epoch_decay": 0.9,
    "batch_size": 32,
    "max_seq_length": 128,
    "max_epochs": 1 if args.mini_data else 10,
    "early_stopping": 5,

    "strategy": "baseline",
    "unique_name": "mind_proto",
    "sentence_attention_style": "mind_proto",
    "window_size": 4,

    "use_crf": True,
    "use_sentence_lstm": True,
    "use_word_lstm": True,
    "use_attention_pooling": True,

    # No PCM / no PBR: pure baseline architecture, we only need its own
    # internal sentence representations and its own softmax predictions.
    "centroid_paths": f"matching-context/new_similarity_outputs_with_labels/decoder/mean/{args.task}",  # unused, ctx_position=""
    "centroid_dim": 768,
    "ctx_fusion": "concat_proj",
    "ctx_position": "",
    "use_prototypes": False,

    # Disable the (unrelated, pre-existing) file-dump side channel in
    # models.py's forward() -- we use return_sent_repr instead, in-memory.
    "save_representations_to": None,
}

MAX_DOCS = 2 if args.mini_data else -1


def collect(model, batches, device, need_probs):
    """One forward pass (no grad) over `batches`, returning:
        reprs  (N, D) -- sent_repr at every non-padded sentence position
        golds  (N,)   -- gold label id at those same positions
        probs  (N, C) or None -- p_baseline (CRFOutputLayer's softmax "logits")
    """
    reprs_list, golds_list, probs_list = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            tensor_dict_to_gpu(batch, device)
            output = model(batch=batch, labels=None, return_sent_repr=True)
            label_ids = batch["label_ids"]
            active = label_ids > 0  # 0 = "mask"/pad, same filter as eval.py
            reprs_list.append(output["sent_repr"][active].detach().cpu())
            golds_list.append(label_ids[active].detach().cpu())
            if need_probs:
                probs_list.append(output["logits"][active].detach().cpu())
            tensor_dict_to_cpu(batch)
    reprs = torch.cat(reprs_list, dim=0)
    golds = torch.cat(golds_list, dim=0)
    probs = torch.cat(probs_list, dim=0) if need_probs else None
    return reprs, golds, probs


def build_prototypes(reprs, golds, n_classes):
    d = reprs.size(-1)
    proto = torch.zeros(n_classes, d)
    for c in range(1, n_classes):  # skip 0 ("mask")
        m = golds == c
        if m.any():
            proto[c] = reprs[m].mean(dim=0)
    return proto


def p_knn(reprs, proto, tau):
    """Eq. 4 restricted to the n_classes-1 real labels (index 0 excluded)."""
    dist = pairwise_dist(reprs, proto[1:], metric="euclidean", squared=False)  # (N, C-1)
    w = torch.softmax(-dist / tau, dim=-1)
    out = torch.zeros(reprs.size(0), proto.size(0))
    out[:, 1:] = w
    return out


def evaluate(reprs, golds, probs, proto, tau, lam, labels):
    p_final = lam * probs + (1 - lam) * p_knn(reprs, proto, tau)
    preds = p_final.argmax(dim=-1)
    true_labels = [labels[t] for t in golds.tolist()]
    pred_labels = [labels[p] for p in preds.tolist()]
    metrics, _, _ = calc_classification_metrics(true_labels, pred_labels, labels)
    return metrics


def main():
    task = pubmed_task(train_batch_size=config["batch_size"], max_docs=MAX_DOCS,
                       data_folder=args.tokenized_folder, task_type=args.task)
    device = get_device(0)

    base_dir = f"{args.output_dir}/{args.task}/mind_proto/seed_{args.seed}"
    os.makedirs(base_dir, exist_ok=True)
    result_writer = ResultWriter(f"{base_dir}/0_0_results.jsonl")

    trainer = SentenceClassificationTrainer(device, config, task, result_writer)
    fold = task.get_folds()[0]

    log("=== Mind-Your-Neighbours: training baseline ===")
    best_model = trainer.run_training_for_fold(0, fold, return_best_model=True, path=base_dir)
    best_model.to(device)
    best_model.eval()

    log("=== Building prototype datastore from train set ===")
    train_reprs, train_golds, _ = collect(best_model, fold.train, device, need_probs=False)
    n_classes = len(task.labels)
    proto = build_prototypes(train_reprs, train_golds, n_classes)
    log(f"Prototypes built: {proto.shape} ({(train_golds.unique() > 0).sum().item()} labels with >=1 train example)")

    log("=== Collecting dev/test representations + baseline predictions ===")
    dev_reprs, dev_golds, dev_probs = collect(best_model, fold.dev, device, need_probs=True)
    test_reprs, test_golds, test_probs = collect(best_model, fold.test, device, need_probs=True)

    log("=== Grid search (lambda, tau) on dev, macro-F1 ===")
    lambdas = [round(i * args.lambda_step, 4) for i in range(int(round(1.0 / args.lambda_step)) + 1)]
    best = None
    for tau in args.tau_grid:
        for lam in lambdas:
            m = evaluate(dev_reprs, dev_golds, dev_probs, proto, tau, lam, task.labels)
            if best is None or m["macro-f1"] > best["metrics"]["macro-f1"]:
                best = {"tau": tau, "lambda": lam, "metrics": m}
    log(f"Best on dev: lambda={best['lambda']} tau={best['tau']} macro-F1={best['metrics']['macro-f1']:.4f}")

    dev_metrics = best["metrics"]
    test_metrics = evaluate(test_reprs, test_golds, test_probs, proto, best["tau"], best["lambda"], task.labels)

    with open(f"{base_dir}/best_hparams.json", "w") as f:
        json.dump({"lambda": best["lambda"], "tau": best["tau"], "k": 1,
                   "variant": "single_prototype"}, f, indent=2)

    row = {
        "task": "mind_proto mean",
        "dev weighted-f1": round(dev_metrics["weighted-f1"] * 100, 2),
        "dev accuracy": round(dev_metrics["acc"] * 100, 2),
        "dev macro-f1": round(dev_metrics["macro-f1"] * 100, 2),
        "test weighted-f1": round(test_metrics["weighted-f1"] * 100, 2),
        "test accuracy": round(test_metrics["acc"] * 100, 2),
        "test macro-f1": round(test_metrics["macro-f1"] * 100, 2),
    }
    std_row = {**row, "task": "mind_proto std",
              **{k: 0.0 for k in row if k != "task"}}
    pd.DataFrame([row, std_row]).to_csv(f"{base_dir}/results.csv")

    log(f"Test: weighted-F1={row['test weighted-f1']} macro-F1={row['test macro-f1']} "
        f"acc={row['test accuracy']}  (lambda={best['lambda']}, tau={best['tau']})")
    log(f"=> {base_dir}/results.csv")


if __name__ == "__main__":
    main()
