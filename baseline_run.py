import time
import gc
from datetime import datetime
from os import makedirs

import torch

from eval_run import eval_and_save_metrics
from utils import get_device, ResultWriter, log
from task import pubmed_task
from train import SentenceClassificationTrainer
from models import BertHSLN
import os
import random
import numpy as np

# Import conditionnel pour Jean Zay
# import idr_torch
try:
    import idr_torch
    on_jean_zay = True
except ImportError:
    on_jean_zay = False



# UPDATE
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', '1'):
        return True
    elif v.lower() in ('false', 'no', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Ajout des arguments
parser = argparse.ArgumentParser(description="Script de formation pour les modèles de classification de phrases.")
parser.add_argument("--task", type=str, required=True, help="Tâche à tester : category, rhetorical_function ou steps.")
parser.add_argument("--strategy", type=str, required=True, help="Stratégie de contexte (par ex. bm25, random, bertopic).")
parser.add_argument("--seed", type=int, required=True, help="Seed pour contrôler la reproductibilité.")
parser.add_argument("--tokenized_folder", type=str, required=True, help="Chemin vers le dossier contenant les fichiers tokenisés.")
parser.add_argument("--output_dir", type=str, required=True, help="Chemin vers le dossier contenant les fichiers tokenisés.")

# HSLN NN Blocs
parser.add_argument("--use_crf", type=str2bool, default=True, help="Use CRF pour le classification.")
parser.add_argument("--use_sentence_lstm", type=str2bool, default=True, help="Use Sentence Lstm pour context enrichment.")
parser.add_argument("--use_word_lstm", type=str2bool, default=True, help="Use Sentence Lstm pour context enrichment.")
parser.add_argument("--use_attention_pooling", type=str2bool, default=True, help="Use Sentence Lstm pour context enrichment.")
## Additional
parser.add_argument("--windows_size", type=int, default="4", help="Window size")
parser.add_argument("--mini_data", type=str2bool, default=True, help="for testing purposes.")
## For unique repo name
parser.add_argument("--unique_name", type=str, default="None")
parser.add_argument("--emb_type", type=str, default="")
parser.add_argument("--centroid_strategy", type=str, default="")
parser.add_argument("--ctx_fusion", type=str, help="")
parser.add_argument("--ctx_position", type=str, help="")
parser.add_argument("--centroid_path", type=str, help="")


# à voir...
parser.add_argument("--ablation_study", type=str, default="None", help="")

args = parser.parse_args()


# Initialisation de la seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# BERT Model configuration
BERT_MODEL = "models/bert-base-uncased"


# Control the hyperparameters of the model
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
    "batch_size":  32,
    "max_seq_length": 128,
    "max_epochs": 1 if args.mini_data else 10,
    # "max_epochs": 1,
    "early_stopping": 5,


    ## NEW ARGS
    "strategy" : args.strategy,
    "unique_name": args.unique_name,

    ## HSLN BLOCKS
    "use_crf": args.use_crf,
    "use_sentence_lstm":  args.use_sentence_lstm,
    "use_word_lstm":args.use_word_lstm,
    "use_attention_pooling": args.use_attention_pooling,
    "sentence_attention_style": args.unique_name,
    ## Additional
    "window_size" : args.windows_size,
    "centroid_paths" : f"matching-context/{args.centroid_path}/{args.emb_type}/{args.centroid_strategy}/{args.task}",
    "centroid_dim" : 768,
    "ctx_fusion" : args.ctx_fusion,
    "ctx_position" : args.ctx_position,


}

# MAX_DOCS = -1
MAX_DOCS = 2 if args.mini_data else -1
#
def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS, data_folder=args.tokenized_folder, task_type=args.task)

# UPDATE


def create_generic_task(task_name):
    return generic_task(task_name, train_batch_size=config["batch_size"], max_docs=MAX_DOCS)



task = create_task(pubmed_task)

# ADAPT: Set to False if you do not want to save the best model
save_best_models = True


# Configuration pour Jean Zay (si détecté)
if on_jean_zay:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Initialisation des variables d'environnement
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=idr_torch.size,
        rank=idr_torch.rank
    )

    torch.cuda.set_device(idr_torch.local_rank)
    device = torch.device("cuda")
else:
    # Configuration pour l'exécution en local
    device = get_device(0)


# strategy = args.strategy if args.ablation_study == "None" else "ablation_study"
# sentence_attention_style =  args.sentence_attention_style if args.ablation_study == "None" else args.ablation_study


# Définition des répertoires de sauvegarde des résultats

if args.unique_name == "None":
    auto_name = f"{args.emb_type}_{args.centroid_strategy}_" \
                f"{args.ctx_position}_{args.ctx_fusion}"
else:
    auto_name = args.unique_name

# base_dir = f"{args.output_dir}/{args.task}/{args.strategy}-WS_{args.windows_size}-CRF_{args.use_crf}-SentenceLSTM_{args.use_sentence_lstm}/seed_{args.seed}"
base_dir = f"{args.output_dir}/{args.task}/{auto_name}/seed_{args.seed}"

makedirs(base_dir, exist_ok=True)

# preload data if not already done
task.get_folds()

log(f"Début de l'entraînement pour la tâche : {args.task} avec la stratégie de contexte : {args.strategy} et seed : {args.seed}")


restarts = 1 if task.num_folds == 1 else 1
for restart in range(restarts):
    for fold_num, fold in enumerate(task.get_folds()):
        start = time.time()
        result_writer = ResultWriter(f"{base_dir}/{restart}_{fold_num}_results.jsonl")
        result_writer.log(f"Fold {fold_num} sur {task.num_folds}")
        result_writer.log(f"Début de l'entraînement pour le fold {fold_num}...")

        trainer = SentenceClassificationTrainer(device, config, task, result_writer)
        best_model = trainer.run_training_for_fold(fold_num, fold, return_best_model=True, path=base_dir)
        # if best_model is not None:
        #     model_path = os.path.join(base_dir, f"{restart}_{fold_num}_model.pt")
        #     result_writer.log(f"Sauvegarde du meilleur modèle dans {model_path}")
        #     torch.save(best_model.state_dict(), model_path)

        result_writer.log(f"finished training {restart} for fold {fold_num}: {time.time() - start}")

        # explicitly call garbage collector so that CUDA memory is released
        gc.collect()

log("Training finished.")

log("Calculating metrics...")
eval_and_save_metrics(base_dir, args.task, args.tokenized_folder)
log("Calculating metrics finished")