import json, random, argparse, logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, AutoModel

from aggregators import STRATEGIES

ROOT_DATASETS = Path("datasets")
ROOT_EMB = Path("document-grouping/document-groupe")
ROOT_OUT = Path("context-extraction/prototypes-without-labels-fv")
ROOT_OUT_CLUSTERS = Path("context-extraction/clustered-sentences-without-labels-fv")
ROOT_OUT.mkdir(parents=True, exist_ok=True)
ROOT_OUT_CLUSTERS.mkdir(parents=True, exist_ok=True)

LEGAL_DATASETS = {
    "scotus-category", "scotus-rhetorical_function", "scotus-steps", "legal-eval", "DeepRhole"
}

MODEL_LEGAL = "models/legal-bert-base-uncased"
MODEL_SCI = "models/scibert_scivocab_uncased"

MINI_N = 100
MAX_LEN = 512
BATCH_SIZE = 16
RANDOM_STATE = 42

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Using CPU")

def load_doc_clusters(train_jsonl: Path) -> Dict[str, int]:
    mapping = {}
    with train_jsonl.open() as f:
        for line in f:
            obj = json.loads(line)
            mapping[str(obj["id"])] = int(obj["cluster"])
    return mapping

def load_doc_sentences(train_json: Path) -> Dict[str, List]:
    out = defaultdict(list)
    for doc in json.load(train_json.open()):
        doc_id = str(doc["id"])
        for ann in doc.get("annotations", []):
            for res in ann.get("result", []):
                lbls = res["value"]["labels"]
                if not lbls:
                    continue
                label = lbls[0].upper()
                text = res["value"]["text"].strip()
                if text:
                    out[doc_id].append((label, text))
    return out

def encode_batch(model, tokenizer, texts: List[str]) -> np.ndarray:
    toks = tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LEN)
    toks = {k: v.to(DEVICE) for k, v in toks.items()}
    with torch.no_grad():
        vec = model(**toks).last_hidden_state[:, 0, :]
    return vec.cpu().numpy()

def get_labels(dataset: str) -> List[str]:
    label_set = set()
    with (ROOT_DATASETS / dataset / "train.json").open() as f:
        for doc in json.load(f):
            for ann in doc.get("annotations", []):
                for res in ann.get("result", []):
                    lbls = res["value"]["labels"]
                    if lbls:
                        label_set.add(lbls[0].upper())
    return sorted(label_set)

def cluster_sentences(vecs: np.ndarray, k: int) -> List[int]:
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto", batch_size=1024)
    return km.fit_predict(vecs)

def save_cluster_metadata(dataset, emb_type, strategy, use_labels, metadata):
    label_tag = "with_labels" if use_labels else "no_labels"
    filename = f"{dataset}__{emb_type}__{strategy}__{label_tag}.jsonl"
    path = ROOT_OUT_CLUSTERS / filename
    with path.open("w") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ... (imports and constants remain unchanged)

def build_centroids(dataset: str, emb_type: str, strategy: str, mini_data: bool, use_labels: bool):
    ssc_train = ROOT_DATASETS / dataset / "train.json"
    if not ssc_train.exists():
        logging.warning(f"{ssc_train} missing"); return []

    doc2sent = load_doc_sentences(ssc_train)
    model_name = MODEL_LEGAL if dataset in LEGAL_DATASETS else MODEL_SCI
    logging.info(f"[{dataset}] Loading model {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    k = len(get_labels(dataset))
    agg_fn = STRATEGIES[strategy]
    metadata = []
    centroids = []

    if emb_type == "none":
        all_texts = []
        doc_sent_refs = []
        for doc_id, sent_list in doc2sent.items():
            for sid, (label, text) in enumerate(sent_list):
                all_texts.append(text)
                doc_sent_refs.append((doc_id, sid, label))
        if mini_data and len(all_texts) > MINI_N:
            random.seed(RANDOM_STATE)
            sampled = random.sample(list(zip(all_texts, doc_sent_refs)), MINI_N)
            all_texts, doc_sent_refs = zip(*sampled)
        vecs = []
        for i in range(0, len(all_texts), BATCH_SIZE):
            vecs.append(encode_batch(mdl, tok, all_texts[i:i+BATCH_SIZE]))
        vecs = np.vstack(vecs)
        if use_labels:
            for (doc_id, sid, label), _ in zip(doc_sent_refs, vecs):
                metadata.append({
                    "doc_id": doc_id,
                    "sentence_id": sid,
                    "label": label,
                    "cluster": "UNKNOWN",
                    "document_cluster": 0
                })
            groups = defaultdict(list)
            for (_, _, label), vec in zip(doc_sent_refs, vecs):
                groups[label].append(vec)
            for label, group_vecs in groups.items():
                centroids.append({
                    "cluster": 0,
                    "class": label,
                    "strategy": strategy,
                    "vector": agg_fn(group_vecs).tolist(),
                    "count": len(group_vecs)
                })
        else:
            cluster_ids = cluster_sentences(vecs, k)
            for (doc_id, sid, _), cid in zip(doc_sent_refs, cluster_ids):
                metadata.append({
                    "doc_id": doc_id,
                    "sentence_id": sid,
                    "label": "UNKNOWN",
                    "cluster": int(cid),
                    "document_cluster": 0
                })
            groups = defaultdict(list)
            for cid, vec in zip(cluster_ids, vecs):
                groups[cid].append(vec)
            for cid, group_vecs in groups.items():
                centroids.append({
                    "cluster": 0,
                    "class": str(cid),
                    "strategy": strategy,
                    "vector": agg_fn(group_vecs).tolist(),
                    "count": len(group_vecs)
                })

    else:
        emb_file = ROOT_EMB / emb_type / dataset / "train.jsonl"
        if not emb_file.exists():
            logging.warning(f"{emb_file} missing"); return []

        doc2cluster = load_doc_clusters(emb_file)
        if use_labels:
            triples = [(doc2cluster[doc_id], label, text)
                       for doc_id, sent_list in doc2sent.items()
                       if doc_id in doc2cluster
                       for label, text in sent_list]
            groups = defaultdict(list)
            for clu, lab, txt in tqdm(triples, desc=f"{dataset} encoding", ncols=80):
                groups[(clu, lab)].append(txt)
            for (clu, lab), texts in groups.items():
                vecs = []
                for i in range(0, len(texts), BATCH_SIZE):
                    vecs.append(encode_batch(mdl, tok, texts[i:i+BATCH_SIZE]))
                vecs = np.vstack(vecs)
                centroids.append({
                    "cluster": int(clu),
                    "class": lab,
                    "strategy": strategy,
                    "vector": agg_fn(vecs).tolist(),
                    "count": len(vecs)
                })
                # Metadata uniquement pour use_labels = True
                for doc_id, sent_list in doc2sent.items():
                    if doc_id not in doc2cluster:
                        continue
                    for sid, (label, _) in enumerate(sent_list):
                        metadata.append({
                            "doc_id": doc_id,
                            "sentence_id": sid,
                            "label": label,
                            "cluster": "UNKNOWN",
                            "document_cluster": doc2cluster[doc_id]
                        })


        else:

            # Cas : emb_type != "none" et use_labels == False

            # On applique un clustering de phrases à l’intérieur de chaque cluster de document

            cluster_texts = defaultdict(list)  # map: doc_cluster_id -> list of sentence texts

            sent_refs = defaultdict(list)  # map: doc_cluster_id -> list of (doc_id, sid)

            for doc_id, sent_list in doc2sent.items():

                if doc_id in doc2cluster:

                    doc_cluster = doc2cluster[doc_id]  # le cluster du document

                    for sid, (_, text) in enumerate(sent_list):
                        cluster_texts[doc_cluster].append(text)

                        sent_refs[doc_cluster].append((doc_id, sid))

            for doc_cluster_id, texts in cluster_texts.items():

                if mini_data and len(texts) > MINI_N:
                    random.seed(RANDOM_STATE)

                    zipped = list(zip(texts, sent_refs[doc_cluster_id]))

                    sampled = random.sample(zipped, MINI_N)

                    texts, sent_refs[doc_cluster_id] = zip(*sampled)

                # Encodage

                vecs = []

                for i in range(0, len(texts), BATCH_SIZE):
                    vecs.append(encode_batch(mdl, tok, texts[i:i + BATCH_SIZE]))

                vecs = np.vstack(vecs)

                # Clustering de phrases à l'intérieur du document_cluster

                sentence_cluster_ids = cluster_sentences(vecs, k)

                # Sauvegarde du metadata avec:

                # - cluster : ID de cluster de phrase

                # - document_cluster : ID de cluster de document

                for (doc_id, sid), phrase_cluster_id in zip(sent_refs[doc_cluster_id], sentence_cluster_ids):
                    metadata.append({

                        "doc_id": doc_id,

                        "sentence_id": sid,

                        "label": "UNKNOWN",

                        "cluster": int(phrase_cluster_id),  # phrase-level cluster

                        "document_cluster": int(doc_cluster_id)  # document-level cluster

                    })

                # Agrégation pour les centroides :

                # - cluster = ID du document_cluster

                # - class = ID du cluster de phrases à l’intérieur de ce document_cluster

                groups = defaultdict(list)

                for phrase_cluster_id, vec in zip(sentence_cluster_ids, vecs):
                    groups[phrase_cluster_id].append(vec)

                for phrase_cluster_id, group_vecs in groups.items():
                    centroids.append({

                        "cluster": int(doc_cluster_id),  # document-level cluster

                        "class": str(phrase_cluster_id),  # phrase-level cluster (comme label de la classe)

                        "strategy": strategy,

                        "vector": agg_fn(group_vecs).tolist(),

                        "count": len(group_vecs)

                    })

    save_cluster_metadata(dataset, emb_type, strategy, use_labels, metadata)
    return centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--emb_type", required=True)
    ap.add_argument("--strategy", default="mean", choices=STRATEGIES.keys())
    ap.add_argument("--mini_data", action="store_true")
    ap.add_argument("--use_labels", action="store_true")
    args = ap.parse_args()

    subdir = "with_labels" if args.use_labels else "no_labels"
    dir_name = args.emb_type if args.emb_type != "none" else "no_cluster"
    base_out = ROOT_OUT / dir_name / args.strategy
    base_out.mkdir(parents=True, exist_ok=True)

    for ds in tqdm(args.datasets, desc="Datasets", ncols=80):
        logging.info(f"=== {ds} ({args.emb_type}/{args.strategy}/{subdir}) ===")
        cents = build_centroids(ds, args.emb_type, args.strategy, args.mini_data, args.use_labels)
        out_dir = base_out / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "centroids.jsonl").open("w") as f:
            for c in cents:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        logging.info(f"✓ Metadata and centroids written for {ds}")

    logging.info("Done.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    main()
