#!/usr/bin/env python
"""
Associe chaque phrase à son centroïde le plus proche
OU à celui correspondant au vrai label (expérience oracle),
et sauvegarde dans similarity_outputs/<emb_type>/<strategy>/<dataset>/<split>.jsonl
"""

import json, argparse, os, logging
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

from utils_matching import (
    load_centroids, load_doc_clusters,
    load_bert, embed_sentence,
    closest_centroid, weighted_prototype_vector
)

LEGAL_DATASETS = {
    "scotus-category", "scotus-rhetorical_function",
    "scotus-steps", "legal-eval", "DeepRhole"
}
MODEL_LEGAL = "models/legal-bert-base-uncased"
MODEL_SCI   = "models/scibert_scivocab_uncased"


def get_doc2cluster(ds: str, split: str, emb_type: str) -> Dict[str, int]:
    if emb_type == "none":
        return defaultdict(lambda: 0)

    root = Path("document-grouping/document-groupe") / emb_type / ds
    file_split = root / f"{split}.jsonl"
    file_train = root / "train.jsonl"

    if file_split.exists():
        return load_doc_clusters(file_split)
    if file_train.exists():
        logging.warning(f"[{ds}/{split}] {file_split.name} absent – fallback train.jsonl")
        return load_doc_clusters(file_train)

    logging.warning(f"[{ds}] mapping cluster introuvable – cluster 0 appliqué")
    return defaultdict(lambda: 0)


def process_split(dataset: str, split: str,
                  json_path: Path,
                  out_path: Path,
                  centroids_bank,
                  doc2cluster,
                  tokenizer, model,
                  max_docs: Optional[int] = None,
                  use_weighted=False,
                  use_oracle=False):

    with json_path.open() as f:
        docs = json.load(f)
    if max_docs:
        docs = docs[:max_docs]

    rows = []
    available_clusters = list(centroids_bank.keys())
    default_cluster = available_clusters[0]
    missing_warned = set()

    for doc in tqdm(docs, desc=f"{dataset}/{split}", ncols=80):
        doc_id = str(doc["id"])
        clu_id = doc2cluster.get(doc_id)

        if clu_id is None or clu_id not in centroids_bank:
            if clu_id not in missing_warned:
                logging.warning(
                    f"[{dataset}/{split}] doc {doc_id} → cluster {clu_id} absent. "
                    f"Fallback cluster {default_cluster}"
                )
                missing_warned.add(clu_id)
            clu_id = default_cluster

        if clu_id in centroids_bank:
            labels, mat = centroids_bank[clu_id]
        elif 0 in centroids_bank:
            if clu_id not in missing_warned:
                logging.warning(f"[{dataset}/{split}] cluster {clu_id} absent – fallback cluster 0")
                missing_warned.add(clu_id)
            labels, mat = centroids_bank[0]
        else:
            if clu_id not in missing_warned:
                logging.warning(f"[{dataset}/{split}] cluster {clu_id} absent – fallback cluster {default_cluster}")
                missing_warned.add(clu_id)
            labels, mat = centroids_bank[default_cluster]

        for ann in doc.get("annotations", []):
            for idx, res in enumerate(ann.get("result", [])):
                text = res["value"]["text"].strip()
                true_lab = res["value"]["labels"][0] if res["value"]["labels"] else "UNKNOWN"

                vec = embed_sentence(text, tokenizer, model)


                # print(f"Oracle: {use_oracle}")
                # print(f"true_label: {true_lab}")
                if use_oracle:
                    if true_lab in labels:
                        centroid_idx = labels.index(true_lab)
                        out_vec = mat[centroid_idx]
                        nearest_lab = true_lab
                    else:
                        nearest_lab, out_vec = closest_centroid(vec, mat, labels)
                elif use_weighted:
                    weighted_vec = weighted_prototype_vector(vec, mat)
                    nearest_lab, _ = closest_centroid(vec, mat, labels)
                    out_vec = weighted_vec
                else:
                    nearest_lab, out_vec = closest_centroid(vec, mat, labels)

                rows.append({
                    "doc_id": doc_id,
                    "cluster": clu_id,
                    "sentence_idx": idx,
                    "sentence_text": text,
                    "true_label": true_lab,
                    "closest_centroid_label": nearest_lab,
                    "centroid_vector": out_vec.tolist()
                })

    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--centroids_dir", required=True)
    p.add_argument("--emb_type", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--out_root", default="similarity_outputs")
    p.add_argument("--max_docs", type=int)
    p.add_argument("--multi_proto", action="store_true",
                   help="Utilise une représentation pondérée multi-prototypes")
    p.add_argument("--oracle", action="store_true",
                   help="Utilise le centroïde du vrai label (expérience oracle)")

    args = p.parse_args()

    centroids_path = Path(args.centroids_dir) / "centroids.jsonl"
    if not centroids_path.exists():
        raise FileNotFoundError(centroids_path)

    centroids_bank = load_centroids(centroids_path)
    model_path = MODEL_LEGAL if args.dataset in LEGAL_DATASETS else MODEL_SCI
    tokenizer, model = load_bert(model_path)

    for split in ("train", "dev", "test"):
        inp_json  = Path(f"ssc-datasets/{args.dataset}/{split}.json")
        out_jsonl = (Path(args.out_root) / args.emb_type / args.strategy /
                     args.dataset / f"{split}.jsonl")

        doc2clu = get_doc2cluster(args.dataset, split, args.emb_type)

        process_split(args.dataset, split,
                      inp_json, out_jsonl,
                      centroids_bank, doc2clu,
                      tokenizer, model,
                      max_docs=args.max_docs,
                      use_weighted=args.multi_proto,
                      use_oracle=args.oracle)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s - %(message)s")
    main()
