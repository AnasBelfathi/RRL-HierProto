#!/usr/bin/env python
# kmeans_k5_bi_encoder_decoder.py
# -----------------------------------------------------------
# besoin : pip install scikit-learn numpy tqdm

import json, numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

# ------------ PARAMÈTRES ---------------------------------------------
DATASETS  = [
             "PubMed_20k_RCT",
    "csabstracts"
            ]

EMB_ROOT  = Path("document-grouping/embeddings")     # source
OUT_ROOT  = Path("document-grouping/groupe-embeddings-v2")  # cible
OUT_ROOT.mkdir(parents=True, exist_ok=True)

K = 8                      # K-means fixe
RANDOM_STATE = 42
N_INIT = "auto"
BATCH  = 1024
# ----------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            vec = obj["embedding"]
            if isinstance(vec[0], list):        # cas encoder [[…]]
                vec = vec[0]
            obj["embedding"] = vec
            data.append(obj)
    return data

def cluster_dataset(ds: str, emb_type: str):
    src_dir = EMB_ROOT / emb_type / ds
    if not src_dir.is_dir():
        print(f"⚠️  {ds} absent dans {emb_type}")
        return

    print(f"\n📂 {ds}  ({emb_type})")
    splits = {}
    for sp in ("train", "dev", "test"):
        f = src_dir / f"{sp}.jsonl"
        if f.exists():
            splits[sp] = load_jsonl(f)

    if "train" not in splits:
        print("   (split train manquant)"); return

    # === CLUSTERING SUR LE TRAIN UNIQUEMENT ===
    X_train = np.array([r["embedding"] for r in splits["train"]], dtype=np.float32)
    km = MiniBatchKMeans(n_clusters=K, random_state=RANDOM_STATE,
                         n_init=N_INIT, batch_size=BATCH)
    train_labels = km.fit_predict(X_train)
    centroids = km.cluster_centers_

    # assignation des clusters au train
    for rec, lab in zip(splits["train"], train_labels):
        rec["cluster"] = int(lab)

    # === AFFECTATION DES CLUSTERS AUX AUTRES SPLITS PAR SIMILARITÉ ===
    for sp in ("dev", "test"):
        if sp not in splits:
            continue
        for rec in splits[sp]:
            vec = np.array(rec["embedding"]).reshape(1, -1)
            sims = cosine_similarity(vec, centroids)[0]
            best_cluster = int(np.argmax(sims))
            rec["cluster"] = best_cluster

    # ---------- sauvegarde ----------
    dst_dir = OUT_ROOT / emb_type / ds
    dst_dir.mkdir(parents=True, exist_ok=True)
    for sp, records in splits.items():
        out = dst_dir / f"{sp}.jsonl"
        with out.open("w") as fo:
            for r in records:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   ✓ {sp} → {out}")

# ---------------- MAIN -------------------------------------------------
if __name__ == "__main__":
    for emb_type in ("encoder", "decoder"):
        for ds in DATASETS:
            cluster_dataset(ds, emb_type)

    print(f"\n🎉  K-means (K={K}) appliqué au split train uniquement, clusters affectés pour dev/test par similarité.")
