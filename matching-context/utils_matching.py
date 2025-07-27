"""
Utilitaires : chargement des centroïdes, des clusters et encodage BERT
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────────────────────
# Centroïdes
# -------------------------------------------------------------------
def load_centroids(path: Path) -> Dict[int, Tuple[List[str], np.ndarray]]:
    """
    Retourne un dict :
      cluster_id → (list[labels], array[n_centroids, d])
    """
    bank: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            clu   = int(obj["cluster"])
            label = obj["class"]
            vec   = np.array(obj["vector"], dtype=np.float32)
            bank.setdefault(clu, []).append((label, vec))

    # Transformer valeurs en (labels, matrix)
    out = {}
    for clu, pairs in bank.items():
        labels, vecs = zip(*pairs)
        out[clu] = (list(labels), np.stack(vecs))
    return out


# ────────────────────────────────────────────────────────────────────
# Mapping doc_id → cluster (encoder / decoder)
# -------------------------------------------------------------------
def load_doc_clusters(train_jsonl: Path) -> Dict[str, int]:
    mapping = {}
    with train_jsonl.open() as f:
        for line in f:
            obj = json.loads(line)
            mapping[str(obj["id"])] = int(obj["cluster"])
    return mapping


# ────────────────────────────────────────────────────────────────────
# Encodage BERT
# -------------------------------------------------------------------
def load_bert(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
    return tok, mdl


def embed_sentence(sentence: str, tokenizer, model) -> np.ndarray:
    toks = tokenizer(sentence, return_tensors="pt",
                     truncation=True, max_length=512, padding=True)
    toks = {k: v.to(DEVICE) for k, v in toks.items()}
    with torch.no_grad():
        vec = model(**toks).last_hidden_state[:, 0, :]   # CLS
    return vec.squeeze(0).cpu().numpy()


# ────────────────────────────────────────────────────────────────────
# Similarité
# -------------------------------------------------------------------
# … imports et fonctions inchangés …

def closest_centroid(
    vec: np.ndarray,
    centroids_mat: np.ndarray,
    labels: List[str]
) -> Tuple[str, np.ndarray]:       # <─ utiliser Tuple[…]
    sims = cosine_similarity(vec.reshape(1, -1), centroids_mat)[0]
    idx  = int(np.argmax(sims))
    return labels[idx], centroids_mat[idx]

def weighted_prototype_vector(
    vec: np.ndarray,
    centroids_mat: np.ndarray
) -> np.ndarray:
    """
    Combine plusieurs prototypes pondérés par leur similarité à `vec`.
    """
    sims = cosine_similarity(vec.reshape(1, -1), centroids_mat)[0]  # (n_centroids,)
    weights = sims / sims.sum()  # normalisation
    return np.average(centroids_mat, axis=0, weights=weights)
