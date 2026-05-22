#!/usr/bin/env python
# full_ssc_llm_simple_filtered.py
#   – sélection de datasets précis
#   – gestion robuste BadRequest 400

import json, time
from pathlib import Path
from typing import List, Dict, Tuple

import tiktoken
import openai
from openai import OpenAI
from tqdm import tqdm

# ---------------- Sélection des datasets ---------------------------------
DATASETS_TO_PROCESS = [
    # "scotus-rhetorical_function",
    # "scotus-category",
    "scotus-steps",

    # ← ajoute / retire ce que tu veux
]

ROOT_DATA_DIR = Path("datasets")
ROOT_SAVE_DIR = Path("document-grouping/embeddings/decoder")
ROOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------------------------

MODEL_NAME     = "text-embedding-3-small"
ENCODING       = tiktoken.encoding_for_model(MODEL_NAME)
TOKENS_LIMIT_PM = 1_000_000        # quota org/minute
REQ_LIMIT_PM    = 3_000            # quota org/minute
BATCH_SIZE      = 50               # inputs / requête
SAFETY_GAP      = 10               # marge sur 8 191

client = OpenAI()

# ---------------- Utils --------------------------------------------------
def trim(text: str) -> Tuple[str, int]:
    ids = ENCODING.encode(text.replace("\n", " "))
    ids = ids[: 8_191 - SAFETY_GAP]
    return ENCODING.decode(ids), len(ids)

def chunk(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embeddings + retry BadRequest 400 et RateLimit 429."""
    while True:
        try:
            return [
                d.embedding
                for d in client.embeddings.create(model=MODEL_NAME, input=texts).data
            ]
        except openai.BadRequestError as e:
            # Souvent causé par un texte vide ou un type invalide
            if len(texts) == 1:
                raise  # texte vraiment illisible → on laissera tomber plus haut
            # On divise le lot et on ré-essaye
            mid = len(texts) // 2
            return (embed_batch(texts[:mid]) + embed_batch(texts[mid:]))
        except openai.RateLimitError as e:
            wait = 2
            msg = e.response.json()["error"]["message"]
            if "try again in" in msg:
                wait = float(msg.split("try again in")[1].split("s")[0]) + 0.2
            print(f"⏳ 429 → pause {wait:.1f}s")
            time.sleep(wait)

# ---------------- Traitement --------------------------------------------
for ds_name in DATASETS_TO_PROCESS:
    dataset_dir = ROOT_DATA_DIR / ds_name
    if not dataset_dir.is_dir():
        print(f"⚠️  {ds_name} n'existe pas, on saute.")
        continue

    print(f"\n📂 Dataset : {ds_name}")
    save_dir = ROOT_SAVE_DIR / ds_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        in_file = dataset_dir / f"{split}.json"
        if not in_file.exists():
            continue

        print(f"  → Split {split}")
        records = json.load(open(in_file, encoding="utf-8"))

        docs = []
        for r in records:
            txt, ntok = trim(r["data"])
            if txt.strip():                       # on ignore les textes vides
                docs.append({"id": r["id"], "text": txt, "n_tokens": ntok})

        tokens_sent = req_sent = 0
        window_start = time.time()

        out_path = save_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fout:

            for batch in tqdm(list(chunk(docs, BATCH_SIZE)),
                              desc=f"{ds_name}/{split}", ncols=80):
                tok_this = sum(d["n_tokens"] for d in batch)

                # ► throttling simple TPM / RPM
                now = time.time()
                if now - window_start >= 60:
                    tokens_sent = req_sent = 0
                    window_start = now

                while (tokens_sent + tok_this > TOKENS_LIMIT_PM or
                       req_sent + 1 > REQ_LIMIT_PM):
                    wait = 60 - (time.time() - window_start) + 0.5
                    print(f"⏳ throttle {wait:.1f}s TPM/RPM")
                    time.sleep(wait)
                    tokens_sent = req_sent = 0
                    window_start = time.time()

                # ► appel embeddings avec gestion 400/429
                embeddings = embed_batch([d["text"] for d in batch])

                tokens_sent += tok_this
                req_sent    += 1

                for d, emb in zip(batch, embeddings):
                    fout.write(json.dumps(
                        {"id": d["id"], "embedding": emb,
                         "n_tokens": d["n_tokens"]},
                        ensure_ascii=False) + "\n")

        print(f"    ✓ Sauvegardé → {out_path}")

print("\n🎉  Datasets choisis encodés sans 400 ni 429.")
