#!/usr/bin/env python
# full_ssc_gpu_save_all.py

import json, time
from pathlib import Path
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------- Configuration ----------------------------------------
ROOT_DATA_DIR = Path("datasets")
MODEL_PATH = "models/nomic-embed-text-v1"
ROOT_SAVE_DIR = Path("document-grouping/embeddings/encoder")
ROOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------------------------------------------------

# --- GPU Detection -----------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"[INFO] GPU détecté : {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Utilisation du CPU")

# --- Load model on GPU -------------------------------------------------
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
model.to(DEVICE)

# --- Traitement --------------------------------------------------------
for dataset_dir in sorted(ROOT_DATA_DIR.iterdir()):
    if dataset_dir.is_dir():
        print(f"\n📂 Traitement du dataset : {dataset_dir.name}")
        save_dir = ROOT_SAVE_DIR / f"{dataset_dir.name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "dev", "test"):
            input_file = dataset_dir / f"{split}.json"
            if not input_file.exists():
                print(f"⚠️ Fichier {split}.json manquant dans {dataset_dir.name}, passage...")
                continue

            print(f"\n[INFO] Traitement du split : {split}")
            with open(input_file, encoding="utf-8") as f:
                raw_data = json.load(f)

            results = []  # Liste de dictionnaires

            for ex in tqdm(raw_data, desc=f"→ {split} ({dataset_dir.name})", ncols=80):
                doc_id = ex["id"]
                text   = ex["data"]

                sentence = [f"clustering: {text}"]

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                emb = model.encode(
                    sentence,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    device=DEVICE
                )

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                record = {
                    "id": doc_id,
                    "embedding": emb.cpu(),              # reste en Tensor si torch.save
                    "n_tokens": len(model.tokenizer.encode(text)),
                    "elapsed_time": dt
                }
                results.append(record)

                tqdm.write(f"{doc_id} | {tuple(emb.shape)} | {dt:.3f}s")

            # --- Sauvegarde torch -----------------------------------------------
            # torch_file = save_dir / f"{split}.pt"
            # torch.save(results, torch_file)

            # --- Sauvegarde JSONL indexable -------------------------------------
            jsonl_file = save_dir / f"{split}.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for r in results:
                    out = {
                        "id": r["id"],
                        "embedding": r["embedding"].tolist(),
                        "n_tokens": r["n_tokens"],
                        "elapsed_time": r["elapsed_time"]
                    }
                    f.write(json.dumps(out) + "\n")

            print(f"[INFO] {split} terminé pour {dataset_dir.name}. Sauvé dans {jsonl_file}")

print("\n✅ Tous les datasets et splits traités et sauvegardés.")
