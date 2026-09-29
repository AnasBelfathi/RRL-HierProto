"""
Run this ONCE on a Jean Zay FRONTEND node (login node, has internet) -- NOT
inside srun/sbatch, compute nodes have none (see the "Network is unreachable"
error when build_centroids.py/baseline_run.py try to resolve these ids).

    cd $WORK/rrl-prototype-methods
    module load anaconda-py3/2024.06 && conda activate my_new_env
    python slurms/jeanzay/pipeline/download_models.py

Uses snapshot_download (simpler/lighter than AutoModel.from_pretrained().save_pretrained():
no need to instantiate the model just to cache its weights), but with
allow_patterns so only the pytorch weights + tokenizer/config files are
fetched -- not the tf/flax/onnx/rust duplicates that make the *local*
models/legal-bert-base-uncased folder ~1.4GB instead of ~440MB. Saves under
models/<name>, matching the relative paths baseline_run.py / build_centroids.py
/ match_centroids.py already expect (BERT_MODEL = "models/bert-base-uncased",
MODEL_LEGAL = "models/legal-bert-base-uncased").
"""
import os
from huggingface_hub import snapshot_download

MODELS = ["bert-base-uncased", "nlpaueb/legal-bert-base-uncased"]

ALLOW_PATTERNS = [
    "*.json",           # config.json, tokenizer_config.json, special_tokens_map.json, tokenizer.json
    "vocab.txt",
    "merges.txt",        # harmless if absent (BERT tokenizers don't have one)
    "*.safetensors",
    "pytorch_model.bin",
]

for model_id in MODELS:
    # legal-bert-base-uncased is saved locally as "legal-bert-base-uncased"
    # (dropping the "nlpaueb/" org prefix) to match the hardcoded relative
    # paths used throughout this codebase.
    local_name = model_id.split("/")[-1]
    out_dir = os.path.join("models", local_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {model_id} -> {out_dir}")
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=out_dir,
        allow_patterns=ALLOW_PATTERNS,
    )

print("Done.")
