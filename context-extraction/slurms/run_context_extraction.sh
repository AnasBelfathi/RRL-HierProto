#!/bin/bash
#SBATCH --job-name=run_extraction
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread


set -euo pipefail

# ───────────────────────────────────────────────────────────────
# 🔵 Paramètres explicites
DATASETS="scotus-rhetorical_function"  # <-- ici tu choisis
STRATEGY="mean"  # mean, median, etc.

# ───────────────────────────────────────────────────────────────
# 🔵 Modules
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/python_pkgs_sentence_transformers

# ───────────────────────────────────────────────────────────────
# 🔵 Lancement
for EMB_TYPE in encoder decoder none; do
    echo "═══════════════════════════════════════════════════════════════"
    echo "▶ Traitement Emb_type=${EMB_TYPE} Strategy=${STRATEGY}"
    echo "Datasets : ${DATASETS}"
    echo "═══════════════════════════════════════════════════════════════"

    srun python context-extraction/build_centroids.py \
          --datasets ${DATASETS} \
          --emb_type ${EMB_TYPE} \
          --strategy ${STRATEGY}

    echo ""
done

echo "🎯 Tous les embeddings types traités."