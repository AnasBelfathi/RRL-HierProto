#!/usr/bin/env bash
#SBATCH --job-name=match_centroids
#SBATCH --output=./job_out_err/%x_%A.out
#SBATCH --error=./job_out_err/%x_%A.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --account=bvh@v100


set -euo pipefail

# ───────────────────────────────────────────────────────────────
# 🔵 Modules
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/python_pkgs_sentence_transformers

# ───────────────────────────────────────────────────────────────
# 1. Choix des DATASETS, EMB_TYPES et STRATEGIES à traiter
DATASETS=("scotus-rhetorical_function" "scotus-category" "scotus-steps" "legal-eval" "DeepRhole" "PubMed_20k_RCT" "csabstracts") # "biorc" "scotus-rhetorical_function" "scotus-category" "scotus-steps" "legal-eval" "DeepRhole" "PubMed_20k_RCT" "csabstracts"
EMB_TYPES=("none" "decoder" "encoder")   # none = pas de clusters
STRATEGIES=("mean")                      # rajoute "median" si besoin
OUT_ROOT="matching-context/similarity_outputs_multiprototypes"
# Nombre max de docs par split (debug) ; mettre vide pour complet
MAX_DOCS=                                # ex: MAX_DOCS=500

# ───────────────────────────────────────────────────────────────
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/python_pkgs_sentence_transformers

# ───────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  for EMB_TYPE in "${EMB_TYPES[@]}"; do
    for STRATEGY in "${STRATEGIES[@]}"; do

      echo "═══════════════ ${DATASET} | ${EMB_TYPE} | ${STRATEGY} ═══════════════"

      # Si emb_type=none → on n'a pas besoin de centroids_dir variable
      if [[ "${EMB_TYPE}" == "none" ]]; then
        CENTROIDS_DIR="context-extraction/centroids/no_cluster/${STRATEGY}/${DATASET}"
      else
        CENTROIDS_DIR="context-extraction/centroids/${EMB_TYPE}/${STRATEGY}/${DATASET}"
      fi

      # Skip si le fichier centroids n'existe pas
      if [[ ! -f "${CENTROIDS_DIR}/centroids.jsonl" ]]; then
        echo "⚠️  ${CENTROIDS_DIR}/centroids.jsonl introuvable – on saute."
        continue
      fi

      srun --cpu_bind=none \
           python matching-context/match_centroids.py \
              --dataset "${DATASET}" \
              --centroids_dir "${CENTROIDS_DIR}" \
              --emb_type "${EMB_TYPE}" \
              --strategy "${STRATEGY}" \
              --out_root "${OUT_ROOT}" \
              ${MAX_DOCS:+--max_docs ${MAX_DOCS}} \
              --multi_proto

      echo ""
    done
  done
done

echo "🎉 Tous les jobs terminés."
