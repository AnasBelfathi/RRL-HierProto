#!/bin/bash
#SBATCH --job-name=unique_with_labels
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH -C v100-32g  # Type de GPU (ici V100 avec 32GB)
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-34%35  # Ajusté pour inclure toutes les combinaisons des scénarios et seeds

set -e

# 🛠 Load required modules
module purge
#module load cpuarch/amd
module load anaconda-py3/2024.06
conda activate my_new_env

# ───────────────────────── grilles ──────────────────────────
TASKS=(scotus-rhetorical_function scotus-category scotus-steps legal-eval DeepRhole PubMed_20k_RCT csabstracts)
EMB_TYPES=(none)
CENT_STRATS=(mean)
CTX_FUSIONS=(cross_attn concat_proj film gated_add cln)
CTX_POSITIONS=(pre)      # "" = pas d’injection
SEEDS=(1)

TOKENIZED_DIR="processed-datasets"
OUT_DIR="injection_strategies_output"
MINI_DATA="FALSE"
CENTROID_PATH="unique_proto_with_labels"

mkdir -p "$OUT_DIR"

# ─────────── 1. liste exhaustive des combos ────────────────
COMBOS=()
for task in "${TASKS[@]}";        do
  for emb  in "${EMB_TYPES[@]}";   do
    for cst in "${CENT_STRATS[@]}";do
      for fus in "${CTX_FUSIONS[@]}";do
        for pos in "${CTX_POSITIONS[@]}";do
          for sd  in "${SEEDS[@]}";   do
            COMBOS+=("$task|$emb|$cst|$fus|$pos|$sd")
          done
        done
      done
    done
  done
done

TOTAL=${#COMBOS[@]}

# 🚨 pense à mettre --array=0-$(($TOTAL-1))%40 dans l’en-tête !
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "Index $SLURM_ARRAY_TASK_ID hors limite (TOTAL=$TOTAL)"; exit 1;
fi

IFS='|' read TASK EMB CSTR FUS POS SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

echo "▶ combo #$SLURM_ARRAY_TASK_ID / $TOTAL  →  $TASK | $EMB/$CSTR | $FUS-$POS | seed=$SEED"

# unique_name seulement si POS == ""
UNAME_ARG=()
[[ -z "$POS" ]] && UNAME_ARG=(--unique_name full)

# ─────────── 2. exécution ──────────────────────────────────
srun python baseline_run.py \
     --task "$TASK" \
     --strategy baseline \
     --seed "$SEED" \
     --tokenized_folder "$TOKENIZED_DIR" \
     --output_dir "$OUT_DIR" \
     --mini_data "$MINI_DATA" \
     --emb_type "$EMB" \
     --centroid_strategy "$CSTR" \
     --ctx_fusion "$FUS" \
     --ctx_position "$POS" \
     --use_crf True \
     --use_sentence_lstm True \
     --use_word_lstm True \
     --use_attention_pooling True \
     --centroid_path "$CENTROID_PATH"
     "${UNAME_ARG[@]}"

echo "✅ Terminé"
