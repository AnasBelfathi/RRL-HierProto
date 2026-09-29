#!/bin/bash
#SBATCH --job-name=rrl-pipeline-22-pbr
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=your_account@v100   # <-- set to your Jean Zay allocation
#SBATCH --mail-type=ALL
#SBATCH --array=0-2%3
# NB: --array above is only the default for a bare `sbatch 22_pbr.slurm`
# (3 seeds). launch.sh overrides it with the right size for $PCM_SEEDS.

# PBR (Prototype-Based Regularization). Unlike PCM, PBR needs no external
# centroids/matching step (00/10) -- it learns its own prototype bank via
# gradient descent, purely as an auxiliary loss on sent_repr. Independent of
# 00/10/20/21, one array task per (task, seed).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    HERE="$SLURM_SUBMIT_DIR/slurms/jeanzay/pipeline"
else
    HERE="$(cd "$(dirname "$0")" && pwd)"
fi
source "$HERE/env.sh" || { echo "FATAL: could not source $HERE/env.sh" >&2; exit 1; }
load_training_modules

SEEDS=($PCM_SEEDS)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="pbr-output-training-${PCM_TASK}"
mkdir -p "$OUT_DIR"

RESULTS_CSV="$OUT_DIR/$PCM_TASK/pbr/seed_$SEED/results.csv"
if [[ -f "$RESULTS_CSV" ]]; then
    echo "▶ PBR $PCM_TASK seed=$SEED -- déjà fait ($RESULTS_CSV), on saute"
    exit 0
fi

echo "▶ PBR $PCM_TASK seed=$SEED (n_prototypes=8, joint, euclidean)"

srun python baseline_run.py \
    --task "$PCM_TASK" \
    --strategy baseline \
    --seed "$SEED" \
    --tokenized_folder "$PCM_TOKENIZED_DIR" \
    --output_dir "$OUT_DIR" \
    --mini_data False \
    --emb_type decoder --centroid_strategy mean \
    --ctx_fusion concat_proj --ctx_position "" \
    --use_crf True --use_sentence_lstm True --use_word_lstm True --use_attention_pooling True \
    --unique_name pbr \
    --use_prototypes True --proto_training joint --n_prototypes 8 \
    --lambda_c 0.9 --lambda_s 0.9 --proto_dist euclidean

echo "✅ Terminé (seed=$SEED) -> $RESULTS_CSV"
