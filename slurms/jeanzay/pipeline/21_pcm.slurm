#!/bin/bash
#SBATCH --job-name=rrl-pipeline-21-pcm
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=your_account@v100   # <-- set to your Jean Zay allocation
#SBATCH --mail-type=ALL
#SBATCH --array=0-8%9
# NB: --array above is only the default for a bare `sbatch 21_pcm.slurm`
# (3 strategies x 3 seeds). launch.sh overrides it with the right size.

# HSLN + PCM (concat_proj fusion, injected pre-LSTM), one array task per
# (emb_type, seed) combo, task/seeds coming from $PCM_TASK/$PCM_SEEDS.
# Requires 00_build_centroids.slurm + 10_match_centroids.slurm to have run first.
# $0 is unreliable under sbatch on this cluster (Slurm runs a spooled copy of
# the script, e.g. /var/spool/slurmd/jobXXXX/slurm_script) -- fall back to
# $SLURM_SUBMIT_DIR (the cwd `sbatch` was invoked from, set by launch.sh)
# for the job case, and to dirname "$0" for direct/interactive execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    HERE="$SLURM_SUBMIT_DIR/slurms/jeanzay/pipeline"
else
    HERE="$(cd "$(dirname "$0")" && pwd)"
fi
source "$HERE/env.sh" || { echo "FATAL: could not source $HERE/env.sh" >&2; exit 1; }
load_training_modules

SEEDS=($PCM_SEEDS)
COMBOS=()
for EMB in "${PCM_EMB_TYPES[@]}"; do
    for SD in "${SEEDS[@]}"; do
        COMBOS+=("$EMB|$SD")
    done
done
IFS='|' read EMB SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "$PCM_MATCH_OUT_ROOT/$EMB/mean/$PCM_TASK/train.jsonl" ]]; then
    echo "❌ $PCM_MATCH_OUT_ROOT/$EMB/mean/$PCM_TASK/train.jsonl introuvable -- lancez 00_build_centroids.slurm puis 10_match_centroids.slurm d'abord." >&2
    exit 1
fi

OUT_DIR="new-output-training-${PCM_TASK}"
mkdir -p "$OUT_DIR"

RESULTS_CSV="$OUT_DIR/$PCM_TASK/${EMB}_mean_pre_concat_proj/seed_$SEED/results.csv"
if [[ -f "$RESULTS_CSV" ]]; then
    echo "▶ PCM $PCM_TASK seed=$SEED (emb_type=$EMB) -- déjà fait ($RESULTS_CSV), on saute"
    exit 0
fi

echo "▶ PCM $PCM_TASK seed=$SEED (emb_type=$EMB, concat_proj, pre)"

srun python baseline_run.py \
    --task "$PCM_TASK" \
    --strategy baseline \
    --seed "$SEED" \
    --tokenized_folder "$PCM_TOKENIZED_DIR" \
    --output_dir "$OUT_DIR" \
    --mini_data False \
    --emb_type "$EMB" --centroid_strategy mean \
    --ctx_fusion concat_proj --ctx_position pre \
    --use_crf True --use_sentence_lstm True --use_word_lstm True --use_attention_pooling True

echo "✅ Terminé (emb_type=$EMB, seed=$SEED) -> $OUT_DIR/$PCM_TASK/${EMB}_mean_pre_concat_proj/seed_$SEED/"
