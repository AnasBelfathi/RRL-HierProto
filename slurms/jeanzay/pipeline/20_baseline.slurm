#!/bin/bash
#SBATCH --job-name=rrl-pipeline-20-baseline
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
# NB: --array above is only the default for a bare `sbatch 20_baseline.slurm`
# (3 seeds). launch.sh overrides it with the right size for $PCM_SEEDS.

# HSLN baseline, no context injection (ctx_position="") -- one array task per
# (task, seed), task/seeds coming from $PCM_TASK/$PCM_SEEDS (see env.sh).
# Independent of 00/10: can run in parallel with the PCM prototype pipeline.
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
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="output-training-${PCM_TASK}"
mkdir -p "$OUT_DIR"

RESULTS_CSV="$OUT_DIR/$PCM_TASK/full/seed_$SEED/results.csv"
if [[ -f "$RESULTS_CSV" ]]; then
    echo "▶ baseline $PCM_TASK seed=$SEED -- déjà fait ($RESULTS_CSV), on saute"
    exit 0
fi

echo "▶ baseline $PCM_TASK seed=$SEED"

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
    --unique_name full

echo "✅ Terminé (seed=$SEED) -> $OUT_DIR/$PCM_TASK/full/seed_$SEED/"
