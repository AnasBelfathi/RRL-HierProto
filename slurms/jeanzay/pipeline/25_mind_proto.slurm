#!/bin/bash
#SBATCH --job-name=rrl-pipeline-25-mind-proto
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
# NB: --array above is only the default for a bare `sbatch` (3 seeds).
# launch.sh overrides it with the right size for $PCM_SEEDS.

# Single-prototype interpolation -- see mind_neighbours_run.py docstring.
# Unlike PCM/PBR/baseline, this script trains its OWN baseline in-process
# (no checkpoint reuse), so it's independent of 00/10/20/21/22. One array
# task per (task, seed).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    HERE="$SLURM_SUBMIT_DIR/slurms/jeanzay/pipeline"
else
    HERE="$(cd "$(dirname "$0")" && pwd)"
fi
source "$HERE/env.sh" || { echo "FATAL: could not source $HERE/env.sh" >&2; exit 1; }
load_training_modules

SEEDS=($PCM_SEEDS)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="mind-output-training-${PCM_TASK}"
mkdir -p "$OUT_DIR"

RESULTS_CSV="$OUT_DIR/$PCM_TASK/mind_proto/seed_$SEED/results.csv"
if [[ -f "$RESULTS_CSV" ]]; then
    echo "▶ Mind-proto $PCM_TASK seed=$SEED -- déjà fait ($RESULTS_CSV), on saute"
    exit 0
fi

echo "▶ Mind-proto $PCM_TASK seed=$SEED (single prototype, tau grid, lambda step 0.1)"

srun python mind_neighbours_run.py \
    --task "$PCM_TASK" \
    --seed "$SEED" \
    --tokenized_folder "$PCM_TOKENIZED_DIR" \
    --output_dir "$OUT_DIR" \
    --mini_data False \
    --lambda_step 0.1 \
    --tau_grid 0.5 1.0 2.0 5.0

echo "✅ Terminé (seed=$SEED) -> $RESULTS_CSV"
