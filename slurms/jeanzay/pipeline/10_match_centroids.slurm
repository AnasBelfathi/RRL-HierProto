#!/bin/bash
#SBATCH --job-name=rrl-pipeline-10-matching
#SBATCH --output=./job_out_err/%x_%A.out
#SBATCH --error=./job_out_err/%x_%A.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=your_account@v100   # <-- set to your Jean Zay allocation
#SBATCH --mail-type=ALL

# PCM step 2/3: assign each sentence its nearest prototype, for each strategy
# in $PCM_EMB_TYPES. Requires 00_build_centroids.slurm to have produced
# centroids.jsonl for that strategy (missing ones are skipped with a warning,
# not fatal, so one missing document-grouping source doesn't block the rest).
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
load_extraction_modules

for EMB in "${PCM_EMB_TYPES[@]}"; do
    CDIR="$(pcm_centroids_dir "$EMB")"
    if [[ ! -f "$CDIR/centroids.jsonl" ]]; then
        echo "⚠️  $CDIR/centroids.jsonl introuvable -- lancez 00_build_centroids.slurm d'abord. On saute $EMB." >&2
        continue
    fi

    MOUT="$PCM_MATCH_OUT_ROOT/$EMB/mean/$PCM_TASK"
    if [[ -f "$MOUT/train.jsonl" && -f "$MOUT/dev.jsonl" && -f "$MOUT/test.jsonl" ]]; then
        echo "═══ matching: emb_type=$EMB -- déjà fait ($MOUT), on saute ═══"
        continue
    fi

    echo "═══ matching: emb_type=$EMB ═══"
    srun --cpu_bind=none \
        python matching-context/match_centroids.py \
            --dataset "$PCM_TASK" \
            --centroids_dir "$CDIR" \
            --emb_type "$EMB" \
            --strategy mean \
            --out_root "$PCM_MATCH_OUT_ROOT"

    echo "✅ matched sentences -> $PCM_MATCH_OUT_ROOT/$EMB/mean/$PCM_TASK/{train,dev,test}.jsonl"
done
