#!/bin/bash
#SBATCH --job-name=rrl-pipeline-00-centroids
#SBATCH --output=./job_out_err/%x_%A.out
#SBATCH --error=./job_out_err/%x_%A.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=your_account@v100   # <-- set to your Jean Zay allocation
#SBATCH --mail-type=ALL

# PCM step 1/3: prototypes (one per rhetorical-role label), for each of the 3
# document-grouping strategies in $PCM_EMB_TYPES (none/random-clusters/
# supervised-clustering). random-clusters and supervised-clustering need
# document-grouping/document-groupe/<emb_type>/<task>/train.json to
# already exist (see document-grouping/openai_kmeans_clustering.py --dataset <task> for
# supervised-clustering) -- missing ones are skipped with a warning by
# build_centroids.py itself, not fatal here.
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
    if [[ -f "$CDIR/centroids.jsonl" ]]; then
        echo "═══ centroids: emb_type=$EMB -- déjà fait ($CDIR/centroids.jsonl), on saute ═══"
        continue
    fi
    echo "═══ centroids: emb_type=$EMB ═══"
    srun python context-extraction/build_centroids.py \
        --datasets "$PCM_TASK" \
        --emb_type "$EMB" \
        --strategy mean \
        --use_labels
    echo "✅ centroids -> $CDIR/centroids.jsonl"
done
