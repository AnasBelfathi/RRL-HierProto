#!/usr/bin/env bash
# Launch baseline + PCM (3 strategies) + PBR for a given task and seed list.
#
#   cd $WORK/rrl-prototype-methods
#   bash slurms/jeanzay/pipeline/launch.sh <task> <seed> [seed...]
#
# Examples:
#   bash slurms/jeanzay/pipeline/launch.sh legal-eval-v2 4 6
#   bash slurms/jeanzay/pipeline/launch.sh scotus-rhetorical_function 1 2 3
#
# <task> must be a task_type already registered in task.py (pubmed_task()).
# 00/10 (centroids/matching) run first regardless of task -- they no-op
# quickly if that task's centroids/matched files already exist (see the
# "déjà fait" guards in each .slurm), so it's always safe to include them.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <task> <seed> [seed...]" >&2
    exit 1
fi

TASK="$1"; shift
SEEDS_STR="$*"
read -ra SEEDS_ARR <<< "$SEEDS_STR"
N_SEEDS=${#SEEDS_ARR[@]}

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"     # sets PCM_ACCOUNT*, PCM_EMB_TYPES, cd's to PCM_HOME
mkdir -p job_out_err

N_EMB=${#PCM_EMB_TYPES[@]}
N_BASE=$N_SEEDS
N_PCM=$((N_SEEDS * N_EMB))
N_PBR=$N_SEEDS
N_MIND=$N_SEEDS

AX=(--account "$PCM_ACCOUNT_EXTRACT")
AT=(--account "$PCM_ACCOUNT")
EXPORT_VARS="ALL,PCM_TASK=$TASK,PCM_SEEDS=$SEEDS_STR"

echo "repo    : $PCM_HOME"
echo "task    : $TASK"
echo "seeds   : $SEEDS_STR ($N_SEEDS)"
echo "account : ${AX[1]} (extraction/matching) / ${AT[1]} (training)"
echo

S=slurms/jeanzay/pipeline
J_CENT=$(sbatch "${AX[@]}" --export="$EXPORT_VARS" --parsable "$S/00_build_centroids.slurm")
echo "00 centroids     : $J_CENT"
J_MATCH=$(sbatch "${AX[@]}" --export="$EXPORT_VARS" --parsable --dependency=afterok:$J_CENT "$S/10_match_centroids.slurm")
echo "10 matching      : $J_MATCH"
J_PCM=$(sbatch "${AT[@]}" --export="$EXPORT_VARS" --parsable --dependency=afterok:$J_MATCH --array=0-$((N_PCM - 1))%$N_PCM "$S/21_pcm.slurm")
echo "21 PCM training  : $J_PCM  ($N_PCM tasks = $N_SEEDS seeds x $N_EMB strategies)"

J_BASE=$(sbatch "${AT[@]}" --export="$EXPORT_VARS" --parsable --array=0-$((N_BASE - 1))%$N_BASE "$S/20_baseline.slurm")
echo "20 baseline      : $J_BASE  ($N_BASE tasks)"

J_PBR=$(sbatch "${AT[@]}" --export="$EXPORT_VARS" --parsable --array=0-$((N_PBR - 1))%$N_PBR "$S/22_pbr.slurm")
echo "22 PBR training  : $J_PBR  ($N_PBR tasks)"

# Mind-Your-Neighbours: trains its own baseline in-process, independent of
# everything else (no 00/10 dependency, unlike PCM).
J_MIND=$(sbatch "${AT[@]}" --export="$EXPORT_VARS" --parsable --array=0-$((N_MIND - 1))%$N_MIND "$S/25_mind_proto.slurm")
echo "25 Mind-proto    : $J_MIND  ($N_MIND tasks)"

echo
echo "monitor : squeue -u \$USER    |    logs in $PCM_HOME/job_out_err/"
echo "results : python slurms/jeanzay/pipeline/aggregate_results.py --task $TASK --seeds $SEEDS_STR"
