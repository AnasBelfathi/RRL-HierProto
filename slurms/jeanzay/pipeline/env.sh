#!/usr/bin/env bash
# Sourced by every job in this pipeline. PCM_TASK selects which registered
# task.py task to run on -- not tied to any one dataset.
# Modeled after masking-discursive-injection/discourse-legal-masking/scripts/jeanzay/env.sh.
#
# Strict mode only when run non-interactively (a .slurm script), never when you
# `source` it by hand in an interactive shell -- otherwise a failing command
# would kill your shell. No -u (nounset): `module load`/`conda activate` source
# system hook scripts (e.g. libblas_mkl_activate.sh referencing
# $MKL_INTERFACE_LAYER) that are not nounset-safe and would otherwise abort here.
case $- in *i*) ;; *) set -eo pipefail ;; esac

# --- allocation --------------------------------------------------------
# Set these to YOUR Jean Zay (or other Slurm/GENCI) allocation before running
# anything (see `idracct`). Training and extraction/matching can use the
# same or different accounts -- override either with
# `PCM_ACCOUNT=... PCM_ACCOUNT_EXTRACT=... sbatch ...` if you only have hours
# on one of them.
export PCM_ACCOUNT=${PCM_ACCOUNT:-"your_account@v100"}
export PCM_ACCOUNT_EXTRACT=${PCM_ACCOUNT_EXTRACT:-"your_account@v100"}

# --- paths ---------------------------------------------------------------
export PCM_HOME=${PCM_HOME:-"$WORK/rrl-prototype-methods"}
cd "$PCM_HOME"

# Overridable so the same scripts work for any task/seed set, e.g.:
#   sbatch --export=ALL,PCM_TASK=scotus-rhetorical_function,PCM_SEEDS="1 2 3" 20_baseline.slurm
# (see launch.sh, which computes this for you).
export PCM_TASK=${PCM_TASK:-legal-eval-v2}
export PCM_SEEDS=${PCM_SEEDS:-"1 2 3"}
export PCM_TOKENIZED_DIR=processed-datasets
export PCM_MATCH_OUT_ROOT="matching-context/new_similarity_outputs_with_labels"
# All 3 document-grouping strategies compared for PCM (see build_centroids.py):
#   none                = corpus-level prototypes, no document clustering
#   random-clusters     = documents randomly grouped first (control condition)
#   supervised-clustering = documents grouped by OpenAI-embedding KMeans first
export PCM_EMB_TYPES=(none random-clusters supervised-clustering)

# build_centroids.py names emb_type=none's output dir "no_cluster" (see its
# main()); mirror that here so every script agrees on the same paths.
pcm_centroids_dir() {
    local emb="$1"
    if [[ "$emb" == "none" ]]; then
        echo "context-extraction/proto-with-labels-v2/no_cluster/mean/${PCM_TASK}"
    else
        echo "context-extraction/proto-with-labels-v2/${emb}/mean/${PCM_TASK}"
    fi
}

# Compute nodes have NO internet access. models/bert-base-uncased and
# models/legal-bert-base-uncased must be pre-downloaded on a frontend node
# first (see download_models.py). Offline mode makes a missing/misnamed model
# fail immediately with a clear error instead of a multi-minute network retry
# loop against huggingface.co.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# ~/.local/lib/pythonX.Y/site-packages (pip install --user) leaks into every
# Python run regardless of the active conda env, unless disabled. A numpy
# installed there conflicts with the one inside my_new_env and crashes
# thinc's compiled extension (allennlp -> spacy -> thinc) with "numpy.dtype
# size changed, may indicate binary incompatibility". Confirmed in practice:
# the smoke test dodged this only because PYTHONUSERBASE was still pointing
# at load_extraction_modules' path from an earlier step in the same shell --
# a coincidence, not a fix. Disable user-site packages outright instead.
export PYTHONNOUSERSITE=1

# --- software ------------------------------------------------------------
# Two different environments are needed: extraction/matching (sentence
# embeddings only) vs. training (allennlp HSLN model). Call the matching
# function at the top of each job/step instead of duplicating module loads.
load_extraction_modules() {
    module purge
    module load cpuarch/amd
    module load pytorch-gpu/py3/2.6.0
    export PYTHONUSERBASE=$WORK/rrl_pkgs_sentence_transformers
}

load_training_modules() {
    module purge
    module load anaconda-py3/2024.06
    conda activate my_new_env
}
