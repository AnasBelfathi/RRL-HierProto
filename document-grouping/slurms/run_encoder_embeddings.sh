#!/bin/bash
#SBATCH --job-name=run_extraction
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH -C v100-32g  # Type de GPU (ici V100 avec 32GB)
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread



set -e

# 🛠 Load required modules
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/python_pkgs_sentence_transformers

srun python ./document-grouping/encoder_doc_embeddings.py