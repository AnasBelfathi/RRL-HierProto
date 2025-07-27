#!/bin/bash
#SBATCH --job-name=bert_hsln
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=zsl@a100
#SBATCH --mail-user=anas.belfathi@etu.univ-nantes.fr
#SBATCH --mail-type=ALL
#SBATCH --array=0-2%3

set -e

# Chargement des modules
module purge
module load cpuarch/amd
module load anaconda-py3/2024.06
conda activate my_new_env

# Définir la liste des noms de tâches
TASK_NAMES=("category" "rhetorical_function" "steps")

# Sélectionner la tâche en fonction de l'indice SLURM_ARRAY_TASK_ID
TASK_NAME=${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "============================="
echo "Exécution pour la tâche : $TASK_NAME"
echo "============================="

# Lancer le script Python pour la tâche spécifiée
srun python3 baseline_run.py --task $TASK_NAME

echo "Tâche $TASK_NAME terminée."
