#!/bin/bash
#SBATCH --time=00:30:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --gres=gpu:2
#SBATCH --partition=alpha
#SBATCH --mem=64GB 
#SBATCH -A p_lv_clickbait
#SBATCH -J "ft-nmp-"
#SBATCH -o "ft-nmp-%j.out"
#SBATCH --error="ft-nmp-%j.err"
#SBATCH --mail-user=jakob_marius.mueller@tu-dresden.de
#SBATCH --mail-type ALL

module load Python/3.10.4

source venv/bin/activate

venv/bin/python3 finetuning_classification_transformer_non_multipart.py    

