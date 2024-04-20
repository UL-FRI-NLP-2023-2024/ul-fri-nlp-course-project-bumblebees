#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=5-0:00:00
#SBATCH --job-name=tsdae_classifier
#SBATCH --output=/logs/tsdae_classifier_%j.out
#SBATCH --error=/logs/tsdae_classifier_%j.err

srun --gres=gpu:1 --partition=gpu singularity exec --nv ../containers/tsdae python code/TSDAE.py
