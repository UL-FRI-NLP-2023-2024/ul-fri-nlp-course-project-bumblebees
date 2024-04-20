#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=tsdae_classifier
#SBATCH --output=/logs/tsdae_classifier_%j.out
#SBATCH --error=/logs/tsdae_classifier_%j.err

srun singularity exec --nv ../containers/tsdae python code/TSDAE.py
