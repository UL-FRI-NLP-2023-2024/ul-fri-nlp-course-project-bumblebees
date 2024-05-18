#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=gpl_classifier
#SBATCH --output=./logs/gpl_classifier_%j.out
#SBATCH --error=./logs/gpl_classifier_%j.err

srun singularity exec --nv ../containers/container-pytorch-onj.sif python code/GPL.py
