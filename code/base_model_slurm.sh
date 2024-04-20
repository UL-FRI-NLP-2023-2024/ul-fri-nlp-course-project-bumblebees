#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=base_model_classifier
#SBATCH --output=./logs/base_model_classifier_%j.out
#SBATCH --error=./logs/base_model_classifier_%j.err

srun singularity exec --nv ../containers/container-pytorch-onj.sif python code/base_model.py
