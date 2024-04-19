#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=10-0:00:00
#SBATCH --job-name=tsdae_classifier
#SBATCH --output=tsdae_classifier_%j.out
#SBATCH --error=tsdae_classifier_%j.err

module load python

python TSDAE.py
