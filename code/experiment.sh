#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=5:10:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

# Activate the environment
source ../.venv/bin/activate

# Run your script
python inference.py