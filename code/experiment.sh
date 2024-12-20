#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=7:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=/home/s28zabed/CodIngAssignmentEvaluation/code/output.out 

# Activate the environment
source ../.venv/bin/activate

# Run your script
python inference.py