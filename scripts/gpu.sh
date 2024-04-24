#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH -t 24:00:00
#SBATCH --job-name="mf_toy_gs_1"
#SBATCH --output=../logs/mf_toy_gs_1
#SBATCH --error=../logs/e_tmp
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u mf_gs.py
