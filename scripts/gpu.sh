#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH --job-name="gpu_test"
#SBATCH --output=../logs/gpu_test
#SBATCH --error=../logs/e_tmp
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u mf_rosenthal.py
