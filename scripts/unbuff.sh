#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --job-name="new_nth"
#SBATCH --output=../logs/outlog_nth_level_new
#SBATCH --error=../logs/errlog_nth_level_new
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u nth_level_error.py
