#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH --job-name="kfold_clean"
#SBATCH --output=../logs/outlog_cf_kfold_clean
#SBATCH --error=../logs/errlog_cf_kfold_clean
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u /home/spn1560/hiec/src/collaborative_filtering.py
