#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 16:00:00
#SBATCH --job-name="comp_clf"
#SBATCH --output=../logs/outlog_comp_clf
#SBATCH --error=../logs/errlog_comp_clf
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u compare_classifiers.py
