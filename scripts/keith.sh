#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=150G
#SBATCH -t 2:00:00
#SBATCH --job-name="rf_fit_test"
#SBATCH --output=../logs/rf_fit_test
#SBATCH --error=../logs/e_tmp
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u rf_convenient_nested_cv.py
