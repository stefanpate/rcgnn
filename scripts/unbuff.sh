#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH --job-name="rf_fit_10_est"
#SBATCH --output=../logs/rf_fit_10_est
#SBATCH --error=../logs/e_tmp
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u rf_convenient_nested_cv.py