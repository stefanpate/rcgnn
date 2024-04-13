#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 9
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --job-name="rf_top_150_classes_swissprot"
#SBATCH --output=../logs/rf_top_150_classes_swissprot
#SBATCH --error=../logs/e_tmp
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u rf_convenient_nested_cv.py