#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --job-name="rf_top_500_classes_swissprot_run_1"
#SBATCH --output=../logs/rf_top_500_classes_swissprot_run_1
#SBATCH --error=../logs/e_tmp
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u rf_convenient_cv.py
