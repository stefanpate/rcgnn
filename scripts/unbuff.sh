#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH --job-name="kfold_esm"
#SBATCH --output=../logs/outlog_cf_kfold_esm_knn_speedup
#SBATCH --error=../logs/errlog_cf_kfold_esm_knn_speedup
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u /home/spn1560/hiec/src/collaborative_filtering.py
