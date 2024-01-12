#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH --job-name="price_mask"
#SBATCH --output=../logs/outlog_masked_price
#SBATCH --error=../logs/errlog_masked_price
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u masked_label_prediction.py
