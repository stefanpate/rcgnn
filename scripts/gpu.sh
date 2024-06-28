#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -t 12:00:00
#SBATCH --job-name="vn_only_agg_pred_min_ops"
#SBATCH --output=../logs/out/vn_only_agg_pred_min_ops
#SBATCH --error=../logs/error/e_tmp_1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u gnn_min_multiclass_cv.py
