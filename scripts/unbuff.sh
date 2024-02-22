#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH --job-name="simmat_esm_swissprot_price"
#SBATCH --output=../logs/outlog_cf_build_sim_mat_esm_swiss_price
#SBATCH --error=../logs/errlog_cf_build_sim_mat_esm_swiss_price
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u /home/spn1560/hiec/src/collaborative_filtering.py
