#!/bin/bash
#SBATCH -A p30041
#SBATCH -p long
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 150:00:00
#SBATCH --job-name="sp_fold_pth_esm"
#SBATCH --output=../logs/sp_fold_pth_esm
#SBATCH --error=../logs/e_tmp
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u seq2esm.py
