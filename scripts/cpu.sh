#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=10G
#SBATCH -t 8:00:00
#SBATCH --job-name="sp_ops_continue_fit_8"
#SBATCH --output=../logs/sp_ops_continue_fit_8
#SBATCH --error=../logs/e_tmp_3
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u mf_fit.py
