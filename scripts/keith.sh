#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 25
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --job-name="sp_ops_gs_3"
#SBATCH --output=../logs/sp_ops_gs_3
#SBATCH --error=../logs/e_tmp
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u mf_gs.py
