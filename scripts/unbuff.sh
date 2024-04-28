#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 33
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH --job-name="sp_ops_gs_1"
#SBATCH --output=../logs/sp_ops_gs_1
#SBATCH --error=../logs/e_tmp_1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u mf_gs.py