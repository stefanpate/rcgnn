#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --mem=10G
#SBATCH -t 1:00:00
#SBATCH --job-name="time_sp_gs_3_core_1"
#SBATCH --output=../logs/time_sp_gs_3_core_1
#SBATCH --error=../logs/e_tmp
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u mf_gs.py
