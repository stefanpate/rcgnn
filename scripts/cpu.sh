#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --job-name="incresm"
#SBATCH --output=../logs/out/esm_pull_v3
#SBATCH --error=../logs/error/e_tmp_cpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module purge
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u seq2esm.py

