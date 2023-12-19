#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 36:00:00
#SBATCH --job-name="rem_esm"
#SBATCH -o outlog
#SBATCH -e errlog
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python get_esm_swissprot.py
