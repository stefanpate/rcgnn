#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH --job-name="swissprot"
#SBATCH -o outlog
#SBATCH -e errlog
ulimit -c 0
module load python/anaconda3.6
source activate mine
python get_esm_swissprot.py
