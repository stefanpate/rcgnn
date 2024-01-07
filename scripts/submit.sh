#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --job-name="new2clean"
#SBATCH --output=outlog
#SBATCH --error=errlog
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python esm2clean.py
