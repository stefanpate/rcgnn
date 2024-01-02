#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --job-name="esm_clean"
#SBATCH --output=outlog
#SBATCH --error=errlog
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python esm2clean.py
