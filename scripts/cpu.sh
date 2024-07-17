#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 2:00:00
#SBATCH --job-name="batch_fit"
#SBATCH --output=../logs/out/batch_fit
#SBATCH --error=../logs/error/e_tmp_cpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python-miniconda3/4.12.0
module load gcc/9.2.0
source activate hiec
python -u batch_fit.py

