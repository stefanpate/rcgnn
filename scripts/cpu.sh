#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -t 1:00:00
#SBATCH --job-name="batch_fit"
#SBATCH --output=../logs/out/batch_fit
#SBATCH --error=../logs/error/e_tmp_cpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pasha@u.northwestern.edu
ulimit -c 0
module purge
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u batch_fit.py

