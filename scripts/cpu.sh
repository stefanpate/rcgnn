#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH --job-name="batch_resume"
#SBATCH --output=../logs/out/batch_resume
#SBATCH --error=../logs/error/e_tmp_cpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module purge
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u batch_resume.py

