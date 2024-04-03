#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=192G
#SBATCH -t 10:00:00
#SBATCH --job-name="fit_scl"
#SBATCH --output=../logs/nest_max_smpl_scl
#SBATCH --error=../logs/tmp_e1
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python model_fit_scaling.py
