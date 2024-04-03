#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=192G
#SBATCH -t 4:00:00
#SBATCH --job-name="fit_scl"
#SBATCH --output=../logs/nest_max_smpl_scl
#SBATCH --error=../logs/tmp_e1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u model_fit_scaling.py