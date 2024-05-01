#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH --job-name=mf_sp_ops_0_split_0_hp_0
#SBATCH --output=../logs/out/mf_sp_ops_0_split_0_hp_0
#SBATCH --error=../logs/error/mf_sp_ops_0_split_0_hp_0
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u mf_fit.py -d sp_ops -e 1234 -n 3 -s 0 -p 0 -g mf_sp_ops_0
