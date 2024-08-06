#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -t 36:00:00
#SBATCH --job-name=cv_hp_idx_235_split_4
#SBATCH --output=../logs/out/cv_hp_idx_235_split_4
#SBATCH --error=../logs/error/cv_hp_idx_235_split_4
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u two_channel_fit.py -s 4 -p 235