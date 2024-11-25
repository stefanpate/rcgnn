#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH --job-name=cv_hp_idx_313_split_4
#SBATCH --output=/home/spn1560/hiec/logs/out/cv_hp_idx_313_split_4
#SBATCH --error=/home/spn1560/hiec/logs/error/cv_hp_idx_313_split_4
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u /home/spn1560/hiec/scripts/two_channel_fit_split.py -s 4 -p 313