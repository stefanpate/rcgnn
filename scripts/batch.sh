#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --job-name=test_hp_saving_hps_0_split_1
#SBATCH --output=../logs/out/test_hp_saving_hps_0_split_1
#SBATCH --error=../logs/error/test_hp_saving_hps_0_split_1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u two_channel_fit.py -d sprhea -t sp_folded_pt_20p -e 1234 -n 2 -s 1 -p 0 -g test_hp_saving -m esm
