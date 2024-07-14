#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --job-name=test_1_hps_2_split_2
#SBATCH --output=../logs/out/test_1_hps_2_split_2
#SBATCH --error=../logs/error/test_1_hps_2_split_2
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u two_channel_fit.py -d sprhea -t sp_folded_pt_test -a homology -r 0.8 -e 1234 -n 3 -m 1 -s 2 -p 2 -g test_1
