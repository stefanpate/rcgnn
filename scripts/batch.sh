#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 18:00:00
#SBATCH --job-name=two_channel_gs_0_hps_23_split_2
#SBATCH --output=../logs/out/two_channel_gs_0_hps_23_split_2
#SBATCH --error=../logs/error/two_channel_gs_0_hps_23_split_2
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u two_channel_fit.py -d sprhea -t sp_folded_pt -e 1234 -n 3 -s 2 -p 23 -g two_channel_gs_0 -m esm
