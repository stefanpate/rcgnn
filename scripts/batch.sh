#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -t 12:00:00
#SBATCH --job-name=vn_agg_depths_homology_80_0_hps_17_split_4
#SBATCH --output=../logs/out/vn_agg_depths_homology_80_0_hps_17_split_4
#SBATCH --error=../logs/error/vn_agg_depths_homology_80_0_hps_17_split_4
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u two_channel_fit.py -d sprhea -t sp_folded_pt -a homology -r 0.8 -e 1234 -n 5 -m 1 -b esm -s 4 -p 17 -g vn_agg_depths_homology_80_0
