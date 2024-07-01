#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 16:00:00
#SBATCH --job-name=rc_gnn_two_channel_mean_agg_binaryffn_pred_neg_1_hps_0_split_4
#SBATCH --output=../logs/out/rc_gnn_two_channel_mean_agg_binaryffn_pred_neg_1_hps_0_split_4
#SBATCH --error=../logs/error/rc_gnn_two_channel_mean_agg_binaryffn_pred_neg_1_hps_0_split_4
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u two_channel_fit.py -d sprhea -t sp_folded_pt -e 1234 -n 5 -s 4 -p 0 -g rc_gnn_two_channel_mean_agg_binaryffn_pred_neg_1 -m esm
