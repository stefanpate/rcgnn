#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --job-name="sm"
#SBATCH --output=../logs/out/r4_pxr
#SBATCH --error=../logs/error/e_tmp_cpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module purge
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u similarity_matrix.py prot-rxn sprhea v3_folded_pt_ns rc_agg_r4/proteins rc_agg_r4/reactions