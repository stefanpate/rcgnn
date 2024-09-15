#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 5:00:00
#SBATCH --job-name="sm"
#SBATCH --output=../logs/out/rcmcs_sim_mat
#SBATCH --error=../logs/error/e_tmp_cpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module purge
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u similarity_matrix.py rcmcs sprhea v3_folded_pt_ns