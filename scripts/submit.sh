#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 36:00:00
#SBATCH --job-name=sim_mats
#SBATCH --output=/home/spn1560/hiec/logs/out/%A
#SBATCH --error=/home/spn1560/hiec/logs/error/%A
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
scripts_dir=/home/spn1560/hiec/scripts

# Commands
ulimit -c 0
module purge
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python $scripts_dir/similarity_matrix.py gsi sprhea v3_folded_pt_ns 1000