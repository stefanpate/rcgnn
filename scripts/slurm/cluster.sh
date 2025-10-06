#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64GB
#SBATCH -t 3:00:00
#SBATCH --job-name=clstr
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
scripts_dir=/home/spn1560/hiec/scripts
similarity_score=esm
dataset=sprhea
toc=v3_folded_pt_ns

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/hiec
python $scripts_dir/cluster.py similarity_score=$similarity_score dataset=$dataset toc=$toc