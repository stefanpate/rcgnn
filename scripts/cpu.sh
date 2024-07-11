#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=17G
#SBATCH -t 2:00:00
#SBATCH --job-name="cd_hit_test"
#SBATCH --output=../logs/out/cd_hit_test
#SBATCH --error=../logs/error/e_tmp_cpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
cd-hit -i ../data/new/new.fasta -o ../data/new/new_80.fasta -c 0.8 -n 5 -M 16000 –d 0 -T 8
