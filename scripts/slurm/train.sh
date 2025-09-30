#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16GB
#SBATCH -t 1:00:00
#SBATCH --job-name="train"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-3
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/train.py
exp=dev_drfp
data=sprhea_n_100
training=dev
model=drfp


# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/hiec2
python $script exp=$exp data=$data training=$training model=$model