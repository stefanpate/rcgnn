#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=40GB
#SBATCH -t 20:00:00
#SBATCH --job-name=extract
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
scripts_dir=/home/spn1560/hiec/scripts
script=seq2esm.py

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
eval "$(conda shell.bash hook)"
conda activate /home/spn1560/.conda/envs/hiec2
python $scripts_dir/$script