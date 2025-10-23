#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48GB
#SBATCH -t 12:00:00
#SBATCH --job-name="train"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-5
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/train.py
data=(
    sprhea_rcmcs
    sprhea_rcmcs
    sprhea_random_rc_arc
    sprhea_random_rc_arc
    sprhea_random_rxn_arc
    sprhea_random_rxn_arc
)
model=rxnfp

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/hiec2
python $script data=${data[$SLURM_ARRAY_TASK_ID]} model=$model data.split_idx=$((($SLURM_ARRAY_TASK_ID % 2) * -1))