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
#SBATCH --array=0-2
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/train.py
data=(
    sprhea_rcmcs
    sprhea_rcmcs
    sprhea_rcmcs
    sprhea_rcmcs
)
model=(
    rc_agg
    rc_cxn
    bom
    rxnfp
)
split_idx=0

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
eval "$(conda shell.bash hook)"
source activate /home/spn1560/.conda/envs/hiec2
python $script data=${data[$SLURM_ARRAY_TASK_ID]} model=${model[$SLURM_ARRAY_TASK_ID]} data.split_idx=$split_idx