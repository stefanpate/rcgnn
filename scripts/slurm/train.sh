#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48GB
#SBATCH -t 96:00:00
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
    sprhea_random_rc_arc
    sprhea_random_rc_arc
    sprhea_random_rc_arc
    sprhea_random_rxn_arc
    sprhea_random_rxn_arc
    sprhea_random_rxn_arc
)
model=(
    rc_agg
    rc_cxn
    bom
    rc_agg
    rc_cxn
    bom
)
n_epochs=(
    6
    12
    24
    21
    24
    24
)
split_idx=-2

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate /home/spn1560/.conda/envs/hiec2
python $script data=${data[$SLURM_ARRAY_TASK_ID]} model=${model[$SLURM_ARRAY_TASK_ID]} data.split_idx=$split_idx model.n_epochs=${n_epochs[$SLURM_ARRAY_TASK_ID]}