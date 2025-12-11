#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64GB
#SBATCH -t 5:00:00
#SBATCH --job-name="tr_clip"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
#SBATCH --array=0-1

# Args
script=/home/spn1560/hiec/scripts/train_clipzyme.py
data=(
    sprhea_rcmcs
    sprhea_rcmcs
    # sprhea_drfp
    # sprhea_drfp
    # sprhea_esm
    # sprhea_esm
    # sprhea_random_rxn_arc
    # sprhea_random_rxn_arc
    # sprhea_random_rc_arc
    # sprhea_random_rc_arc
)

ckpt=(
    epoch_29-step_127950.ckpt
    epoch_29-step_164550.ckpt
    # epoch_29-step_100230.ckpt
    # epoch_29-step_152550.ckpt
    # epoch_29-step_123960.ckpt
    # epoch_28-step_151409.ckpt
    # epoch_29-step_88500.ckpt
    # epoch_29-step_141330.ckpt
    # epoch_29-step_88500.ckpt
    # epoch_29-step_146220.ckpt
    
)

exp=clipzyme
test_only=true

# ckpt=epoch_23-step_131640.ckpt # Replace '=' with '_' for bash compatibility |  model.ckpt_fn=$ckpt in cmd below

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
eval "$(conda shell.bash hook)"
source activate /home/spn1560/.conda/envs/clipzyme
export HYDRA_FULL_ERROR=1
python $script data=${data[$SLURM_ARRAY_TASK_ID]} exp=$exp test_only=$test_only model.ckpt_fn=${ckpt[$SLURM_ARRAY_TASK_ID]} data.split_idx=$((($SLURM_ARRAY_TASK_ID % 2) * -1))