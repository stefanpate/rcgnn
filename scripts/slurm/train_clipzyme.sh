#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH -t 24:00:00
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
    sprhea_random_rxn_arc
    sprhea_random_rc_arc
)
split_idx=0
exp=clipzyme
test_only=false

ckpt=(
    epoch_26-step_79650.ckpt
    epoch_27-step_81704.ckpt
)

# ckpt=epoch_23-step_131640.ckpt # Replace '=' with '_' for bash compatibility |  model.ckpt_fn=$ckpt in cmd below

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/clipzyme
export HYDRA_FULL_ERROR=1
python $script data=${data[$SLURM_ARRAY_TASK_ID]} data.split_idx=$split_idx exp=$exp test_only=$test_only model.ckpt_fn=${ckpt[$SLURM_ARRAY_TASK_ID]}