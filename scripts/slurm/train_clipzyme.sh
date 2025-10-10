#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64GB
#SBATCH -t 48:00:00
#SBATCH --job-name="tr_clip"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
#SBATCH --array=0

# Args
script=/home/spn1560/hiec/scripts/train_clipzyme.py
data=(
    # sprhea_rcmcs
    # sprhea_blosum
    # sprhea_drfp
    # sprhea_random_rxn_arc
    sprhea_random_rc_arc
    # sprhea_esm
    # sprhea_gsi
)
split_idx=-1

# ckpt=epoch_0-step_31.ckpt # Replace '=' with '_' for bash compatibility |  model.ckpt_path=$ckpt in cmd below

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/clipzyme
export HYDRA_FULL_ERROR=1
python $script data=${data[$SLURM_ARRAY_TASK_ID]} data.split_idx=$split_idx