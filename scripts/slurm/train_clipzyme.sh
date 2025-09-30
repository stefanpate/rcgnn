#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32GB
#SBATCH -t 3:00:00
#SBATCH --job-name="tr_clip"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/train_clipzyme.py
exp=dev_clipzyme
data=sprhea_rcmcs
training=dev
model=clipzyme
# ckpt=epoch_0-step_31.ckpt # Replace '=' with '_' for bash compatibility |  model.ckpt_path=$ckpt in cmd below

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/clipzyme
export HYDRA_FULL_ERROR=1
python $script exp=$exp data=$data training=$training model=$model