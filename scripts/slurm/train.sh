#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=24GB
#SBATCH -t 96:00:00
#SBATCH --job-name="train"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-2
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/train.py
data=sprhea_rcmcs
training=base
model=mfp
exp=low_epoch
n_epochs=1
split_idx=(
    -1
    1
    2
    # 0
)

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/hiec2
python $script data=$data training=$training model=$model data.split_idx=${split_idx[$SLURM_ARRAY_TASK_ID]} exp=$exp training.n_epochs=$n_epochs