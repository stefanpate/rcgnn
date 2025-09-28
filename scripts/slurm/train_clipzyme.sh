#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH -t 14:00:00
#SBATCH --job-name="tr_clip"
#SBATCH --output=/home/spn1560/hiec/logs/out/%A
#SBATCH --error=/home/spn1560/hiec/logs/error/%A
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-3
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/train_clipzyme.py
exp=clipzyme
split_idx=(
    0
    1
    2
    -1
)

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
source activate /home/spn1560/.conda/envs/clipzyme
python $script exp=$exp data.split_idx=${split_idx[$SLURM_ARRAY_TASK_ID]}