#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=50GB
#SBATCH -t 2:00:00
#SBATCH --job-name="st_dat"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/stage_data.py
data=(
    sprhea_drfp
    # sprhea_esm
    # sprhea_n_100
    # sprhea_rcmcs
    # sprhea_homology
)

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/hiec
python $script data=${data[$SLURM_ARRAY_TASK_ID]}