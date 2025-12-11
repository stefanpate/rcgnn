#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48GB
#SBATCH -t 24:00:00
#SBATCH --job-name="hpo"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/hiec/scripts/bhpo.py
model=(
    rc_agg
    rc_cxn
    bom
    cgr
    mfp
    drfp
    rxnfp
)
n_trials=1000
timeout=86400 # seconds

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/hiec2
python $script model=${data[$SLURM_ARRAY_TASK_ID]} n_trials=$n_trials timeout=$timeout