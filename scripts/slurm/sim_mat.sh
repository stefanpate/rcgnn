#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 14:00:00
#SBATCH --job-name=sim_mat
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
#SBATCH --array=0-2

# Args
scripts_dir=/home/spn1560/hiec/scripts
start_chunk=(
    0
    3
    6
)

end_chunk=(
    2
    5
    8
)

# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
module load python-miniconda3/4.12.0
source activate /home/spn1560/.conda/envs/hiec2
python $scripts_dir/similarity_matrix.py rcmcs sprhea v3_folded_pt_ns 5000 ${start_chunk[$SLURM_ARRAY_TASK_ID]} ${end_chunk[$SLURM_ARRAY_TASK_ID]}