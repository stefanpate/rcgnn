#!/bin/bash
#SBATCH -A p30041
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -t 1:00:00
#SBATCH --job-name="test_2c_eval"
#SBATCH --output=../logs/out/test_2c_eval
#SBATCH --error=../logs/error/e_tmp_gpu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pasha@u.northwestern.edu
ulimit -c 0
module load python/anaconda3.6
module load gcc/9.2.0
source activate hiec
python -u evaluate_two_channel.py
