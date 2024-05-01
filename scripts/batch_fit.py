from itertools import product
from src.utils import construct_sparse_adj_mat, split_data, save_hps_to_scratch
import numpy as np
import subprocess

def write_shell_script(
        allocation,
        partition,
        mem,
        time,
        ds_name,
        gs_name,
        seed,
        n_splits,
        split_idx,
        hp_idx
        ):
    
    shell_script = f"""#!/bin/bash
    #SBATCH -A {allocation}
    #SBATCH -p {partition}
    #SBATCH -N 1
    #SBATCH -n 1
    #SBATCH --mem={mem}
    #SBATCH -t {time}:00:00
    #SBATCH --job-name={gs_name}_split_{split_idx}_hp_{hp_idx}
    #SBATCH --output=../logs/out/{gs_name}_split_{split_idx}_hp_{hp_idx}
    #SBATCH --error=../logs/error/{gs_name}_split_{split_idx}_hp_{hp_idx}
    #SBATCH --mail-type=END
    #SBATCH --mail-type=FAIL
    #SBATCH --mail-user=stefan.pate@northwestern.edu
    ulimit -c 0
    module load python/anaconda3.6
    module load gcc/9.2.0
    source activate hiec
    python -u mf_fit.py -d {ds_name} -e {seed} -n {n_splits} -s {split_idx} -p {hp_idx} -g {gs_name}
    """
    shell_script = shell_script.replace("    ", "") # Remove tabs
    return shell_script

# Args
dataset_name = 'sp_ops'
embed_type = None # esm | clean
n_splits = 3
seed = 1234
gs_name = 'mf_sp_ops_0'
allocation = 'b1039'
partition = 'b1039'
mem = '8G'
time = '4' # Hours

# Load data
print("Loading data")
adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset_name)
X = np.array(list(zip(*adj.nonzero())))
y = np.ones(shape=(X.shape[0], 1))

# Hyperparameters
hps = {
    'lr':[5e-3,],
    'max_epochs':[1,], # 7500
    'batch_size':[10, 25],
    'optimizer__weight_decay':[1e-4, 5e-5],
    'module__n_factors':[20, 50],
    'module__scl_embeds':[True, False]
}

# Cartesian product of hyperparams
hp_keys = list(hps.keys())
hp_cart_prod = [{hp_keys[i] : elt[i] for i in range(len(elt))} for elt in product(*hps.values())]

# Save hp dicts to scratch
print("Saving hyperparams to scratch")
for i, hp in enumerate(hp_cart_prod):
    save_hps_to_scratch(hp, gs_name, i)
    
# Split data
print("Splitting dataset")
split_data(dataset_name, embed_type, n_splits, seed, X, y)



# Submit slurm jobs to fit
print("Submitting jobs")
for hp_idx, hp in enumerate(hp_cart_prod):
    for split_idx in range(n_splits):
        shell_script = write_shell_script(
            allocation,
            partition,
            mem,
            time,
            dataset_name,
            gs_name,
            seed,
            n_splits,
            split_idx,
            hp_idx
        )
        
        with open("batch.sh", 'w') as f:
            f.write(shell_script)

        subprocess.run(["sbatch", "batch.sh"])

        break
    break