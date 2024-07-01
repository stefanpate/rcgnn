from itertools import product
from src.utils import construct_sparse_adj_mat, split_data, save_hps_to_scratch, write_shell_script
import numpy as np
import subprocess


# Args
dataset_name = 'sprhea'
toc= 'sp_folded_pt' # 'sp_ops'
neg_multiple = 1
n_splits = 5
seed = 1234
gs_name = 'two_channel_mean_agg_binaryffn_pred_neg_1'
allocation = 'p30041'
partition = 'gengpu'
mem = '16G'
time = '16' # Hours
sample_embed_type = 'esm'
fit_script = 'two_channel_fit.py'
save_models = True

# Hyperparameters

# # Matrix factorization
# hps = {
#     'lr':[5e-3],
#     'max_epochs':[7500],
#     'batch_size':[5],
#     'optimizer__weight_decay':[5e-5],
#     'module__scl_embeds':[True],
#     'neg_multiple': [1],
#     # 'module__n_factors':[20, 50, 100]
#     'user_embeds':["esm_rank_20", "esm_rank_50", "esm_rank_100"]
# }

# RC GNN
hps = {
    'n_epochs':[75],
    'pred_head':['binary'],
    'agg':['mean'],
    'd_prot':[1280],
    'd_h_mpnn':[300],
    'neg_multiple':[neg_multiple]
}

# TODO: backwards compatibility rcgnn w/ mf on handling duplicates under same gs_name
# # Check gs_name not used before
# old_gs_path = "../artifacts/model_evals/old_gs_names.txt"
# with open(old_gs_path, 'r') as f:
#     old_gs_names = [elt.rstrip() for elt in f.readlines()]

# if gs_name in old_gs_names:
#     raise ValueError(f"{gs_name} has already been used as a grid search name")

# old_gs_names.append(gs_name) # Add current gs_name

# with open(old_gs_path, 'w') as f:
#     f.write('\n'.join(elt for elt in old_gs_names))

# Load data
print("Loading data")
adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset_name, toc)
X = np.array(list(zip(*adj.nonzero())))
y = np.ones(shape=(X.shape[0], 1))

# Cartesian product of hyperparams
hp_keys = list(hps.keys())
hp_cart_prod = [{hp_keys[i] : elt[i] for i in range(len(elt))} for elt in product(*hps.values())]

# Save hp dicts to scratch
print("Saving hyperparams to scratch")
for i, hp in enumerate(hp_cart_prod):
    save_hps_to_scratch(hp, gs_name, i)
    
# Split data
print("Splitting dataset")
split_guide = split_data(X,
                        y,
                        dataset_name,
                        toc,
                        n_splits,
                        seed,
                        neg_multiple
                        )

# Submit slurm jobs to fit
'''
arg_str
-------
MF: f"-d {ds_name} -e {seed} -n {n_splits} -s {split_idx} -p {hp_idx} -g {gs_name}{should_save(save_models)"
RC GNN / 2c: f"-d {dataset_name} -t {toc} -e {seed} -n {n_splits} -s {split_idx} -p {hp_idx} -g {gs_name} -m {sample_embed_type}" 
'''

# print("Submitting jobs")
# for hp_idx, hp in enumerate(hp_cart_prod):
#     for split_idx in range(n_splits):
#         arg_str = f"-d {dataset_name} -t {toc} -e {seed} -n {n_splits} -s {split_idx} -p {hp_idx} -g {gs_name} -m {sample_embed_type}"
#         shell_script = write_shell_script(
#             allocation,
#             partition,
#             mem,
#             time,
#             fit_script,
#             arg_str,
#             job_name=f"{gs_name}_hps_{hp_idx}_split_{split_idx}"
#         )
        
#         with open("batch.sh", 'w') as f:
#             f.write(shell_script)

#         subprocess.run(["sbatch", "batch.sh"])