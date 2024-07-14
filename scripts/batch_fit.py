# from itertools import product
# from src.utils import construct_sparse_adj_mat, split_data, save_hps_to_scratch, write_shell_script, ensure_dirs, negative_sample_bipartite
# import numpy as np
# import pandas as pd
# import subprocess
from src.cross_validation import BatchGridSearch, BatchScriptParams

# Args
dataset_name = 'sprhea'
toc = 'sp_folded_pt_test' # Name of file with protein id | features/labels | sequence
n_splits = 2
seed = 1234
gs_name = 'test_2' # Grid search name
allocation = 'p30041'
partition = 'gengpu'
mem = '16G' # 16G
time = '3' # Hours 13
fit_script = 'two_channel_fit.py'
neg_multiple = 1
split_strategy = 'homology'
split_sim_threshold = 0.8
batch_script_params = BatchScriptParams(allocation=allocation, partition=partition, mem=mem, time=time, script=fit_script)

# Hyperparameters

# RC GNN
hps = {
    'n_epochs':[2],
    'pred_head':['dot_sig'], # 'binary' | 'dot_sig'
    'message_passing':['bondwise_dict'], # 'bondwise' | 'bondwise_dict'
    'agg':['attention'], # 'mean' | 'last' | 'attention'
    'd_prot':[1280],
    'd_h_mpnn':[10],
    'model':['mpnn_dim_red'] # 'mpnn' | 'mpnn_dim_red'
}

gs = BatchGridSearch(
    dataset_name=dataset_name,
    toc=toc,
    neg_multiple=neg_multiple,
    gs_name=gs_name,
    n_splits=n_splits,
    split_strategy=split_strategy,
    seed=seed,
    split_sim_threshold=split_sim_threshold,
    batch_script_params=batch_script_params,
    hps=hps
)

gs.run()


print('hold')

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

# # Load data
# print("Loading data")
# adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset_name, toc)
# X = np.array(list(zip(*adj.nonzero())))
# y = np.ones(shape=(X.shape[0], 1))

# # Cartesian product of hyperparams
# hp_keys = list(hps.keys())
# hp_cart_prod = [{hp_keys[i] : elt[i] for i in range(len(elt))} for elt in product(*hps.values())]

# # Save hp dicts to scratch
# print("Saving hyperparams to scratch")
# for i, hp in enumerate(hp_cart_prod):
#     save_hps_to_scratch(hp, gs_name, i)
    
# # Write a hp idx csv
# eval_dir = f"/projects/p30041/spn1560/hiec/artifacts/model_evals/gnn/{dataset_name}_{toc}_{gs_name}"
# ensure_dirs(eval_dir)
# cols = hp_keys
# data = [list(elt.values()) for elt in hp_cart_prod]
# hp_df = pd.DataFrame(data=data, columns=cols)
# hp_df.to_csv(f"{eval_dir}/hp_toc.csv", sep='\t')

# # Submit slurm jobs to fit
# '''
# arg_str
# -------
# Matrix Factorization: f"-d {ds_name} -e {seed} -n {n_splits} -s {split_idx} -p {hp_idx} -g {gs_name}{should_save(save_models)"
# RC GNN / 2channel: f"-d {dataset_name} -t {toc} -e {seed} -n {n_splits} -s {split_idx} -p {hp_idx} -g {gs_name} -m {sample_embed_type}" 
# '''

# print("Submitting jobs")
# for hp_idx, hp in enumerate(hp_cart_prod):


#     neg_multiple = hp['neg_multiple']
#     n_rows = X[:,0].max()
#     n_cols = X[:, 1].max()
#     negative_data = negative_sample_bipartite(int(X.shape[0] * neg_multiple), n_rows, n_cols, obs_pairs=X, seed=seed)

#     # Split data
#     split_guide = split_data(X,
#                             y,
#                             dataset_name,
#                             toc,
#                             n_splits,
#                             seed,
#                             neg_multiple
#                             )
    
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
