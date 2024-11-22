from src.cross_validation import BatchGridSearch, BatchScript, HyperHyperParams
from src.config import filepaths

dataset_name = 'sprhea'
toc = 'v3_folded_pt_ns' # Name of file with protein id | features/labels | sequence
n_splits = 5
seed = 1234
allocation = 'p30041'
partition = 'gengpu'
mem = '32G' # 12G
time = '12' # Hours 36
fit_script = 'two_channel_fit_split.py'
neg_multiple = 3
split_strategy = 'rcmcs'
split_sim_threshold = 0.8
embed_type = 'esm'
res_dir = filepaths['model_evals'] / "gnn"

# Configurtion stuff
hhps = HyperHyperParams(
    dataset_name=dataset_name,
    toc=toc,
    neg_multiple=neg_multiple,
    n_splits=n_splits,
    split_strategy=split_strategy,
    embed_type=embed_type,
    seed=seed,
    split_sim_threshold=split_sim_threshold,
)

# Create grid search object
gs = BatchGridSearch(
    hhps=hhps,
    res_dir=res_dir,
)

# Choose hyperparameters for grid search
hps = {
    'n_epochs':[25], # int
    'pred_head':['dot_sig'], # 'binary' | 'dot_sig'
    'message_passing':['bondwise'], # 'bondwise' | 'bondwise_dict' | None
    'agg':['last'], # 'mean' | 'last' | 'attention' | None
    'd_h_encoder':[300], # int
    'model':['mpnn_dim_red'], # 'mpnn' | 'mpnn_dim_red' | 'ffn' | 'linear'
    'featurizer':['rxn_rc'], # 'rxn_simple' | 'rxn_rc' | 'mfp'
    'encoder_depth':[4], # int | None
}

# Slurm stuff
batch_script = BatchScript(
    allocation=allocation,
    partition=partition,
    mem=mem,
    time=time,
    script=fit_script
)

# Run grid search
gs.run(
    hps=hps,
    batch_script=batch_script,
)