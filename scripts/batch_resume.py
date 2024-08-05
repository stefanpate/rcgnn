import pandas as pd
from catalytic_function.cross_validation import BatchGridSearch, BatchScript, HyperHyperParams
from dataclasses import fields
from math import isnan

def fix_pd(hp_dict):
    to_fix = [
        'encoder_depth',
        'embed_dim',
        'seed',
        'n_epochs',
        'd_h_encoder',
        'n_splits',
        'neg_multiple',
        'message_passing',
        'agg',
    ]
    
    for elt in to_fix:
        if elt not in hp_dict:
            continue
        elif type(hp_dict[elt]) is str:
            continue
        elif isnan(hp_dict[elt]):
            hp_dict[elt] = None
        else:
            hp_dict[elt]  = int(hp_dict[elt])
    
    return hp_dict

# Args
allocation = 'b1039'
partition = 'b1039'
mem = '12G' # 12G
time = '12' # Hours 12
fit_script = 'two_channel_fit.py'
batch_script = BatchScript(allocation=allocation, partition=partition, mem=mem, time=time, script=fit_script)
res_dir = "/projects/p30041/spn1560/hiec/artifacts/model_evals/gnn"

# Old hp_idxs : total epochs to train up to
hp_idx_epochs = {
    88:25,
    89:25,
    90:25,
    91:25,
    92:25,
}

experiments = pd.read_csv(f"{res_dir}/experiments.csv", sep='\t', index_col=0)
to_resume = experiments.loc[hp_idx_epochs.keys()]
by = [field.name for field in fields(HyperHyperParams)]
gb = to_resume.groupby(by=by)
other_columns = [col for col in experiments.columns if col not in by]

# Chunk hps into groups w/ shared hyper hps
hhp_args = []
hps = []
chkpt_idxs = [] # Where to load model ckpt from
for hhp_vals, group in gb:
    hhp_args.append({k: v for k,v in zip(by, hhp_vals)})
    hp_chunk = []
    chckpt_chunk = []
    for hp_idx, row in group.iterrows():
        tmp = {col : row[col] for col in other_columns}
        tmp['n_epochs'] = hp_idx_epochs[hp_idx] # Resumed fit should have all same hps EXCEPT n_epochs
        hp_chunk.append(tmp)
        chckpt_chunk.append(hp_idx)

    chkpt_idxs.append(chckpt_chunk)
    hps.append(hp_chunk)

# Run bsg
for hhp, hp, chkpt in zip(hhp_args, hps, chkpt_idxs):
    hp = [fix_pd(elt) for elt in hp]
    hhp = fix_pd(hhp)
    hhps = HyperHyperParams(**hhp)
    gs = BatchGridSearch(hhps, res_dir)
    gs.resume(hp, batch_script, chkpt)