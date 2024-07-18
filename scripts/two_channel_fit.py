'''
Fit "two channel" (reaction gnn + esm) model
'''

from chemprop.data import build_dataloader
from chemprop.models import MPNN
from chemprop.nn import MeanAggregation, BinaryClassificationFFN, BondMessagePassing

from src.utils import load_known_rxns, save_json
from src.featurizer import SimpleReactionMolGraphFeaturizer, RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer, ReactionMorganFeaturizer
from src.nn import LastAggregation, DotSig, LinDimRed, AttentionAggregation, BondMessagePassingDict
from src.model import MPNNDimRed, TwoChannelFFN, TwoChannelLinear
from src.data import RxnRCDatapoint, RxnRCDataset, MFPDataset, mfp_build_dataloader
from src.cross_validation import BatchGridSearch

from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger

import numpy as np
from argparse import ArgumentParser
import torch
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Aggregation fcns
aggs = {
    'last':LastAggregation,
    'mean':MeanAggregation,
    'attention':AttentionAggregation
}

# Prediction heads
pred_heads = {
    'binary':BinaryClassificationFFN,
    'dot_sig':DotSig
}

# Message passing
message_passers = {
    'bondwise':BondMessagePassing,
    'bondwise_dict':BondMessagePassingDict
}

# Evaluation metrics
scorers = {
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred),
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred),
    'accuracy': accuracy_score
}

# Featurizers +
featurizers = {
    'rxn_simple': (RxnRCDataset, SimpleReactionMolGraphFeaturizer, build_dataloader),
    'rxn_rc': (RxnRCDataset, RCVNReactionMolGraphFeaturizer, build_dataloader),
    'mfp': (MFPDataset, ReactionMorganFeaturizer, mfp_build_dataloader)
}

# Parse CL args
parser = ArgumentParser()
parser.add_argument("-d", "--dataset-name", type=str)
parser.add_argument("-t", "--toc", type=str)
parser.add_argument("-a", "--split-strategy", type=str)
parser.add_argument("-b", "--embed-type", type=str)
parser.add_argument("-r", "--threshold", type=float)
parser.add_argument("-e", "--seed", type=int)
parser.add_argument("-n", "--n-splits", type=int)
parser.add_argument("-m", "--neg-multiple", type=int)
parser.add_argument("-s", "--split-idx", type=int)
parser.add_argument("-p", "--hp-idx", type=int)
parser.add_argument("-g", "--gs-name", type=str)

args = parser.parse_args()

dataset_name = args.dataset_name
toc = args.toc
split_strategy = args.split_strategy
embed_type = args.embed_type
split_sim_threshold = args.threshold
seed = args.seed
n_splits = args.n_splits
neg_multiple = args.neg_multiple
split_idx = args.split_idx
hp_idx = args.hp_idx
gs_name = args.gs_name

gs = BatchGridSearch(
    dataset_name=dataset_name,
    toc=toc,
    neg_multiple=neg_multiple,
    gs_name=gs_name,
    n_splits=n_splits,
    split_strategy=split_strategy,
    embed_type=embed_type,
    split_sim_threshold=split_sim_threshold,
    seed=seed,
)

print("Loading hyperparameters")
hps = gs.load_hps_from_scratch(hp_idx) # Load hyperparams

# Results directories
gs_dir = gs.res_dir
hp_split_dir = f"{hp_idx}_hp_idx_split_{split_idx+1}_of_{n_splits}"

# Load data split
print("Loading data")
train_data, test_data = gs.load_data_split(split_idx=split_idx)
known_rxns = load_known_rxns(f"../data/{dataset_name}/known_rxns_{toc}.json") # Load reaction dataset

# Init featurizer
mfp_length = 2**10
mfp_radius = 2
datapoint_from_smi = RxnRCDatapoint.from_smi
dataset_base, featurizer_base, generate_dataloader = featurizers[hps['featurizer']]
if hps['featurizer'] == 'mfp':
    featurizer = featurizer_base(radius=mfp_radius, length=mfp_length)
else:
    featurizer = featurizer_base(
        atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
        bond_featurizer=MultiHotBondFeaturizer()
    )
    dv, de = featurizer.shape

# Prep data
print("Constructing datasets & dataloaders")
datapoints_train = []
for row in train_data:
    rxn = known_rxns[row['feature']]
    y = np.array([row['y']]).astype(np.float32)
    datapoints_train.append(datapoint_from_smi(rxn, y=y, x_d=row['sample_embed']))

datapoints_test = []
for row in test_data:
    rxn = known_rxns[row['feature']]
    y = np.array([row['y']]).astype(np.float32)
    datapoints_test.append(datapoint_from_smi(rxn, y=y, x_d=row['sample_embed']))

dataset_train = dataset_base(datapoints_train, featurizer=featurizer)
dataset_test = dataset_base(datapoints_test, featurizer=featurizer)

data_loader_train = generate_dataloader(dataset_train, shuffle=False)
data_loader_test = generate_dataloader(dataset_test, shuffle=False)

# Construct model
print("Building model")
d_h_encoder = hps['d_h_encoder'] # Hidden layer of message passing
embed_dim = gs.embed_dim
encoder_depth = hps['encoder_depth']

if hps['message_passing']:
    mp = message_passers[hps['message_passing']](d_v=dv, d_e=de, d_h=d_h_encoder, depth=encoder_depth)

if hps['agg']:
    agg = aggs[hps['agg']](input_dim=d_h_encoder) if hps['agg'] == 'attention' else aggs[hps['agg']]()

pred_head = pred_heads[hps['pred_head']](input_dim=d_h_encoder * 2)

if hps['model'] == 'mpnn':
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=pred_head,
    )
elif hps['model'] == 'mpnn_dim_red':
    model = MPNNDimRed(
        reduce_X_d=LinDimRed(d_in=embed_dim, d_out=d_h_encoder),
        message_passing=mp,
        agg=agg,
        predictor=pred_head,
    )
elif hps['model'] == 'ffn':
    model = TwoChannelFFN(
        d_rxn=mfp_length,
        d_prot=embed_dim,
        d_h=d_h_encoder,
        encoder_depth=encoder_depth,
        predictor=pred_head,
    )
elif hps['model'] == 'linear':
    model = TwoChannelLinear(
        d_rxn=mfp_length,
        d_prot=embed_dim,
        d_h=d_h_encoder,
        predictor=pred_head,
    )

# Make trainer
n_epochs = hps['n_epochs']
logger = CSVLogger(gs_dir, name=hp_split_dir)
trainer = pl.Trainer(
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=n_epochs, # number of epochs to train for
    logger=logger
)

# Train
print("Training")
trainer.fit(
    model=model,
    train_dataloaders=data_loader_train,
)

# Predict
print("Testing")
with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )
    test_preds = trainer.predict(model, data_loader_test)

# Evaluate
logits = np.vstack(test_preds)
y_pred = (logits > 0.5).astype(np.int64).reshape(-1,)
y_true = test_data['y']

scores = {}
for k, scorer in scorers.items():
    scores[k] = scorer(y_true, y_pred)

print(scores)
sup_dir = f"{gs_dir}/{hp_split_dir}"
versions = sorted([(fn, int(fn.split('_')[-1])) for fn in os.listdir(sup_dir)], key=lambda x : x[-1])
latest_version = versions[-1][0]
save_json(scores, f"{sup_dir}/{latest_version}/test_scores.json")