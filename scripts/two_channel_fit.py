'''
Fit "two channel" (reaction gnn + esm) model
'''

from chemprop.data import build_dataloader
from chemprop.models import MPNN
from chemprop.nn import MeanAggregation, BinaryClassificationFFN, BondMessagePassing
from src.utils import load_known_rxns, construct_sparse_adj_mat, load_data_split, load_hps_from_scratch, save_json, append_hp_yaml, ensure_dirs
from src.featurizer import RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer
from src.nn import LastAggregation, DotSig, LinDimRed
from src.model import MPNNDimRed
from src.data import RxnRCDatapoint, RxnRCDataset
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from argparse import ArgumentParser
import torch
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# Aggregation fcns
agg_dict = {
    'last':LastAggregation,
    'mean':MeanAggregation,
}

# Prediction heads
pred_head_dict = {
    'binary':BinaryClassificationFFN,
    'dot_sig':DotSig
}

# For evaluation
scorers = {
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred),
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred),
    'accuracy': accuracy_score
}

# Parse CL args
parser = ArgumentParser()
parser.add_argument("-d", "--dataset-name", type=str)
parser.add_argument("-t", "--toc", type=str)
parser.add_argument("-m", "--sample_embed_type", type=str)
parser.add_argument("-e", "--seed", type=int)
parser.add_argument("-n", "--n-splits", type=int)
parser.add_argument("-s", "--split-idx", type=int)
parser.add_argument("-p", "--hp-idx", type=int)
parser.add_argument("-g", "--gs-name", type=str)

args = parser.parse_args()

dataset_name = args.dataset_name
toc = args.toc
sample_embed_type = args.sample_embed_type
seed = args.seed
n_splits = args.n_splits
split_idx = args.split_idx
hp_idx = args.hp_idx
gs_name = args.gs_name

print("Loading hyperparameters")
hps = load_hps_from_scratch(gs_name, hp_idx) # Load hyperparams

# Parse hyperparams
d_prot = hps['d_prot'] # Protein embed dimension
d_h_mpnn = hps['d_h_mpnn'] # Hidden layer of message passing
n_epochs = hps['n_epochs']
neg_multiple = hps['neg_multiple']

# Filenames
eval_dir = f"/projects/p30041/spn1560/hiec/artifacts/model_evals/gnn/{dataset_name}_{toc}_{gs_name}"
exp_name_batch = f"{hp_idx}_hp_idx_split_{split_idx+1}_of_{n_splits}" # Name this experiment

adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset_name, toc) # Load adj mat

# Load data split
print("Loading data")
train_data, test_data = load_data_split(dataset_name,
                                                  toc,
                                                  sample_embed_type,
                                                  n_splits,
                                                  seed,
                                                  idx_sample,
                                                  idx_feature,
                                                  split_idx,
                                                  neg_multiple
                                                  )

known_rxns = load_known_rxns(f"../data/{dataset_name}/known_rxns_{toc}.json") # Load reaction dataset

# Init featurizer
featurizer = RCVNReactionMolGraphFeaturizer(
    atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
    bond_featurizer=MultiHotBondFeaturizer()
)

# Prep data
print("Constructing datasets & dataloaders")
datapoints_train = []
for row in train_data:
    rxn = known_rxns[row['feature']]
    y = np.array([row['y']])
    datapoints_train.append(RxnRCDatapoint.from_smi(rxn, y=y, x_d=row['sample_embed']))

datapoints_test = []
for row in test_data:
    rxn = known_rxns[row['feature']]
    y = np.array([row['y']])
    datapoints_test.append(RxnRCDatapoint.from_smi(rxn, y=y, x_d=row['sample_embed']))

dataset_train = RxnRCDataset(datapoints_train, featurizer=featurizer)
dataset_test = RxnRCDataset(datapoints_test, featurizer=featurizer)

data_loader_train = build_dataloader(dataset_train, shuffle=False)
data_loader_test = build_dataloader(dataset_test, shuffle=False)

# Construct model
print("Building model")
dv, de = featurizer.shape
mp = BondMessagePassing(d_v=dv, d_e=de, d_h=d_h_mpnn)
pred_head = pred_head_dict[hps['pred_head']](input_dim=d_h_mpnn * 2)
agg = agg_dict[hps['agg']]()

if hps['model'] == 'mpnn':
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=pred_head,
    )
elif hps['model'] == 'mpnn_dim_red':
    model = MPNNDimRed(
        reduce_X_d=LinDimRed(d_in=d_prot, d_out=d_h_mpnn),
        message_passing=mp,
        agg=agg,
        predictor=pred_head,
    )

# Make trainer
logger = CSVLogger(eval_dir, name=exp_name_batch)
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
sup_dir = f"{eval_dir}/{exp_name_batch}"
versions = sorted([(fn, int(fn.split('_')[-1])) for fn in os.listdir(sup_dir)], key=lambda x : x[-1])
latest_version = versions[-1][0]
save_json(scores, f"{sup_dir}/{latest_version}/test_scores.json")

append_hp_yaml(hps, f"{sup_dir}/{latest_version}/hparams.yaml") # Append to hyperparam yaml file w/ hps from batch_fit