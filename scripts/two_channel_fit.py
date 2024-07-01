'''
Fit "two channel" (reaction gnn + esm) model
'''

from chemprop import models, nn
from chemprop.data import build_dataloader
from src.utils import load_known_rxns, construct_sparse_adj_mat, ensure_dirs, load_data_split, load_hps_from_scratch
from src.featurizer import RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer
from src.nn import LastAggregation
from src.data import RxnRCDatapoint, RxnRCDataset
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from argparse import ArgumentParser

eval_dir = "/projects/p30041/spn1560/hiec/artifacts/model_evals/gnn"

agg_dict = {
    'last':LastAggregation(),
    'mean':nn.MeanAggregation(),
}

pred_head_dict = {
    'binary':nn.BinaryClassificationFFN,
}

# CLI parsing
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

adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset_name, toc) # Load adj mat

print("Loading hyperparameters")
hps = load_hps_from_scratch(gs_name, hp_idx) # Load hyperparams

# Parse hyperparams
d_prot = hps['d_prot'] # Protein embed dimension
d_h_mpnn = hps['d_h_mpnn'] # Hidden layer of message passing
n_epochs = hps['n_epochs']
pred_head = pred_head_dict[hps['pred_head']](input_dim=d_prot+d_h_mpnn)
agg = agg_dict[hps['agg']]
neg_multiple = hps['neg_multiple']

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

known_rxns = load_known_rxns(f"../data/{dataset_name}/known_rxns_{toc}.json")

# Init featurizer
featurizer = RCVNReactionMolGraphFeaturizer(
    atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
    bond_featurizer=MultiHotBondFeaturizer()
)

exp_name_batch = f"{gs_name}_{dataset_name}_{toc}_{n_epochs}_epochs_seed_{seed}_split_{split_idx+1}_of_{n_splits}"

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
mp = nn.BondMessagePassing(d_v=dv, d_e=de, d_h=d_h_mpnn)
mpnn = models.MPNN(
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
    model=mpnn,
    train_dataloaders=data_loader_train,
)