'''
Fit "two channel" (reaction gnn + esm) model on all data
'''

from chemprop.data import build_dataloader
from chemprop.models import MPNN
from chemprop.nn import MeanAggregation, BinaryClassificationFFN, BondMessagePassing

from src.utils import load_json, load_embed_matrix, construct_sparse_adj_mat
from src.featurizer import SimpleReactionMolGraphFeaturizer, RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer, ReactionMorganFeaturizer
from src.nn import LastAggregation, DotSig, LinDimRed, AttentionAggregation, BondMessagePassingDict
from src.model import MPNNDimRed, TwoChannelFFN, TwoChannelLinear
from src.data import RxnRCDatapoint, RxnRCDataset, MFPDataset, mfp_build_dataloader
from src.cross_validation import sample_negatives

from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger

import numpy as np
from argparse import ArgumentParser

# Parse CL args
parser = ArgumentParser()
parser.add_argument("model_name", type=str)
args = parser.parse_args()
model_name = args.model_name

hps = load_json(f"../artifacts/named_model_hps/{model_name}.json")
res_dir = "/projects/p30041/spn1560/hiec/artifacts/trained_models/gnn"

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

# Featurizers +
featurizers = {
    'rxn_simple': (RxnRCDataset, SimpleReactionMolGraphFeaturizer, build_dataloader),
    'rxn_rc': (RxnRCDataset, RCVNReactionMolGraphFeaturizer, build_dataloader),
    'mfp': (MFPDataset, ReactionMorganFeaturizer, mfp_build_dataloader)
}

# Load data
print("Loading data")
adj, idx_sample, idx_feature = construct_sparse_adj_mat(hps['dataset_name'], hps['toc'])
sample_idx = {v: k for k, v in idx_sample.items()}
positive_pairs = list(zip(*adj.nonzero()))
X, y = sample_negatives(positive_pairs, hps['neg_multiple'], hps['seed'])
embeds = load_embed_matrix(hps['dataset_name'], hps['toc'], hps['embed_type'], sample_idx, do_norm=False)
embed_dim = embeds.shape[1]
known_rxns = load_json(f"../data/{hps['dataset_name']}/{hps['toc']}.json") # Load reaction dataset

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

# Featurize data
print("Constructing datasets & dataloaders")
datapoints_train = []
for (i, j), label in zip(X, y):
    rxn = known_rxns[idx_feature[j]]
    label = np.array([label], dtype=np.float32)
    prot_embed = embeds[i, :]
    datapoints_train.append(datapoint_from_smi(rxn, y=label, x_d=prot_embed))

dataset_train = dataset_base(datapoints_train, featurizer=featurizer)
data_loader_train = generate_dataloader(dataset_train, shuffle=False)

# Construct model
print("Building model")
d_h_encoder = hps['d_h_encoder'] # Hidden layer of message passing
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

n_epochs = hps['n_epochs']

# Make trainer
logger = CSVLogger(res_dir, name=model_name)
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