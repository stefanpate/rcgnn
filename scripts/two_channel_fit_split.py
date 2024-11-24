'''
Fit "two channel" (reaction gnn + esm) model on a data split
'''

from chemprop.data import build_dataloader
from chemprop.models import MPNN
from chemprop.nn import MeanAggregation, BinaryClassificationFFN, BondMessagePassing

from src.config import filepaths
from src.utils import load_json, save_json
from src.featurizer import SimpleReactionMolGraphFeaturizer, RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer, ReactionMorganFeaturizer
from src.nn import LastAggregation, DotSig, LinDimRed, AttentionAggregation, BondMessagePassingDict, WeightedBCELoss
from src.model import MPNNDimRed, TwoChannelFFN, TwoChannelLinear
from src.data import RxnRCDatapoint, RxnRCDataset, MFPDataset, mfp_build_dataloader
from src.cross_validation import load_single_experiment, HyperHyperParams, BatchGridSearch

from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger

import numpy as np
from argparse import ArgumentParser
import torch
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

res_dir = filepaths['model_evals'] / "gnn" # TODO this shouldn't be here. Should be set by batch fit/resume

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

# Evaluation metrics
scorers = {
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred),
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred),
    'accuracy': accuracy_score
}

# Parse CL args
parser = ArgumentParser()
parser.add_argument("-s", "--split-idx", type=int)
parser.add_argument("-p", "--hp-idx", type=int)
parser.add_argument("-c", "--checkpoint-idx", type=int)

args = parser.parse_args()

split_idx = args.split_idx
hp_idx = args.hp_idx
chkpt_idx = args.checkpoint_idx

# Load hyperparams
print("Loading hyperparameters")
hps = load_single_experiment(hp_idx)
hhps = HyperHyperParams.from_single_experiment(hps)
hps = {k: v for k,v in hps.items() if k not in hhps.to_dict()} # Make hps, hhps mutually exclusive
gs = BatchGridSearch(hhps, res_dir=res_dir)

n_splits = hhps.n_splits
dataset_name = hhps.dataset_name
toc = hhps.toc

# Results directories
hp_split_dir = f"{hp_idx}_hp_idx_split_{split_idx+1}_of_{n_splits}"

# Load data split
print("Loading data")
train_data, test_data = gs.load_data_split(split_idx=split_idx)
known_rxns = load_json(filepaths['data'] / f"{dataset_name}/{toc}.json") # Load reaction dataset

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
embed_dim = hhps.embed_dim
encoder_depth = hps['encoder_depth']

if hps['message_passing']:
    mp = message_passers[hps['message_passing']](d_v=dv, d_e=de, d_h=d_h_encoder, depth=encoder_depth)

if hps['agg']:
    agg = aggs[hps['agg']](input_dim=d_h_encoder) if hps['agg'] == 'attention' else aggs[hps['agg']]()

pos_weight = torch.ones([1]) * hhps.neg_multiple
criterion = WeightedBCELoss(pos_weight=pos_weight)
pred_head = pred_heads[hps['pred_head']](input_dim=d_h_encoder * 2, criterion=criterion)

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

if chkpt_idx:
    chkpt_dir = res_dir / f"{chkpt_idx}_hp_idx_split_{split_idx+1}_of_{n_splits}/version_0/checkpoints"
    chkpt_file = os.listdir(chkpt_dir)[0]
    chkpt_path = chkpt_dir / f"{chkpt_file}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chkpt = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(chkpt['state_dict'])
    model.max_lr = 1e-4 # Constant lr
    epochs_completed = chkpt['epoch'] + 1
    n_epochs = hps['n_epochs'] - epochs_completed # To tell Trainer
else:
    n_epochs = hps['n_epochs']

# Make trainer
logger = CSVLogger(res_dir, name=hp_split_dir)
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
sup_dir = res_dir / f"{hp_split_dir}"
versions = sorted([(fn, int(fn.split('_')[-1])) for fn in os.listdir(sup_dir)], key=lambda x : x[-1])
latest_version = versions[-1][0]
save_json(scores, sup_dir / f"{latest_version}/test_scores.json")