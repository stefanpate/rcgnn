from chemprop import models, nn
from chemprop.data import build_dataloader
from src.utils import load_known_rxns, construct_sparse_adj_mat, ensure_dirs
from src.featurizer import RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer
from src.nn import LastAggregation
from src.data import RxnRCDatapoint, RxnRCDataset
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from sklearn.model_selection import train_test_split

# Args
eval_dir = "../artifacts/model_evals/gnn"
n_epochs = 100
dataset = 'sprhea'
toc = 'sp_folded_pt_rxns_x_min_ops'
krs = load_known_rxns("../data/sprhea/known_rxns_240310_v2_folded_protein_transcript.json")
seed = 1234

exp_name = f"rc_gnn_last_agg_{dataset}_{toc}_{n_epochs}_epochs_seed_{seed}"
ensure_dirs(eval_dir)

# Load data
adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset, toc)
feature_idx = {v:k for k,v in idx_feature.items()}
X = [krs[k] for k in list(krs.keys())]
min_rule_labels = ["_".join(sorted(elt['min_rules'])) for elt in X] # Rule names
min_rule_idxs = np.array([feature_idx[elt] for elt in min_rule_labels]) # Indices in adj mat
n_samples = len(X)
n_classes = len(feature_idx)
y = min_rule_idxs.reshape(-1,1) # Class index encoding

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

# Init featurizer
featurizer = RCVNReactionMolGraphFeaturizer(
    atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
    bond_featurizer=MultiHotBondFeaturizer()
)

# Construct dataset
datapoints_train = [RxnRCDatapoint.from_smi(kr, y=y_train[i]) for i, kr in enumerate(X_train)]
datapoints_test = [RxnRCDatapoint.from_smi(kr, y=y_test[i]) for i, kr in enumerate(X_test)]

dataset_train = RxnRCDataset(datapoints_train, featurizer=featurizer)
dataset_test = RxnRCDataset(datapoints_test, featurizer=featurizer)

data_loader_train = build_dataloader(dataset_train, shuffle=False)
data_loader_test = build_dataloader(dataset_test, shuffle=False)

# Construct model
dv, de = featurizer.shape
mp = nn.BondMessagePassing(d_v=dv, d_e=de)
agg = LastAggregation()
ffn = nn.MulticlassClassificationFFN(n_classes=n_classes)
mpnn = models.MPNN(
    message_passing=mp,
    agg=agg,
    predictor=ffn,
)

# Make trainer
logger = CSVLogger(eval_dir, name=exp_name)
trainer = pl.Trainer(
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=n_epochs, # number of epochs to train for
    logger=logger
)

# Train
trainer.fit(
    model=mpnn,
    train_dataloaders=data_loader_train,
)

# Test
results = trainer.test(mpnn, data_loader_test)

print('done')