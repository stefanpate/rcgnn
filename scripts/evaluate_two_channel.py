import os
import torch
from chemprop import models
from chemprop.data import build_dataloader
import numpy as np
from lightning import pytorch as pl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from src.utils import load_known_rxns, save_json
from src.featurizer import RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer
from src.data import RxnRCDatapoint, RxnRCDataset

ds_name = 'sprhea'
toc = 'sp_folded_pt_20p'
gs_name = 'two_channel_mean_agg_binaryffn_pred_neg_1_shuffle_batches'
seed = 1234
neg_multiple = 1
known_rxns = load_known_rxns(f"../data/{ds_name}/known_rxns_{toc}.json")
n_epochs = 25
version = 'version_0'
n_splits = 2

model_pref = f"/projects/p30041/spn1560/hiec/artifacts/model_evals/gnn/{gs_name}_{ds_name}_{toc}_{n_epochs}_epochs_seed_1234_split_"
featurizer = RCVNReactionMolGraphFeaturizer(
    atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
    bond_featurizer=MultiHotBondFeaturizer()
)

for i in range(n_splits):
    model_dir = model_pref + f"{i+1}_of_{n_splits}/{version}/checkpoints"
    fn = os.listdir(model_dir)[0]
    model_path = model_dir + f"/{fn}"
    mpnn = models.MPNN.load_from_file(model_path, map_location=torch.device('cpu'))
    test_data_path = f"/scratch/spn1560/{ds_name}_{toc}_{n_splits}_splits_{seed}_seed_{neg_multiple}_neg_multiple_{i}_split_idx_test.npy"
    test_data = np.load(test_data_path)

    datapoints_test = []
    for row in test_data:
        rxn = known_rxns[row['feature']]
        y = np.array([row['y']])
        datapoints_test.append(RxnRCDatapoint.from_smi(rxn, y=y, x_d=row['sample_embed']))

    dataset_test = RxnRCDataset(datapoints_test, featurizer=featurizer)

    data_loader_test = build_dataloader(dataset_test, shuffle=False)

    # Predict
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1
        )
        test_preds = trainer.predict(mpnn, data_loader_test)

    # Evaluate
    logits = np.vstack(test_preds)
    y_pred = (logits > 0.5).astype(np.int64).reshape(-1,)
    y_true = test_data['y']

    scorers = {
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred),
        'accuracy': accuracy_score
    }

    scores = {}

    for k, scorer in scorers.items():
        scores[k] = scorer(y_true, y_pred)

    print(scores)

    save_json(scores, model_pref + f"{i+1}_of_{n_splits}/{version}/test_metrics.json")
    break