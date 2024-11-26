from argparse import ArgumentParser
from src.filepaths import filepaths
from src.task import construct_featurizer, construct_model, featurize_data
from src.cross_validation import BatchGridSearch, HyperHyperParams
from src.utils import load_json, fix_hps_from_dataframe
import torch
import pandas as pd
import numpy as np
from lightning import pytorch as pl


def _predict_cv(args):
    res_dir = filepaths["model_evals"] / "gnn"
    experiments = pd.read_csv(filepath_or_buffer=res_dir / "experiments.csv", sep='\t', index_col=0)
    hps = experiments.loc[args.hp_idx, :].to_dict()
    hps = fix_hps_from_dataframe(hps)
    dataset_base, generate_dataloader, featurizer = construct_featurizer(hps)
    hhps = HyperHyperParams(
        dataset_name=hps['dataset_name'], toc=hps['toc'], neg_multiple=hps['neg_multiple'], n_splits=hps['n_splits'],
        split_strategy=hps['split_strategy'], embed_type=hps['embed_type'], seed=hps['seed'],
        split_sim_threshold=hps['split_sim_threshold'], embed_dim=hps['embed_dim']
    )
    gs = BatchGridSearch(hhps, res_dir=res_dir)
    X, y = gs.sample_negatives()
    split_guide = gs.split_data(X, y, do_save=True)
    known_rxns = load_json(filepaths['data'] / f"{hps['dataset_name']}/{hps['toc']}.json") # Load reaction dataset
    for split_idx in range(hps['n_splits']):
        _, test_data = gs.load_data_split(split_idx=split_idx)
        chkpt = (res_dir / f"{args.hp_idx}_hp_idx_split_{split_idx+1}_of_{hps['n_splits']}" / "version_0" / "checkpoints").glob("*.ckpt")
        chkpt = list(chkpt)[0]
        model = construct_model(hps, featurizer, hps['embed_dim'], chkpt)

        dataloader = featurize_data(test_data, known_rxns, featurizer, dataset_base, generate_dataloader)
        
        # Predict
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="auto",
                devices=1
            )
            test_preds = trainer.predict(model, dataloader)

        # Save
        logits = np.vstack(test_preds).reshape(-1,)
        y_pred = (logits > 0.5).astype(np.int64).reshape(-1,)
        y_true = test_data['y'].reshape(-1,)
        test_labels = split_guide.loc[(split_guide['train/test'] == 'test') & (split_guide['split_idx'] == split_idx), ['X1', 'X2']].copy()
        test_labels.reset_index(drop=True, inplace=True)
        test_labels.columns = ['protein', 'reaction']
        res_df = pd.DataFrame(data={"scores": logits, "y_hat": y_pred, "y_true": y_true})
        res_df = pd.concat((test_labels, res_df), axis=1)
        res_df.to_csv(res_dir / f"{args.hp_idx}_hp_idx_split_{split_idx+1}_of_{hps['n_splits']}" / "predictions.csv", sep='\t')


parser = ArgumentParser(description="Predict score on binary protein-reaction binary classification")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Predict with series of models from a cross validation
predict_cv = subparsers.add_parser("predict-cv", description="Predicts using series of models of same hyperparams from a kfold cross validation")
predict_cv.add_argument("hp_idx", type=int, help="Hyperparameter index in experiments csv")
predict_cv.set_defaults(func=_predict_cv)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()