import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from chemprop.data import build_dataloader
import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from lightning import pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from src.utils import load_json
import src.nn
import src.metrics
from src.model import (
    MPNNDimRed,
    TwoChannelFFN,
    TwoChannelLinear
)
from src.data import (
    RxnRCDataset,
    MFPDataset,
    mfp_build_dataloader,
    RxnRCDatapoint
)
from src.featurizer import (  
    SimpleReactionMolGraphFeaturizer,
    RCVNReactionMolGraphFeaturizer,
    ReactionMorganFeaturizer,
    MultiHotAtomFeaturizer,
    MultiHotBondFeaturizer
)

# Featurizers +
featurizers = {
    'rxn_simple': (RxnRCDataset, SimpleReactionMolGraphFeaturizer, build_dataloader),
    'rxn_rc': (RxnRCDataset, RCVNReactionMolGraphFeaturizer, build_dataloader),
    'mfp': (MFPDataset, ReactionMorganFeaturizer, mfp_build_dataloader)
}

def downsample_negatives(data: pd.DataFrame, neg_multiple: int, rng: np.random.Generator):
    neg_idxs = data[data['y'] == 0].index
    n_to_rm = len(neg_idxs) - (len(data[data['y'] == 1]) * neg_multiple)
    idx_to_rm = rng.choice(neg_idxs, n_to_rm, replace=False)
    return data.drop(axis=0, index=idx_to_rm, inplace=False)

def construct_featurizer(cfg):
    datapoint_from_smi = RxnRCDatapoint.from_smi
    dataset_base, featurizer_base, generate_dataloader = featurizers[cfg.model.featurizer]
    if cfg.model.featurizer == 'mfp':
        featurizer = featurizer_base()
    else:
        featurizer = featurizer_base(
            atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
            bond_featurizer=MultiHotBondFeaturizer()
        )

    return featurizer, datapoint_from_smi, dataset_base, generate_dataloader

def featurize_data(train_data: pd.DataFrame, val_data: pd.DataFrame, cfg: DictConfig):
    featurizer, datapoint_from_smi, dataset_base, generate_dataloader = construct_featurizer(cfg)
    train_datapoints = []
    for _, row in train_data.iterrows():
        y = np.array([row['y']]).astype(np.float32)
        train_datapoints.append(datapoint_from_smi(smarts=row['smarts'], reaction_center=row['reaction_center'], y=y, x_d=row['protein_embedding']))

    val_datapoints = []
    for _, row in val_data.iterrows():
        y = np.array([row['y']]).astype(np.float32)
        val_datapoints.append(datapoint_from_smi(smarts=row['smarts'], reaction_center=row['reaction_center'], y=y, x_d=row['protein_embedding']))

    train_dataset = dataset_base(train_datapoints, featurizer=featurizer)
    val_dataset = dataset_base(val_datapoints, featurizer=featurizer)

    train_dataloader = generate_dataloader(train_dataset, shuffle=True, seed=cfg.data.seed)
    val_dataloader = generate_dataloader(val_dataset, shuffle=False)

    return train_dataloader, val_dataloader, featurizer

@hydra.main(version_base=None, config_path="../configs", config_name="cross_val")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_val_splits = []
    for i in range(cfg.data.n_splits):
        split = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet"
        )
        split['protein_embedding'] = split['protein_embedding'].apply(lambda x : np.array(x))
        train_val_splits.append(split)

    # Arrange data
    if cfg.data.split_idx == -1: # Test on outer fold
        val_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
        )
        val_data['protein_embedding'] = val_data['protein_embedding'].apply(lambda x : np.array(x))
        train_data = pd.concat(train_val_splits, ignore_index=True)
    else: # Test on inner fold
        train_data = pd.concat([train_val_splits[i] for i in range(cfg.data.n_splits) if i != cfg.data.split_idx], ignore_index=True)
        val_data = train_val_splits[cfg.data.split_idx]

    # Downsample negatives   
    train_data = downsample_negatives(train_data, cfg.data.neg_multiple, rng)
    val_data = downsample_negatives(val_data, 1, rng)

    train_dataloader, val_dataloader, featurizer = featurize_data(
        train_data=train_data,
        val_data=val_data,
        cfg=cfg
    )

    # Construct model
    pos_weight = torch.ones([1]) * cfg.data.neg_multiple * cfg.training.pos_multiplier
    pos_weight = pos_weight.to(device)
    agg = getattr(src.nn, cfg.model.agg)() if cfg.model.agg else None
    pred_head = getattr(src.nn, cfg.model.pred_head)(
        input_dim=cfg.model.d_h_encoder * 2,
        criterion = src.nn.WeightedBCELoss(pos_weight=pos_weight)
    )
    metrics = [getattr(src.metrics, m)() for m in cfg.training.metrics]

    if cfg.model.message_passing:
        dv, de = featurizer.shape
        mp = getattr(src.nn, cfg.model.message_passing)(
            d_v=dv,
            d_e=de,
            d_h=cfg.model.d_h_encoder,
            depth=cfg.model.encoder_depth
        )

    # TODO streamline model api, get rid of LinDimRed
    # NOTE you can use hydra.utils.instantiate and partial to move
    # some of this up to configs
    embed_dim = train_data.loc[0, 'protein_embedding'].shape[0]
    if cfg.model.model == 'mpnn_dim_red':
        model = MPNNDimRed(
            reduce_X_d=src.nn.LinDimRed(d_in=embed_dim, d_out=cfg.model.d_h_encoder),
            message_passing=mp,
            agg=agg,
            predictor=pred_head,
            metrics=metrics
        )
    elif cfg.model.model == 'ffn':
        model = TwoChannelFFN(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=cfg.model.d_h_encoder,
            encoder_depth=cfg.model.encoder_depth,
            predictor=pred_head,
            metrics=metrics
        )
    elif cfg.model.model == 'linear':
        model = TwoChannelLinear(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=cfg.model.d_h_encoder,
            predictor=pred_head,
            metrics=metrics
    )
     
    # Track
    logger = MLFlowLogger(
        experiment_name=cfg.exp or "Default",
        save_dir=cfg.filepaths.runs,
        log_model=True,
    )
    mlflow.set_experiment(experiment_id=logger.experiment_id)

    # Train
    with mlflow.start_run(run_id=logger.run_id):
        flat_resolved_cfg = pd.json_normalize(
            {k: v for k,v in OmegaConf.to_container(cfg, resolve=True).items() if k != 'filepaths'}, # Resolved interpolated values
            sep='/'
        ).to_dict(orient='records')[0]
        mlflow.log_params(flat_resolved_cfg)
        trainer = pl.Trainer(
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=cfg.training.n_epochs, # number of epochs to train for
            logger=logger
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

if __name__ == '__main__':
    main()