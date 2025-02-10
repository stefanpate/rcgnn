import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from src.ml_utils import (
    featurize_data,
    construct_model,
    downsample_negatives
)

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
        downsample_negatives(val_data, 1, rng) # Inner fold val are oversampled

    downsample_negatives(train_data, cfg.data.neg_multiple, rng) # Inner fold train are oversampled

    train_dataloader, val_dataloader, featurizer = featurize_data(
        cfg=cfg,
        rng=rng,
        train_data=train_data,
        val_data=val_data,
    )

    # Construct model
    embed_dim = train_data.loc[0, 'protein_embedding'].shape[0]
    model = construct_model(cfg, embed_dim, featurizer, device)

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