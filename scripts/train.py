import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
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
    if cfg.data.split_idx == -2: # Train on full dataset
        more_train_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
        )
        more_train_data['protein_embedding'] = more_train_data['protein_embedding'].apply(lambda x : np.array(x))
        train_data = pd.concat(train_val_splits + [more_train_data], ignore_index=True)
        val_data = None
    elif cfg.data.split_idx == -1: # Test on outer fold
        val_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
        )
        val_data['protein_embedding'] = val_data['protein_embedding'].apply(lambda x : np.array(x))
        train_data = pd.concat(train_val_splits, ignore_index=True)
    else: # Test on inner fold
        train_data = pd.concat([train_val_splits[i] for i in range(cfg.data.n_splits) if i != cfg.data.split_idx], ignore_index=True)
        val_data = train_val_splits[cfg.data.split_idx]

    downsample_negatives(train_data, cfg.model.neg_multiple, rng) # Inner fold train are oversampled

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

    # Set up checkpointing
    k_val=3
    if cfg.model.n_epochs:
        checkpoint_callback = None
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/roc",
            save_top_k=1,
            mode="max",
            every_n_epochs=k_val,
            filename="best-checkpoint-{epoch:02d}-val_roc-{val/roc:.3f}",
            auto_insert_metric_name=False,
        )

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
            max_epochs=cfg.model.n_epochs or cfg.training.n_epochs, # Use hpoed model n_epochs if available else default
            logger=logger,
            callbacks=checkpoint_callback,
            check_val_every_n_epoch=k_val,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

if __name__ == '__main__':
    main()