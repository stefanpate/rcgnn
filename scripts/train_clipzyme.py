import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from src.clip import EnzymeReactionCLIP, ClipDataset, clip_collate
import torch
from torch.utils.data import DataLoader

def downsample_negatives(data: pd.DataFrame, neg_multiple: int, rng: np.random.Generator):
    neg_idxs = data[data['y'] == 0].index
    n_to_rm = len(neg_idxs) - (len(data[data['y'] == 1]) * neg_multiple)
    idx_to_rm = rng.choice(neg_idxs, n_to_rm, replace=False)
    data.drop(axis=0, index=idx_to_rm, inplace=True)

@hydra.main(version_base=None, config_path="../configs", config_name="train_clipzyme")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)

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
        version = "full_data"
        more_train_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
        )
        more_train_data['protein_embedding'] = more_train_data['protein_embedding'].apply(lambda x : np.array(x))
        train_data = pd.concat(train_val_splits + [more_train_data], ignore_index=True)
        val_data = None
    elif cfg.data.split_idx == -1: # Test on outer fold
        version = "outer_fold"
        val_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
        )
        val_data['protein_embedding'] = val_data['protein_embedding'].apply(lambda x : np.array(x))
        train_data = pd.concat(train_val_splits, ignore_index=True)
    else: # Test on inner fold
        version = f"inner_fold_{cfg.data.split_idx + 1}_of_{cfg.data.n_splits}"
        train_data = pd.concat([train_val_splits[i] for i in range(cfg.data.n_splits) if i != cfg.data.split_idx], ignore_index=True)
        val_data = train_val_splits[cfg.data.split_idx]
        downsample_negatives(val_data, 1, rng) # Inner fold val are oversampled

    downsample_negatives(train_data, cfg.data.neg_multiple, rng) # Inner fold train are oversampled

    # Prepare data
    fmt_data = lambda df: (df['smarts'].tolist(), torch.tensor(np.stack(df['protein_embedding'].to_numpy())), torch.tensor(df['y'].to_numpy()).float().unsqueeze(1))
    train_reactions, train_proteins, train_targets = fmt_data(train_data)
    val_reactions, val_proteins, val_targets = (None, None, None) if val_data is None else fmt_data(val_data)

    train_dataset = ClipDataset(
        reactions=train_reactions,
        protein_embeddings=train_proteins,
        targets=train_targets
    )
    val_dataset = None if val_data is None else ClipDataset(
        reactions=val_reactions,
        protein_embeddings=val_proteins,
        targets=val_targets
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=clip_collate,
        batch_size=cfg.training.batch_size,
    )
    val_dataloader = None if val_data is None else DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=clip_collate,
        batch_size=cfg.training.batch_size,
    )

    # Construct model
    model = EnzymeReactionCLIP(
        model_hps=cfg.model,
        negative_multiple=cfg.data.neg_multiple,
        positive_multiplier=cfg.training.pos_multiplier,
    )

    # Track
    logger = CSVLogger(
        name=cfg.exp or "Default",
        save_dir=cfg.filepaths.runs,
        version=version,
    )

    # Train
    trainer = pl.Trainer(
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=cfg.training.n_epochs,
        logger=logger,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Predict
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )
        val_preds = trainer.predict(model, val_dataloader)

    logits = np.vstack(val_preds).reshape(-1,)
    
    # Save outputs
    target_output = val_data.loc[:, ["protein_idx", "reaction_idx", "pid", "rid", "y"]]
    target_output.loc[:, "logits"] = logits # These are not really logits but keep naming for consistency

    subdir = Path(f"{cfg.exp}/{version}")
    if not subdir.exists():
        subdir.mkdir(parents=True)

    target_output.to_parquet(subdir / "target_output.parquet", index=False)

if __name__ == '__main__':
    main()