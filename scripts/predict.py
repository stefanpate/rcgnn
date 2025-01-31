import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
from lightning import pytorch as pl

from src.ml_utils import (
    featurize_data,
    construct_model
)

def downsample_negatives(data: pd.DataFrame, neg_multiple: int, rng: np.random.Generator):
    neg_idxs = data[data['y'] == 0].index
    n_to_rm = len(neg_idxs) - (len(data[data['y'] == 1]) * neg_multiple)
    idx_to_rm = rng.choice(neg_idxs, n_to_rm, replace=False)
    data.drop(axis=0, index=idx_to_rm, inplace=True)

@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    if cfg.data.split_idx == -1: # Test on outer fold
        val_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
        )
    else:
        val_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{cfg.data.split_idx}.parquet"
        )
    val_data['protein_embedding'] = val_data['protein_embedding'].apply(lambda x : np.array(x))

    downsample_negatives(val_data, 1, rng)

    _, val_dataloader, featurizer = featurize_data(
        cfg=cfg,
        rng=rng,
        val_data=val_data,
    )

    # Construct model
    embed_dim = val_data.loc[0, 'protein_embedding'].shape[0]
    ckpt_dir = Path(cfg.filepaths.results) / 'runs' / str(cfg.exp_id) / cfg.model_id / 'checkpoints' 
    ckpt = ckpt_dir / next(ckpt_dir.glob("*.ckpt"))
    model = construct_model(cfg, embed_dim, featurizer, device, ckpt=ckpt)

    for batch in val_dataloader:
        losses = model._evaluate_batch(batch)
        print({f"val/{m.alias}": l for m, l in zip(model.metrics, losses)})

    # Predict
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )
        test_preds = trainer.predict(model, val_dataloader)

    # Save
    logits = np.vstack(test_preds).reshape(-1,)
    np.save(f"logits_{cfg.data.split_idx}", logits)

if __name__ == '__main__':
    main()