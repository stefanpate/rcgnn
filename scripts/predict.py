import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
from lightning import pytorch as pl
import mlflow

from src.ml_utils import (
    featurize_data,
    construct_model,
    downsample_negatives,
    mlflow_to_omegaconf
)

@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(outer_cfg: DictConfig):
    mlflow.set_tracking_uri(outer_cfg.tracking_uri)
    run_data = mlflow.get_run(run_id=outer_cfg.run_id)
    cfg, artifacts_path = mlflow_to_omegaconf(run_data)
    run_path = artifacts_path.parent

    rng = np.random.default_rng(seed=cfg.data.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    if cfg.data.split_idx == -1: # Test on outer fold
        val_data = pd.read_parquet(
            Path(outer_cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
        )
    else:
        val_data = pd.read_parquet(
            Path(outer_cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{cfg.data.split_idx}.parquet"
        )
        downsample_negatives(val_data, 1, rng) # Inner fold val are oversampled

    val_data['protein_embedding'] = val_data['protein_embedding'].apply(lambda x : np.array(x))

    _, val_dataloader, featurizer = featurize_data(
        cfg=cfg,
        rng=rng,
        val_data=val_data,
        shuffle_val=False
    )

    # Construct model
    embed_dim = val_data.loc[0, 'protein_embedding'].shape[0]
    ckpt_dir = run_path / 'checkpoints' 
    ckpt = ckpt_dir / next(ckpt_dir.glob("*.ckpt"))
    model = construct_model(cfg, embed_dim, featurizer, device, ckpt=ckpt)

    # Predict
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )
        test_preds = trainer.predict(model, val_dataloader)

    logits = np.vstack(test_preds).reshape(-1,)
    
    # Save
    target_output = val_data.loc[:, ["protein_idx", "reaction_idx", "pid", "rid", "y"]]
    target_output.loc[:, "logits"] = logits
    target_output.to_parquet("target_output.parquet", index=False)

if __name__ == '__main__':
    main()