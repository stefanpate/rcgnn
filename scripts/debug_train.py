import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import pandas as pd
from lightning import pytorch as pl
from tqdm import tqdm

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
    print("loading data...")
    for i in range(cfg.data.n_splits):
        train_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet"
        )
        train_data['protein_embedding'] = train_data['protein_embedding'].apply(lambda x : np.array(x))

        print("downsampling negatives...")
        downsample_negatives(train_data, 1, rng) # Inner fold train are oversampled

        print("featurizing data...")
        val_data, val_dataset, val_dataloader, featurizer = featurize_data(
            cfg=cfg,
            rng=rng,
            train_data=None,
            val_data=train_data,
            shuffle_val=False,
        )

        # # Construct model
        # print("constructing model...")
        # embed_dim = train_data.loc[0, 'protein_embedding'].shape[0]
        # model = construct_model(cfg, embed_dim, featurizer, device)

        for (_, r), p in zip(val_data.iterrows(), val_dataset):
            f = p

if __name__ == '__main__':
    main()