import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
from src.clip import EnzymeReactionCLIP, ClipDataset, clip_collate
import torch
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="train_clipzyme")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)

    # Load data
    for i in range(cfg.data.n_splits):
        print(f"loading data for split {i}...")
        train_data = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet"
        )
        train_data['protein_embedding'] = train_data['protein_embedding'].apply(lambda x : np.array(x))

        train_data = train_data.drop_duplicates(subset=['reaction_idx'])

        # Prepare data
        print("preparing data...")
        fmt_data = lambda df: (df['smarts'].tolist(), torch.tensor(np.stack(df['protein_embedding'].to_numpy())), torch.tensor(df['y'].to_numpy()).float().unsqueeze(1))
        train_reactions, train_proteins, train_targets = fmt_data(train_data)

        print("constructing dataset...")
        train_dataset = ClipDataset(
            reactions=train_reactions,
            protein_embeddings=train_proteins,
            targets=train_targets
        )

        for elt in tqdm(train_dataset, total=len(train_dataset), desc="checking data..."):
            f = elt
        

    # Track
if __name__ == '__main__':
    main()