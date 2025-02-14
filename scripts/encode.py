import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
import mlflow

from src.ml_utils import (
    featurize_data,
    construct_model,
    mlflow_to_omegaconf
)

@hydra.main(version_base=None, config_path="../configs", config_name="encode")
def main(outer_cfg: DictConfig):
    mlflow.set_tracking_uri(outer_cfg.tracking_uri)
    run_data = mlflow.get_run(run_id=outer_cfg.run_id)
    cfg, artifacts_path = mlflow_to_omegaconf(run_data)
    run_path = artifacts_path.parent

    rng = np.random.default_rng(seed=cfg.data.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    splits = []
    for i in range(cfg.data.n_splits):
        split = pd.read_parquet(
            Path(outer_cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet"
        )
        split['protein_embedding'] = split['protein_embedding'].apply(lambda x : np.array(x))
        splits.append(split)

    more_data = pd.read_parquet(
        Path(outer_cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
    )
    more_data['protein_embedding'] = more_data['protein_embedding'].apply(lambda x : np.array(x))
    splits.append(more_data)
    data = pd.concat(splits, ignore_index=True)
    data = data.loc[data['y'] == 1] # Take out negative datapoints

    _, dataloader, featurizer = featurize_data(
        cfg=cfg,
        rng=rng,
        val_data=data,
        shuffle_val=False
    )

    # Construct model
    embed_dim = data.loc[0, 'protein_embedding'].shape[0]
    ckpt_dir = run_path / 'checkpoints' 
    ckpt = ckpt_dir / next(ckpt_dir.glob("*.ckpt"))
    model = construct_model(cfg, embed_dim, featurizer, device, ckpt=ckpt)

    print("Encoding")
    with torch.no_grad():
        fingerprints = [
            model.fingerprint(bmg=batch.bmg, X_d=batch.X_d)
            for batch in dataloader
        ]
        fingerprints = torch.cat(fingerprints, 0)

    fingerprints = fingerprints.detach().numpy()

    # Save
    data["output_reaction_embeddings"] = list(fingerprints[: , : cfg.model.d_h_encoder])
    data["output_protein_embeddings"] = list(fingerprints[: , cfg.model.d_h_encoder :])
    p_embeds = data.loc[:, ["protein_idx", "pid", "output_protein_embeddings"]]
    r_embeds = data.loc[:, ["reaction_idx", "rid", "output_reaction_embeddings"]]
    p_embeds.drop_duplicates(inplace=True, subset="pid")
    r_embeds.drop_duplicates(inplace=True, subset="rid")
    p_embeds.rename(columns={'output_protein_embeddings': "embedding"}, inplace=True)
    r_embeds.rename(columns={'output_reaction_embeddings': "embedding"}, inplace=True)
    p_embeds.to_parquet("protein_embeddings.parquet", index=False)
    r_embeds.to_parquet("reaction_embeddings.parquet", index=False)

if __name__ == '__main__':
    main()