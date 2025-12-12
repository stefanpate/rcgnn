import hydra
from hydra import initialize, compose
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import pandas as pd
from lightning import pytorch as pl
import mlflow

from src.similarity import load_similarity_matrix
from src.ml_utils import (
    featurize_data,
    construct_model,
    mlflow_to_omegaconf
)

root_dir = Path(__file__).parent.parent.resolve()
@hydra.main(version_base=None, config_path=f"{root_dir}/configs", config_name="predict")
def main(outer_cfg: DictConfig):
    mlflow.set_tracking_uri(outer_cfg.filepaths.tracking_uri)
    run_data = mlflow.get_run(run_id=outer_cfg.run_id)
    cfg, artifacts_path = mlflow_to_omegaconf(run_data)
    run_path = artifacts_path.parent
    run_path = Path(outer_cfg.filepaths.runs) / run_path.parts[-2] / run_path.parts[-1] # Hack to change filepath to current machine
    cfg['filepaths'] = outer_cfg.filepaths

    if outer_cfg.external_data:
        cfg.data = OmegaConf.load(f"{root_dir}/configs/data/{outer_cfg.external_data}.yaml")

        if outer_cfg.negative_sampling:
            cfg.data.negative_sampling = outer_cfg.negative_sampling
        
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
    
    # Assemble output
    target_output = val_data.loc[:, ["protein_idx", "reaction_idx", "pid", "rid", "y"]]
    target_output.loc[:, "logits"] = logits
    
    # Get max sims
    sim = cfg.data.split_strategy if cfg.data.split_strategy != 'homology' else 'gsi'
    try:
        S = load_similarity_matrix(
            sim_path=Path(outer_cfg.filepaths.similarity_matrices),
            dataset=cfg.data.dataset,
            toc=cfg.data.toc,
            sim_metric=sim
        )
    except ValueError as e:
        print(e)
        if outer_cfg.external_data:
            target_output.to_parquet(f"{cfg.data.dataset}_{cfg.data.negative_sampling}_target_output.parquet", index=False)
        else:
            target_output.to_parquet("target_output.parquet", index=False)
        return
    
    if sim in ['rcmcs', 'drfp']:
        val_idx = target_output.loc[:, 'reaction_idx'].to_list()
    elif sim in ['gsi', 'esm']:
        val_idx = target_output.loc[:, 'protein_idx'].to_list()

    train_idx = [i for i in range(S.shape[0]) if i not in val_idx]
    max_sims = S[:, val_idx][train_idx].max(axis=0)
    target_output.loc[:, "max_sim"] = max_sims
    
    # Save
    if outer_cfg.external_data:
        target_output.to_parquet(f"{cfg.data.dataset}_{cfg.data.negative_sampling}_target_output.parquet", index=False)
    else:
        target_output.to_parquet("target_output.parquet", index=False)

if __name__ == '__main__':
    main()