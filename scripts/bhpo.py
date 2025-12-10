import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import optuna
import logging
import sys
from optuna.integration import PyTorchLightningPruningCallback
import numpy as np
from functools import partial
import pickle
import torch
from lightning import pytorch as pl
from src.ml_utils import (
    featurize_data,
    construct_model,
    downsample_negatives
)

current_dir = Path(__file__).parent.parent.resolve()
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
log = logging.getLogger(__name__)

def get_optuna_hp(group: str, hp_name: str, hp: dict, trial: optuna.trial.Trial):
    combined_key = f"{group}/{hp_name}"
    if hp['type'] == "categorical":
        return trial.suggest_categorical(combined_key, hp['values'])
    elif hp['type'] == "int":
        return trial.suggest_int(combined_key, hp['values'][0], hp['values'][1])
    elif hp['type'] == "int_log":
        return trial.suggest_int(combined_key, hp['values'][0], hp['values'][1], log=True)
    elif hp['type'] == "float":
        return trial.suggest_float(combined_key, hp['values'][0], hp['values'][1])
    elif hp['type'] == "float_log":
        return trial.suggest_float(combined_key, hp['values'][0], hp['values'][1], log=True)

def objective(trial: optuna.trial.Trial, train_data: pd.DataFrame, val_data: pd.DataFrame, cfg: DictConfig, rng: np.random.Generator) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_monitor = "/".join(cfg.objective.split("_")) # Needs to be no / for filepaths, now needs to be /

    # hyperparams = {}
    for group, hps in cfg.hp_bounds.items():
        for hp_name, _hp in hps.items():
            if group == "model" and hp_name == "model" and cfg.model.name not in ["drfp", "mfp", "rxnfp"]:
                continue # there's only a choice to be made for the fp models (linear v ffn)
            hp = get_optuna_hp(group, hp_name, _hp, trial)
            cfg[group][hp_name] = hp
      
    # Prepare data
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

    # Train
    trainer = pl.Trainer(
        enable_progress_bar=True,
        enable_checkpointing=False,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=to_monitor)],
        accelerator="auto",
        devices=1,
        max_epochs=cfg.training.n_epochs, # number of epochs to train for
        logger=False
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    return trainer.callback_metrics[to_monitor].item()

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="bhpo")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)
    
    # Load data
    log.info("Loading & preparing data")
    train_val_splits = []
    for i in range(cfg.data.n_splits):
        split = pd.read_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet"
        )
        split['protein_embedding'] = split['protein_embedding'].apply(lambda x : np.array(x))
        train_val_splits.append(split)

    # Arrange data
    train_data = pd.concat([train_val_splits[i] for i in range(cfg.data.n_splits) if i != cfg.data.split_idx], ignore_index=True)
    val_data = train_val_splits[cfg.data.split_idx]

    # Optimize hyperparameters
    log.info("Optimizing hyperparameters")
    _objective = partial(
        objective,
        train_data=train_data,
        val_data=val_data,
        cfg=cfg,
        rng=rng,
    )
    sampler_path = Path(f"{cfg.study_name}_sampler.pkl")
    
    if sampler_path.exists():
        log.info(f"Loading sampler from {sampler_path}")
        sampler = pickle.load(open(sampler_path, "rb"))
    else:
        log.info(f"Creating new sampler seeded with {cfg.hpo_seed}")
        sampler = optuna.samplers.TPESampler(seed=cfg.hpo_seed)

    study = optuna.create_study(
        direction=cfg.direction,
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(),
        study_name=cfg.study_name,
        storage=f"sqlite:///{cfg.study_name}.db",
        load_if_exists=True
    )
    study.optimize(
        _objective,
        n_trials=cfg.n_trials,
        timeout=cfg.timeout # seconds
    )

    log.info("Best trial:")
    trial = study.best_trial

    log.info(f"  {cfg.objective}: {trial.value}")

    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info(f"    {key}: {value}")

    with open(sampler_path, "wb") as f:
        pickle.dump(study.sampler, f)
    
if __name__ == "__main__":
    main()
