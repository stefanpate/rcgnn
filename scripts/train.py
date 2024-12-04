import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from chemprop.data import build_dataloader
import chemprop.nn
import torch
import numpy as np
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from src.utils import load_json
import src.nn
import src.metrics
from src.model import (
    MPNNDimRed,
    TwoChannelFFN,
    TwoChannelLinear
)
from src.data import (
    RxnRCDataset,
    MFPDataset,
    mfp_build_dataloader,
    RxnRCDatapoint
)
from src.featurizer import (  
    SimpleReactionMolGraphFeaturizer,
    RCVNReactionMolGraphFeaturizer,
    ReactionMorganFeaturizer,
    MultiHotAtomFeaturizer,
    MultiHotBondFeaturizer
)

from src.cross_validation import load_data_split


# Featurizers +
featurizers = {
    'rxn_simple': (RxnRCDataset, SimpleReactionMolGraphFeaturizer, build_dataloader),
    'rxn_rc': (RxnRCDataset, RCVNReactionMolGraphFeaturizer, build_dataloader),
    'mfp': (MFPDataset, ReactionMorganFeaturizer, mfp_build_dataloader)
}

def construct_featurizer(cfg):
    datapoint_from_smi = RxnRCDatapoint.from_smi
    dataset_base, featurizer_base, generate_dataloader = featurizers[cfg.model.featurizer]
    if cfg.model.featurizer == 'mfp':
        featurizer = featurizer_base()
    else:
        featurizer = featurizer_base(
            atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
            bond_featurizer=MultiHotBondFeaturizer()
        )

    return featurizer, datapoint_from_smi, dataset_base, generate_dataloader

def featurize_data(
        train_data,
        test_data,
        reactions,
        cfg
    ):
    featurizer, datapoint_from_smi, dataset_base, generate_dataloader = construct_featurizer(cfg)
    train_datapoints = []
    for row in train_data:
        rxn = reactions[row['feature']]
        y = np.array([row['y']]).astype(np.float32)
        train_datapoints.append(datapoint_from_smi(rxn, y=y, x_d=row['sample_embed']))

    test_datapoints = []
    for row in test_data:
        rxn = reactions[row['feature']]
        y = np.array([row['y']]).astype(np.float32)
        test_datapoints.append(datapoint_from_smi(rxn, y=y, x_d=row['sample_embed']))

    train_dataset = dataset_base(train_datapoints, featurizer=featurizer)
    test_dataset = dataset_base(test_datapoints, featurizer=featurizer)

    train_dataloader = generate_dataloader(train_dataset, shuffle=True)
    val_dataloader = generate_dataloader(test_dataset, shuffle=False)

    return train_dataloader, val_dataloader, featurizer

@hydra.main(version_base=None, config_path="../configs", config_name="cross_val")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 1280 # TODO

    # Load data
    reactions = load_json(Path(cfg.filepaths.data) / cfg.data.dataset / (cfg.data.toc + ".json")) # TODO eliminate by saving smiles and rcs to npy files

    train_data, test_data = load_data_split(
        split_idx=cfg.data.split_idx,
        scratch_path=Path(cfg.filepaths.scratch) / cfg.data.subdir_patt
    )

    train_dataloader, val_dataloader, featurizer = featurize_data(
        train_data=train_data,
        test_data=test_data,
        reactions=reactions,
        cfg=cfg
    )

    # Construct model
    pos_weight = torch.ones([1]) * cfg.data.neg_multiple * cfg.training.pos_multiplier
    pos_weight = pos_weight.to(device)
    agg = getattr(src.nn, cfg.model.agg)()
    pred_head = getattr(src.nn, cfg.model.pred_head)(
        input_dim=cfg.model.d_h_encoder * 2,
        criterion = src.nn.WeightedBCELoss(pos_weight=pos_weight)
    )
    metrics = [getattr(src.metrics, m)() for m in cfg.training.metrics]
    dv, de = featurizer.shape

    if cfg.model.message_passing:
        mp = getattr(src.nn, cfg.model.message_passing)(
            d_v=dv,
            d_e=de,
            d_h=cfg.model.d_h_encoder,
            depth=cfg.model.encoder_depth
        )

    # TODO streamline model api, get rid of LinDimRed
    # NOTE you can use hydra.utils.instantiate and partial to move
    # some of this up to configs
    if cfg.model.model == 'mpnn_dim_red':
        model = MPNNDimRed(
            reduce_X_d=src.nn.LinDimRed(d_in=embed_dim, d_out=cfg.model.d_h_encoder),
            message_passing=mp,
            agg=agg,
            predictor=pred_head,
            metrics=metrics
        )
    elif cfg.model.model == 'ffn':
        model = TwoChannelFFN(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=cfg.model.d_h_encoder,
            encoder_depth=cfg.model.encoder_depth,
            predictor=pred_head,
            metrics=metrics
        )
    elif cfg.model.model == 'linear':
        model = TwoChannelLinear(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=cfg.model.d_h_encoder,
            predictor=pred_head,
            metrics=metrics
    )
     
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