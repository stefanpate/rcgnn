from omegaconf import DictConfig, OmegaConf
from mlflow.entities.run_data import RunData
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import src.nn
import src.metrics

from chemprop.data import (
    build_dataloader,
    ReactionDataset,
)

from src.model import (
    MPNNDimRed,
    TwoChannelFFN,
    TwoChannelLinear
)

from src.data import (
    RxnRCDataset,
    MFPDataset,
    mfp_build_dataloader,
    RxnRCDatapoint,
    PretrainedFPDataset,
    PretrainedDatapoint,
)

from src.featurizer import (  
    SimpleReactionMolGraphFeaturizer,
    RCVNReactionMolGraphFeaturizer,
    ReactionMorganFeaturizer,
    MultiHotAtomFeaturizer,
    MultiHotBondFeaturizer,
    cp_reaction_dp_from_smi,
    ReactionDRFPFeaturizer,
    PretrainedReactionFeaturizer,
)
from chemprop.featurizers import (
    CondensedGraphOfReactionFeaturizer,
    RxnMode,
)

featurizers = {
    'cgr': (ReactionDataset, CondensedGraphOfReactionFeaturizer, build_dataloader),
    'rxn_simple': (RxnRCDataset, SimpleReactionMolGraphFeaturizer, build_dataloader),
    'rxn_rc': (RxnRCDataset, RCVNReactionMolGraphFeaturizer, build_dataloader),
    'mfp': (MFPDataset, ReactionMorganFeaturizer, mfp_build_dataloader),
    'drfp': (MFPDataset, ReactionDRFPFeaturizer, mfp_build_dataloader),
    'rxnfp': (PretrainedFPDataset, PretrainedReactionFeaturizer, mfp_build_dataloader),
    'uni_rxn': (PretrainedFPDataset, PretrainedReactionFeaturizer, mfp_build_dataloader),
    'react_seq': (PretrainedFPDataset, PretrainedReactionFeaturizer, mfp_build_dataloader),
}

def construct_featurizer(cfg: DictConfig):
    
    if cfg.model.featurizer == 'cgr':
        datapoint_from_smi = cp_reaction_dp_from_smi
    elif cfg.model.featurizer in ['rxnfp', 'uni_rxn', 'react_seq']:
        datapoint_from_smi = PretrainedDatapoint.from_smi
    else:
        datapoint_from_smi = RxnRCDatapoint.from_smi

    dataset_base, featurizer_base, generate_dataloader = featurizers[cfg.model.featurizer]
    
    if cfg.model.featurizer == 'mfp':
        featurizer = featurizer_base(radius=cfg.model.radius, length=cfg.model.vec_len)
    elif cfg.model.featurizer == 'drfp':
        featurizer = featurizer_base(length=cfg.model.vec_len)
    elif cfg.model.featurizer == 'cgr':
        featurizer = featurizer_base(mode_=RxnMode.REAC_PROD)
    elif cfg.model.featurizer in ['rxnfp', 'uni_rxn', 'react_seq']:
        featurizer = featurizer_base(embed_loc= Path(cfg.filepaths.pretrained_rxn_embeddings) / f"{cfg.model.featurizer}.npy", length=cfg.model.vec_len)
    else:
        featurizer = featurizer_base(
            atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
            bond_featurizer=MultiHotBondFeaturizer()
        )

    return featurizer, datapoint_from_smi, dataset_base, generate_dataloader

def featurize_data(cfg: DictConfig, rng: np.random.Generator, train_data: pd.DataFrame = None, val_data: pd.DataFrame = None, shuffle_val: bool = True):
    '''
    
    Args
    -----
    shuffle_val:bool
        Set to True when training to avoid batch effects in validation
        Set to False when predicting / testing to avoid shuffling
    '''
    featurizer, datapoint_from_smi, dataset_base, generate_dataloader = construct_featurizer(cfg)

    if cfg.model.featurizer == 'cgr':
        rxn_k = 'am_smarts'
    elif cfg.model.featurizer in ['rxnfp', 'uni_rxn', 'react_seq']:
        rxn_k = 'reaction_idx'
    else:
        rxn_k = 'smarts'
    
    if train_data is not None:
        train_datapoints = []
        for _, row in train_data.iterrows():
            y = np.array([row['y']]).astype(np.float32)
            train_datapoints.append(datapoint_from_smi(row[rxn_k], reaction_center=row['reaction_center'], y=y, x_d=row['protein_embedding']))
        
        train_dataset = dataset_base(train_datapoints, featurizer=featurizer)
        train_dataloader = generate_dataloader(train_dataset, shuffle=True, seed=cfg.data.seed)
    else:
        train_dataloader = None

    if val_data is not None:
        val_datapoints = []
        for _, row in val_data.iterrows():
            y = np.array([row['y']]).astype(np.float32)
            val_datapoints.append(datapoint_from_smi(row[rxn_k], reaction_center=row['reaction_center'], y=y, x_d=row['protein_embedding']))

        if shuffle_val:
            rng.shuffle(val_datapoints) # Avoid weirdness of calculating metrics with only one class in the batch
        
        val_dataset = dataset_base(val_datapoints, featurizer=featurizer)
        val_dataloader = generate_dataloader(val_dataset, shuffle=False, batch_size=500)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader, featurizer

def construct_model(cfg: DictConfig, embed_dim: int, featurizer, device, ckpt=None):
    pos_weight = torch.ones([1]) * cfg.data.neg_multiple * cfg.training.pos_multiplier
    pos_weight = pos_weight.to(device)
    agg = getattr(src.nn, cfg.model.agg)() if cfg.model.agg else None
    pred_head = getattr(src.nn, cfg.model.pred_head)(
        input_dim=cfg.model.d_h_encoder * 2,
        criterion = src.nn.WeightedBCELoss(pos_weight=pos_weight)
    )
    metrics = [getattr(src.metrics, m)() for m in cfg.training.metrics]

    if cfg.model.message_passing:
        dv, de = featurizer.shape
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
        
    # Load from ckpt
    if ckpt:
        ckpt = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        
    return model

def downsample_negatives(data: pd.DataFrame, neg_multiple: int, rng: np.random.Generator):
    neg_idxs = data[data['y'] == 0].index
    n_to_rm = len(neg_idxs) - (len(data[data['y'] == 1]) * neg_multiple)
 
    if n_to_rm <= 0:
        return
    
    idx_to_rm = rng.choice(neg_idxs, n_to_rm, replace=False)
    data.drop(axis=0, index=idx_to_rm, inplace=True)


def mlflow_to_omegaconf(run_data: 'RunData'):
    tmp = {}
    for k, v in run_data.data.params.items():
        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                pass

        if v == 'None': 
            v = None
        
        if '/' in k:
            uk, lk = k.split('/')
            
            if lk == 'metrics':
                v = v.strip("['']").split("', '")
            
            if uk in tmp:
                tmp[uk][lk] = v
            else:
                tmp[uk] = {}
                tmp[uk][lk] = v
        else:
            tmp[k] = v

    cfg = OmegaConf.create(tmp)
    artifacts_path = Path(run_data.info.artifact_uri.removeprefix('file://'))

    return cfg, artifacts_path
     