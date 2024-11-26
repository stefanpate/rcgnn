import hydra
from chemprop.data import build_dataloader
import chemprop.nn
import torch
import numpy as np

from src.utils import load_json
import src.nn
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

from src.filepaths import filepaths
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
    datapoints_train = []
    for row in train_data:
        rxn = reactions[row['feature']]
        y = np.array([row['y']]).astype(np.float32)
        datapoints_train.append(datapoint_from_smi(rxn, y=y, x_d=row['sample_embed']))

    datapoints_test = []
    for row in test_data:
        rxn = reactions[row['feature']]
        y = np.array([row['y']]).astype(np.float32)
        datapoints_test.append(datapoint_from_smi(rxn, y=y, x_d=row['sample_embed']))

    dataset_train = dataset_base(datapoints_train, featurizer=featurizer)
    dataset_test = dataset_base(datapoints_test, featurizer=featurizer)

    data_loader_train = generate_dataloader(dataset_train, shuffle=False)
    data_loader_test = generate_dataloader(dataset_test, shuffle=False)

    return data_loader_train, data_loader_test, featurizer

@hydra.main(version_base=None, config_path=str(filepaths['configs']), config_name="train_base")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.ones([1]) * cfg.data.neg_multiple * 2 # TODO bubble up thru cfgs
    split_idx = 0 # TODO bubble up to cfgs / cl
    embed_dim = 1280 # TODO

    # Load data
    reactions = load_json(filepaths['data'] / f"{cfg.data.dataset}/{cfg.data.toc}.json") # TODO eliminate by saving smiles and rcs to npy files

    train_data, test_data = load_data_split(
        split_idx=split_idx,
        scratch_path=filepaths['scratch'] / cfg.data.subdir_patt
    )

    data_loader_train, data_loader_test, featurizer = featurize_data(
        train_data=train_data,
        test_data=test_data,
        reactions=reactions,
        cfg=cfg
    )

    # Construct model
    pos_weight = pos_weight.to(device)
    agg = getattr(src.nn, cfg.model.agg)()
    pred_head = getattr(src.nn, cfg.model.pred_head)(
        input_dim=cfg.model.d_h_encoder * 2,
        criterion = src.nn.WeightedBCELoss(pos_weight=pos_weight)
    )
    dv, de = featurizer.shape

    if cfg.model.message_passing:
        mp = getattr(src.nn, cfg.model.message_passing)(
            d_v=dv,
            d_e=de,
            d_h=cfg.model.d_h_encoder,
            depth=cfg.model.encoder_depth
        )

    # TODO streamline model api, get rid of LinDimRed
    if cfg.model.model == 'mpnn_dim_red':
        model = MPNNDimRed(
            reduce_X_d=src.nn.LinDimRed(d_in=embed_dim, d_out=cfg.model.d_h_encoder),
            message_passing=mp,
            agg=agg,
            predictor=pred_head,
        )
    elif cfg.model.model == 'ffn':
        model = TwoChannelFFN(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=cfg.model.d_h_encoder,
            encoder_depth=cfg.model.encoder_depth,
            predictor=pred_head,
        )
    elif cfg.model.model == 'linear':
        model = TwoChannelLinear(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=cfg.model.d_h_encoder,
            predictor=pred_head,
    )
    
    
    print()

if __name__ == '__main__':
    main()