from pathlib import Path
from src.nn import LastAggregation, DotSig, LinDimRed, AttentionAggregation, BondMessagePassingDict
from chemprop.nn import MeanAggregation, BinaryClassificationFFN, BondMessagePassing
import torch
from chemprop.models import MPNN
from chemprop.data import build_dataloader
from chemprop.nn import MeanAggregation, BinaryClassificationFFN, BondMessagePassing
from src.model import MPNNDimRed, TwoChannelFFN, TwoChannelLinear
from src.featurizer import SimpleReactionMolGraphFeaturizer, RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer, ReactionMorganFeaturizer
from src.data import RxnRCDataset, MFPDataset, mfp_build_dataloader


def construct_model(hps: dict, featurizer, embed_dim: int, chkpt: Path = None):
    '''
    Args
    ------
    hps:dict
        Hyperparameter dict
    Returns
    -------
    model: LightningModule
    '''
    # Aggregation fcns
    aggs = {
        'last':LastAggregation,
        'mean':MeanAggregation,
        'attention':AttentionAggregation
    }

    # Prediction heads
    pred_heads = {
        'binary':BinaryClassificationFFN,
        'dot_sig':DotSig
    }

    # Message passing
    message_passers = {
        'bondwise':BondMessagePassing,
        'bondwise_dict':BondMessagePassingDict
    }

    d_h_encoder = hps['d_h_encoder'] # Hidden layer of message passing
    encoder_depth = hps['encoder_depth']

    if hps['message_passing']:
        mp = message_passers[hps['message_passing']](d_v=featurizer.shape[0], d_e=featurizer.shape[1], d_h=d_h_encoder, depth=encoder_depth)

    if hps['agg']:
        agg = aggs[hps['agg']](input_dim=d_h_encoder) if hps['agg'] == 'attention' else aggs[hps['agg']]()

    pred_head = pred_heads[hps['pred_head']](input_dim=d_h_encoder * 2)

    if hps['model'] == 'mpnn':
        model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=pred_head,
        )
    elif hps['model'] == 'mpnn_dim_red':
        model = MPNNDimRed(
            reduce_X_d=LinDimRed(d_in=embed_dim, d_out=d_h_encoder),
            message_passing=mp,
            agg=agg,
            predictor=pred_head,
        )
    elif hps['model'] == 'ffn':
        model = TwoChannelFFN(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=d_h_encoder,
            encoder_depth=encoder_depth,
            predictor=pred_head,
        )
    elif hps['model'] == 'linear':
        model = TwoChannelLinear(
            d_rxn=featurizer.length,
            d_prot=embed_dim,
            d_h=d_h_encoder,
            predictor=pred_head,
        )
    
    if chkpt:
        chkpt = torch.load(chkpt, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(chkpt['state_dict'])

    return model

def construct_featurizer(hps: dict):
    mfp_length = 2**10
    mfp_radius = 2

    # Featurizers +
    featurizers = {
        'rxn_simple': (RxnRCDataset, SimpleReactionMolGraphFeaturizer, build_dataloader),
        'rxn_rc': (RxnRCDataset, RCVNReactionMolGraphFeaturizer, build_dataloader),
        'mfp': (MFPDataset, ReactionMorganFeaturizer, mfp_build_dataloader)
    }

    dataset_base, featurizer_base, generate_dataloader = featurizers[hps['featurizer']]
    if hps['featurizer'] == 'mfp':
        featurizer = featurizer_base(radius=mfp_radius, length=mfp_length)
    else:
        featurizer = featurizer_base(
            atom_featurizer=MultiHotAtomFeaturizer.no_stereo(),
            bond_featurizer=MultiHotBondFeaturizer()
        )

    return dataset_base, generate_dataloader, featurizer