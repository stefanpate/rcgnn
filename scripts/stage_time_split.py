'''
Do only random splitting here, load arc negs from disk
Do not split anything, just save to test.parquet in correct subdir per earlyier pattern
Save to scratch and projects
Make sure you filter out any seen rxn ids after applying rules

'''
import hydra
from omegaconf import DictConfig
from src.utils import construct_sparse_adj_mat, load_json, load_embed, augment_idx_feature
from src.cross_validation import sample_negatives

import numpy as np
import pandas as pd
from pathlib import Path

def assemble_data(
        train_val_splits: list[tuple[tuple[int]]],
        test_split: tuple[tuple[int]],
        proteins: dict,
        reactions: dict,
        idx_sample: dict,
        idx_feature: dict
    ):
    '''
    Pull together the actual data from the split indices
    '''
    splits = train_val_splits + [test_split]
    datas = []
    cols = ['protein_idx', 'reaction_idx', 'pid', 'rid', 'protein_embedding', 'smarts', 'am_smarts', 'reaction_center', 'y']
    for X, y in splits:
        data = []
        for (pidx, ridx), yi in zip(X, y):
            data.append(
                (
                    pidx,
                    ridx,
                    idx_sample[pidx],
                    idx_feature[ridx],
                    list(proteins[idx_sample[pidx]]),
                    reactions[idx_feature[ridx]]['smarts'],
                    reactions[idx_feature[ridx]]['am_smarts'],
                    reactions[idx_feature[ridx]]['rcs'],
                    yi
                )
            )
        
        datas.append(
            pd.DataFrame(data, columns=cols)
        )

    train_val_data = datas[:-1]
    test_data = datas[-1]

    return train_val_data, test_data

@hydra.main(version_base=None, config_path="../configs", config_name="stage_data")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)

    print("Loading data...")


    # Load reaction data
    _rxns = load_json(Path(cfg.filepaths.data) / cfg.data.dataset / f"{cfg.data.toc}.json")
    adj, idx_sample, idx_feature = construct_sparse_adj_mat(
        Path(cfg.filepaths.data) / cfg.data.dataset / (cfg.data.toc + ".csv")
    )
    X_pos = list(zip(*adj.nonzero()))

    # Do arc neg sampling
