'''
Do only random sampling here, load arc negs from disk
Do not split anything, just save to test.parquet in correct subdir per earlyier pattern
Save to scratch and projects
'''
import hydra
from omegaconf import DictConfig
from src.utils import construct_sparse_adj_mat, load_json, load_embed, augment_idx_feature
from src.cross_validation import sample_negatives

import numpy as np
import pandas as pd
from pathlib import Path

def assemble_data(
        X: tuple[int],
        y: tuple[int],
        proteins: dict,
        reactions: dict,
        idx_sample: dict,
        idx_feature: dict
    ):
    '''
    Pull together the actual data from the split indices
    '''
    cols = ['protein_idx', 'reaction_idx', 'pid', 'rid', 'protein_embedding', 'smarts', 'am_smarts', 'reaction_center', 'y']
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
    return pd.DataFrame(data, columns=cols)

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

    if cfg.data.negative_sampling == 'alternate_reaction_center':
        unobs_rxns = load_json(Path(cfg.filepaths.data) / cfg.data.dataset / f"{cfg.data.toc}_arc_unobserved_reactions.json")
        reactions = {**_rxns, **unobs_rxns}
        idx_feature = augment_idx_feature(idx_feature, unobs_rxns)
        feature_idx = {v: k for k, v in idx_feature.items()}
        sample_idx = {v: k for k, v in idx_sample.items()}
        neg_data = pd.read_csv(
            Path(cfg.filepaths.data) / cfg.data.dataset / f"{cfg.data.toc}_arc_negative_samples.csv",
            sep='\t',
        )
        X_neg = []
        for _, row in neg_data.iterrows():
            for rid in row['Label'].split(';'):
                X_neg.append((sample_idx[row['Entry']], feature_idx[rid]))
        X = X_pos + X_neg
        y = [1 for _ in range(len(X_pos))] + [0 for _ in range(len(X_neg))]
    elif cfg.data.negative_sampling == 'random':
        reactions = _rxns
        y = [1 for _ in range(len(X_pos))]
        X, y = sample_negatives(X_pos, y, 10, rng)
    else:
        raise ValueError(f"Invalid negative sampling strategy: {cfg.data.negative_sampling}")

    # Load protein embeddings
    proteins = {}
    for pid in idx_sample.values():
        proteins[pid] = np.array(
            load_embed(
                Path(cfg.filepaths.data) / cfg.data.dataset / cfg.data.embed_type / f"{pid}.pt",
                embed_key=33
            )[1]
        )

    # Assemble data
    print("Assembling data...")
    data = assemble_data(X, y, proteins, reactions, idx_sample, idx_feature)
    print(f"Test size: {len(data)}")
    print(f"Pos frac for test: {data['y'].mean():.3f}")
    data.to_parquet(
        Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet",
        index=False
    )

if __name__ == "__main__":
    main()
