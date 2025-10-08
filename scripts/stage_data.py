import hydra
from omegaconf import DictConfig
from src.utils import construct_sparse_adj_mat, load_json, load_embed, augment_idx_feature
from src.cross_validation import stratified_sim_split, random_split, sample_negatives

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
        X, y = sample_negatives(X_pos, y, 1, rng) # Sample to balance labels pre-split
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

    print("Splitting data...")
    if cfg.data.split_strategy == 'random':
        train_val_splits, test_split = random_split(X, y, cfg.data.n_splits, cfg.data.test_percent, seed=cfg.data.seed)
    elif cfg.data.split_strategy == 'random_reaction':
        train_val_splits, test_split = random_split(X, y, cfg.data.n_splits, cfg.data.test_percent, seed=cfg.data.seed, group_by=[int(ridx) for _, ridx in X])
    elif cfg.data.split_strategy == 'random_reaction_center':
        rule2idx = {}
        rule_groups = []
        for _, ridx in X:
            rid = idx_feature[ridx]
            rule = tuple(sorted(reactions[rid]['min_rules']))
            
            if rule not in rule2idx:
                rule2idx[rule] = len(rule2idx)
            
            rule_groups.append(int(rule2idx[rule]))
        train_val_splits, test_split = random_split(X, y, cfg.data.n_splits, cfg.data.test_percent, seed=cfg.data.seed, group_by=rule_groups)

    else:
        train_val_splits, test_split = stratified_sim_split(
            X=X,
            y=y,
            split_strategy=cfg.data.split_strategy,
            split_bounds=cfg.data.split_bounds,
            n_inner_splits=cfg.data.n_splits,
            test_percent=cfg.data.test_percent,
            cluster_dir=Path(cfg.filepaths.clustering),
            adj_mat_idx_to_id=idx_feature if cfg.data.split_strategy in ['rcmcs', 'drfp'] else idx_sample, # TODO: add other split strategies
            dataset=cfg.data.dataset,
            toc=cfg.data.toc,
            rng=rng
        )
    
    # Oversample negatives on train_val side (10 x pos)
    if cfg.data.negative_sampling == 'random':
        tmp = []
        for (Xi, yi) in train_val_splits:
            Xi, yi = sample_negatives(Xi, yi, 10, rng)
            tmp.append((Xi, yi))

        train_val_splits = tmp

        test_split = sample_negatives(test_split[0], test_split[1], 10, rng)

    print("Assembling data...")
    train_val_data, test_data = assemble_data(
        train_val_splits=train_val_splits,
        test_split=test_split,
        proteins=proteins,
        reactions=reactions,
        idx_sample=idx_sample,
        idx_feature=idx_feature
    )

    # Save
    print("Saving data...")
    for i, split in enumerate(train_val_data):
        print(f"Split {i} size: {len(split)}")
        print(f"Pos frac for split {i}: {split['y'].mean():.3f}")
        split.to_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet",
            index=False
        )

    print(f"Test size: {len(test_data)}")
    print(f"Pos frac for test: {test_data['y'].mean():.3f}")
    test_data.to_parquet(
        Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet",
        index=False
    )

if __name__ == '__main__':
    main()