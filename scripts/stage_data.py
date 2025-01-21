import hydra
from omegaconf import DictConfig
from src.utils import construct_sparse_adj_mat, load_json, load_embed
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
    
    splits = train_val_splits + [test_split]
    datas = []
    cols = ['protein_idx', 'reaction_idx', 'pid', 'rid', 'protein_embedding', 'smarts', 'reaction_center', 'y']
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

    # Load adjacency matrix
    adj, idx_sample, idx_feature = construct_sparse_adj_mat(
        Path(cfg.filepaths.data) / cfg.data.dataset / (cfg.data.toc + ".csv")
    )

    # Load protein embeddings
    proteins = {}
    for pid in idx_sample.values():
        proteins[pid] = np.array(
            load_embed(
                Path(cfg.filepaths.data) / cfg.data.dataset / cfg.data.embed_type / f"{pid}.pt",
                embed_key=33
            )[1]
        )

    # Load reactions
    reactions = load_json(Path(cfg.filepaths.data) / cfg.data.dataset / (cfg.data.toc + ".json"))

    X_pos = list(zip(*adj.nonzero()))
    y = [1 for _ in range(len(X_pos))]

    X, y = sample_negatives(X_pos, y, 1, rng) # Sample to balance labels pre-split

    if cfg.data.split_strategy == 'random':
        train_val_splits, test_split = random_split(X, y, cfg.data.n_splits, cfg.data.test_percent)
    else:
        train_val_splits, test_split = stratified_sim_split(
            X=X,
            y=y,
            split_strategy=cfg.data.split_strategy,
            split_bounds=cfg.data.split_bounds,
            n_inner_splits=cfg.data.n_splits,
            test_percent=cfg.data.test_percent,
            cluster_dir=Path(cfg.filepaths.clustering),
            adj_mat_idx_to_id=idx_feature if cfg.data.split_strategy == 'rcmcs' else idx_sample,
            dataset=cfg.data.dataset,
            toc=cfg.data.toc,
            rng=rng
        )
    
    # Oversample negatives on train_val side (10 x pos)
    tmp = []
    for (Xi, yi) in train_val_splits:
        Xi, yi = sample_negatives(Xi, yi, 10, rng)
        tmp.append((Xi, yi))

    train_val_splits = tmp

    train_val_data, test_data = assemble_data(
        train_val_splits=train_val_splits,
        test_split=test_split,
        proteins=proteins,
        reactions=reactions,
        idx_sample=idx_sample,
        idx_feature=idx_feature
    )

    # Save
    for i, split in enumerate(train_val_data):
        split.to_parquet(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet",
            index=False
        )

    test_data.to_parquet(
        Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet",
        index=False
    )

if __name__ == '__main__':
    main()