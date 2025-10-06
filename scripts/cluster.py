from sklearn.cluster import AgglomerativeClustering
from src.similarity import load_similarity_matrix
from src.utils import save_json, construct_sparse_adj_mat
import numpy as np
from omegaconf import DictConfig
import hydra
from pathlib import Path

@hydra.main(version_base=None, config_path="../configs", config_name="cluster")
def main(cfg: DictConfig):
    adj, adj_to_prot_id, adj_to_rxn_id = construct_sparse_adj_mat(
        Path(cfg.filepaths.data) / cfg.dataset / (cfg.toc + ".csv")
        )
    
    if cfg.similarity_score in ['rcmcs', 'drfp']:
        matrix_idx_to_id = adj_to_rxn_id
    elif cfg.similarity_score in ['gsi', 'esm', 'blosum']: # Protein based similarity
        matrix_idx_to_id = adj_to_prot_id
    else:
        raise ValueError(f"Unknown similarity score: {cfg.similarity_score}")
    
    # Load similarity matrix
    S = load_similarity_matrix(
        sim_path=Path(cfg.filepaths.results) / "similarity_matrices",
        dataset=cfg.dataset,
        toc=cfg.toc,
        sim_metric=cfg.similarity_score
    )
    S = (S - S.min()) / (S.max() - S.min())    
    D = 1 - S # Distance matrix

    if cfg.similarity_score == 'blosum':
        # Normalize blosum alignment scores
        S = np.where(S > cfg.blosum_ub, cfg.blosum_ub, S)
        S = np.where(S < cfg.blosum_lb, cfg.blosum_lb, S)
    
    for cutoff in cfg.cutoffs:
        d_cutoff = 1 - (cutoff / 100)

        # Cluster
        ac = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            distance_threshold=d_cutoff,
            linkage='single'
        )
        ac.fit(D)            
        labels = ac.labels_
        id2cluster = {matrix_idx_to_id[i] : int(labels[i]) for i in matrix_idx_to_id}
        
        # Save clusters
        save_json(id2cluster, Path(cfg.filepaths.clustering) / f"{cfg.dataset}_{cfg.toc}_{cfg.similarity_score}_{cutoff}.json")

if __name__ == '__main__':
    main()