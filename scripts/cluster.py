from argparse import ArgumentParser
from sklearn.cluster import AgglomerativeClustering
from src.similarity import rcmcs_similarity_matrix, rcmcs_similarity_matrix_mp
from src.utils import load_json, construct_sparse_adj_mat, save_json
import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("similarity_score", help="Valid scores: {'rcmcs', }")
    parser.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
    parser.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
    parser.add_argument("-c", "--cutoff", nargs='+', help="Upper limit on cross-cluster similarity", type=float, required=True)
    parser.add_argument("-m", "--multi-process", action='store_true')

    args = parser.parse_args()

    needs_rules = ['rcmcs']
    needs_reactions = ['rcmcs']
    needs_toc = []

    if args.similarity_score in needs_rules:
        rules = pd.read_csv(
            filepath_or_buffer='../data/sprhea/minimal1224_all_uniprot.tsv',
            sep='\t'
        )
        rules.set_index('Name', inplace=True)

    if args.similarity_score in needs_reactions:
        rxns = load_json(f"../data/{args.dataset}/{args.toc}.json")
        rxns = {int(k) : v for k, v in rxns.items()}

    if args.similarity_score in needs_toc:
        toc = construct_sparse_adj_mat(args.dataset, args.toc)

    # Calculate similarity matrix
    if args.similarity_score == 'rcmcs' and args.multi_process:
        S, sim_i_to_rxn_idx = rcmcs_similarity_matrix_mp(rxns, rules, norm='max')
    elif args.similarity_score == 'rcmcs':
        S, sim_i_to_rxn_idx = rcmcs_similarity_matrix(rxns, rules, norm='max')

    D = 1 - S # Distance matrix
    
    for cutoff in args.cutoff:
        d_cutoff = 1 - cutoff

        # Cluster
        ac = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            distance_threshold=d_cutoff,
            linkage='single'
        )

        ac.fit(D)

        if len(sim_i_to_rxn_idx) != len(ac.labels_):
            raise Exception("Cluster label array length not equal to similarity matrix dimension")

        # Save clusters
        rxnidx2cluster = {sim_i_to_rxn_idx[i] : int(ac.labels_[i]) for i in sim_i_to_rxn_idx}
        save_json(rxnidx2cluster, f"../artifacts/clustering/{args.dataset}_{args.toc}_{args.similarity_score}_{int(cutoff * 100)}.json")