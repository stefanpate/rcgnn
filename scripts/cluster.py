from argparse import ArgumentParser
from sklearn.cluster import AgglomerativeClustering
from src.similarity import rcmcs_similarity_matrix, homology_similarity_matrix
from src.utils import load_json, construct_sparse_adj_mat, save_json
from src.cross_validation import sample_negatives
import pandas as pd
from Bio import Align

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("similarity_score", help="Valid scores: {'rcmcs', 'homology'}")
    parser.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
    parser.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
    parser.add_argument("-c", "--cutoff", nargs='+', help="Upper limit on cross-cluster similarity", type=float, required=True)

    args = parser.parse_args()

    needs_rules = ['rcmcs', 'combo']
    needs_reactions = ['rcmcs', 'combo']
    needs_seq_aligment = ['homology']

    if args.similarity_score == 'combo':
        neg_multiple = 1
        seed = 1234
        adj, adj_i_prot_id, adj_i_rxn_id = construct_sparse_adj_mat(args.dataset, args.toc)
        X = list(zip(*adj.nonzero())) # Positive protein-reaction pairs
        X, _ = sample_negatives(X, neg_multiple, seed) # Sample negatives
        X = [(adj_i_prot_id[i], int(adj_i_rxn_id[j])) for i, j in X] # Translate back to upid, rxnid

    if args.similarity_score in needs_rules:
        rules = pd.read_csv(
            filepath_or_buffer='../data/sprhea/minimal1224_all_uniprot.tsv',
            sep='\t'
        )
        rules.set_index('Name', inplace=True)

    if args.similarity_score in needs_reactions:
        rxns = load_json(f"../data/{args.dataset}/{args.toc}.json")
        rxns = {int(k) : v for k, v in rxns.items()}

    if args.similarity_score in needs_seq_aligment:
        toc = pd.read_csv(
            filepath_or_buffer=f"../data/{args.dataset}/{args.toc}.csv",
            sep='\t'
        ).set_index("Entry")
        aligner = Align.PairwiseAligner(
            mode="local",
            scoring="blastp"
        )
        sequences = {id: row["Sequence"] for id, row in toc.iterrows()}

    # Calculate similarity matrix
    if args.similarity_score == 'rcmcs':
        S, sim_i_to_id = rcmcs_similarity_matrix(rxns, rules, norm='max')
    elif args.similarity_score == 'homology':
        S, sim_i_to_id = homology_similarity_matrix(sequences, aligner)

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
        labels = ac.labels_
        id2cluster = {sim_i_to_id[i] : int(labels[i]) for i in sim_i_to_id}
        
        # Save clusters
        save_json(id2cluster, f"../artifacts/clustering/{args.dataset}_{args.toc}_{args.similarity_score}_{int(cutoff * 100)}.json")