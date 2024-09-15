'''
Calculate similarity matrices with traditional similarity measures
like RCMCS, sequence homology
'''


from argparse import ArgumentParser
from src.similarity import rcmcs_similarity_matrix
from src.utils import load_json, save_json
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("similarity_score", help="Valid scores: {'rcmcs'}")
    parser.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
    parser.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")

    args = parser.parse_args()

    rules = pd.read_csv(
        filepath_or_buffer='../data/sprhea/minimal1224_all_uniprot.tsv',
        sep='\t'
    )
    rules.set_index('Name', inplace=True)

    rxns = load_json(f"../data/{args.dataset}/{args.toc}.json")
    rxns = {int(k) : v for k, v in rxns.items()}


    # Calculate similarity matrix
    S, sim_i_to_id = rcmcs_similarity_matrix(rxns, rules, norm='max')
    S = S.astype(np.float32)
        
    # Save clusters
    np.save(f"../artifacts/sim_mats/{args.dataset}_{args.toc}_{args.similarity_score}_similarity_matrix", S)
    save_json(sim_i_to_id, f"../artifacts/sim_mats/{args.dataset}_{args.toc}_{args.similarity_score}_sim_i_to_id.json")