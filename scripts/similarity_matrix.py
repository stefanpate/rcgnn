from src.utils import load_embed_matrix, construct_sparse_adj_mat, load_json
from src.similarity import embedding_similarity_matrix, rcmcs_similarity_matrix
from src.config import filepaths
from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter
import pandas as pd
import numpy as np

embeddings_superdir = filepaths['artifacts_embeddings']
sim_mats_dir = filepaths['artifacts_sim_mats']
data_fp = filepaths['data']

def save_sim_mat(S: np.ndarray, save_to: Path):
    parent = save_to.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    np.save(save_to, S)

def calc_rxn_embed_sim(args, embeddings_superdir: Path = embeddings_superdir, sim_mats_dir: Path = sim_mats_dir):
    embed_path = embeddings_superdir / args.embed_path
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_{'_'.join(args.embed_path.split('/'))}"
    _, _, idx_feature = construct_sparse_adj_mat(args.dataset, args.toc)
    X = load_embed_matrix(embed_path, idx_feature)
    tic = perf_counter()
    S = embedding_similarity_matrix(X)
    toc = perf_counter()
    print(f"Matrix multiplication took: {toc - tic} seconds")
    save_sim_mat(S, save_to)  

def calc_rcmcs_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_rcmcs"
    rules = pd.read_csv(
        filepath_or_buffer=data_filepath / "minimal1224_all_uniprot.tsv",
        sep='\t'
    )
    rules.set_index('Name', inplace=True)

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(args.dataset, args.toc)

    S = rcmcs_similarity_matrix(rxns, rules, idx_feature)
    save_sim_mat(S, save_to)

parser = ArgumentParser(description="Simlarity matrix calculator")
subparsers = parser.add_subparsers(title="Commands", description="Available comands")

# Reaction embedding similarity
parser_rxn_embed = subparsers.add_parser("rxn-embed", help="Calculate reaction embedding similarity")
parser_rxn_embed.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_rxn_embed.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_rxn_embed.add_argument("embed_path", help="Embedding path relative to embeddings super dir")
parser_rxn_embed.set_defaults(func=calc_rxn_embed_sim)

# RCMCS similarity
parser_rcmcs = subparsers.add_parser("rcmcs", help="Calculate RCMCS similarity")
parser_rcmcs.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_rcmcs.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_rcmcs.set_defaults(func=calc_rcmcs_sim)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()