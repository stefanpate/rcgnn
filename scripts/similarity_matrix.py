'''
Calculate similarity matrix
'''

from src.utils import load_embed_matrix, construct_sparse_adj_mat
from src.similarity import embedding_similarity
from src.config import filepaths
from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter

embeddings_superdir = filepaths['artifacts_embeddings']
sim_mats_dir = filepaths['artifacts_sim_mats']

def reaction_embedding_similarity(args, embeddings_superdir: Path = embeddings_superdir, sim_mats_dir: Path = sim_mats_dir):
    embed_path = embeddings_superdir / args.embed_path
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_{'_'.join(args.embed_path.split('/'))}"
    _, _, idx_feature = construct_sparse_adj_mat(args.dataset, args.toc)
    X = load_embed_matrix(embed_path, idx_feature)
    tic = perf_counter()
    embedding_similarity(X, save_to)
    toc = perf_counter()
    print(f"Matrix multiplication took: {toc - tic} seconds")

parser = ArgumentParser(description="Simlarity matrix calculator")
subparsers = parser.add_subparsers(title="Commands", description="Available comands")

# Reaction embedding similarity
parser_rxn_embed = subparsers.add_parser("rxn-embed", help="Calculate reaction embedding similarity")
parser_rxn_embed.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_rxn_embed.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_rxn_embed.add_argument("embed_path", help="Embedding path relative to embeddings super dir")
parser_rxn_embed.set_defaults(func=reaction_embedding_similarity)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()