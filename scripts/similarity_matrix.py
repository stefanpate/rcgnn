from src.utils import load_embed_matrix, construct_sparse_adj_mat, load_json
from src.similarity import embedding_similarity_matrix, rcmcs_similarity_matrix, mcs_similarity_matrix, tanimoto_similarity_matrix, agg_mfp_cosine_similarity_matrix
from src.config import filepaths
from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter
import pandas as pd
import numpy as np

embeddings_superdir = filepaths['embeddings']
sim_mats_dir = filepaths['sim_mats']
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
    X = load_embed_matrix(embed_path, idx_feature, args.dataset, args.toc)
    tic = perf_counter()
    S = embedding_similarity_matrix(X)
    toc = perf_counter()
    print(f"Matrix multiplication took: {toc - tic} seconds")
    save_sim_mat(S, save_to)

def calc_prot_embed_sim(args, embeddings_superdir: Path = embeddings_superdir, sim_mats_dir: Path = sim_mats_dir):
    embed_path = embeddings_superdir / args.embed_path
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_{'_'.join(args.embed_path.split('/'))}"
    _, idx_sample, _ = construct_sparse_adj_mat(args.dataset, args.toc)
    X = load_embed_matrix(embed_path, idx_sample, args.dataset, args.toc)
    tic = perf_counter()
    S = embedding_similarity_matrix(X)
    toc = perf_counter()
    print(f"Matrix multiplication took: {toc - tic} seconds")
    save_sim_mat(S, save_to)

def calc_prot_by_rxn_sim(args, embeddings_superdir: Path = embeddings_superdir, sim_mats_dir: Path = sim_mats_dir):
    prot_embed_path = embeddings_superdir / args.prot_embed_path
    rxn_embed_path = embeddings_superdir / args.rxn_embed_path
    prot_parent = prot_embed_path.parent.name
    rxn_parent = rxn_embed_path.parent.name

    if prot_parent != rxn_parent:
        raise ValueError("Embedding type for proteins and reactions must be the same")

    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_{prot_parent}_proteins_x_reactions"
    _, idx_sample, idx_feature = construct_sparse_adj_mat(args.dataset, args.toc)
    X = load_embed_matrix(prot_embed_path, idx_sample, args.dataset, args.toc)
    X2 = load_embed_matrix(rxn_embed_path, idx_feature, args.dataset, args.toc)
    tic = perf_counter()
    S = embedding_similarity_matrix(X, X2=X2)
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

def calc_mcs_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_mcs"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(args.dataset, args.toc)

    S = mcs_similarity_matrix(rxns, idx_feature)
    save_sim_mat(S, save_to)

def calc_tani_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_tanimoto"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(args.dataset, args.toc)

    S = tanimoto_similarity_matrix(rxns, idx_feature)
    save_sim_mat(S, save_to)

def calc_agg_mfp_cosine_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_agg_mfp_cosine"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(args.dataset, args.toc)

    S = agg_mfp_cosine_similarity_matrix(rxns, idx_feature)
    save_sim_mat(S, save_to)

parser = ArgumentParser(description="Simlarity matrix calculator")
subparsers = parser.add_subparsers(title="Commands", description="Available comands")

# Reaction embedding similarity
parser_rxn_embed = subparsers.add_parser("rxn-embed", help="Calculate reaction embedding similarity")
parser_rxn_embed.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_rxn_embed.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_rxn_embed.add_argument("embed_path", help="Embedding path relative to embeddings super dir")
parser_rxn_embed.set_defaults(func=calc_rxn_embed_sim)

# Protein embedding similarity
parser_prot_embed = subparsers.add_parser("prot-embed", help="Calculate protein embedding similarity")
parser_prot_embed.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_prot_embed.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_prot_embed.add_argument("embed_path", help="Embedding path relative to embeddings super dir")
parser_prot_embed.set_defaults(func=calc_prot_embed_sim)

# Protein by reaction embedding similarity
parser_prot_rxn_embed = subparsers.add_parser("prot-rxn", help="Calculate protein by reaction embedding similarity")
parser_prot_rxn_embed.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_prot_rxn_embed.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_prot_rxn_embed.add_argument("prot_embed_path", help="Protein embedding path relative to embeddings super dir")
parser_prot_rxn_embed.add_argument("rxn_embed_path", help="Reaction embedding path relative to embeddings super dir")
parser_prot_rxn_embed.set_defaults(func=calc_prot_by_rxn_sim)

# RCMCS similarity
parser_rcmcs = subparsers.add_parser("rcmcs", help="Calculate RCMCS similarity")
parser_rcmcs.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_rcmcs.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_rcmcs.set_defaults(func=calc_rcmcs_sim)

# MCS similarity
parser_mcs = subparsers.add_parser("mcs", help="Calculate MCS similarity")
parser_mcs.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_mcs.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_mcs.set_defaults(func=calc_mcs_sim)

# Tanimoto similarity
parser_tanimoto = subparsers.add_parser("tanimoto", help="Calculate substrate-aligned Tanimoto similarity")
parser_tanimoto.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_tanimoto.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_tanimoto.set_defaults(func=calc_tani_sim)

# Agg mfp cosine similarity
parser_agg_mfp_cosine = subparsers.add_parser("agg-mfp-cosine", help="Calculate cosine similarity of aggregated Morgan FPs")
parser_agg_mfp_cosine.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_agg_mfp_cosine.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_agg_mfp_cosine.set_defaults(func=calc_agg_mfp_cosine_sim)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()