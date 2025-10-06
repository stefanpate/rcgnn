from src.utils import load_embed, construct_sparse_adj_mat, load_json, load_embed_matrix
from src.similarity import(
    embedding_similarity_matrix,
    rcmcs_similarity_matrix,
    mcs_similarity_matrix,
    tanimoto_similarity_matrix,
    morgan_fingerprint,
    homology_similarity_matrix,
    blosum_similarity_matrix
)
from omegaconf import OmegaConf
from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter
import pandas as pd
import numpy as np
import scipy.sparse as sp
from Bio import Align
from Bio.Align import substitution_matrices
from drfp import DrfpEncoder
from tqdm import tqdm
from rdkit import Chem

filepaths = OmegaConf.load("../configs/filepaths/base.yaml")
embeddings_superdir = Path(filepaths['results']) / "embeddings"
sim_mats_dir = Path(filepaths['results']) / "similarity_matrices"
data_fp = Path(filepaths['data'])

def save_sim_mat(S: np.ndarray, save_to: Path):
    parent = save_to.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    np.save(save_to, S)

def calc_rxn_embed_sim(args, embeddings_superdir: Path = embeddings_superdir, sim_mats_dir: Path = sim_mats_dir):
    embed_path = embeddings_superdir / args.embed_path
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_{'_'.join(args.embed_path.split('/'))}"
    _, _, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")
    X = load_embed_matrix(embed_path, idx_feature, args.dataset, args.toc)
    tic = perf_counter()
    S = embedding_similarity_matrix(X)
    toc = perf_counter()
    print(f"Matrix multiplication took: {toc - tic} seconds")
    save_sim_mat(S, save_to)

def calc_prot_embed_sim(args, embeddings_superdir: Path = embeddings_superdir, sim_mats_dir: Path = sim_mats_dir):
    embed_path = embeddings_superdir / args.embed_path
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_{'_'.join(args.embed_path.split('/'))}"
    _, idx_sample, _ = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")
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
    _, idx_sample, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")
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

    print("Loading reactions...")
    _rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    unobs_rxns = load_json(data_filepath / args.dataset / f"{args.toc}_arc_unobserved_reactions.json")
    rxns = {**_rxns, **unobs_rxns}

    _, _, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")
    idx_feature = augment_idx_feature(idx_feature, unobs_rxns)
    n_chunks = -(len(rxns) // - args.chunk_size)
    for i in range(n_chunks):
        print(f"Processing chunk {i+1} of {n_chunks}...")
        start = i * args.chunk_size
        end = (i + 1) * args.chunk_size
        S_chunk = rcmcs_similarity_matrix(rxns, rules, idx_feature, start, end)
        
        save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_rcmcs_chunk_{i}"
        parent = save_to.parent
        
        if not parent.exists():
            parent.mkdir(parents=True)

        sp.save_npz(save_to, S_chunk)

        del S_chunk

    # S = rcmcs_similarity_matrix(rxns, rules, idx_feature)
    # save_sim_mat(S, save_to)

def calc_mcs_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_mcs"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")

    S = mcs_similarity_matrix(rxns, idx_feature)
    save_sim_mat(S, save_to)

def calc_tani_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_tanimoto"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")

    S = tanimoto_similarity_matrix(rxns, idx_feature)
    save_sim_mat(S, save_to)

def calc_drfp_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_drfp"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")
    V = []
    for i, id in tqdm(idx_feature.items(), total=len(idx_feature), desc="Calculating DRFP embeddings"):
        rxn = rxns[id]['am_smarts']
        embed = DrfpEncoder.encode(rxn)[0]
        V.append(embed)
    

    V = np.array(V)
    D = (V @ V.T)
    N = np.square(V).sum(axis=1).reshape(-1, 1)
    S = D / (N + N.T - D)

    assert S.shape == (len(idx_feature), len(idx_feature))
    assert (S >= 0).all() and (S <= 1).all()

    save_sim_mat(S, save_to)
    print(f"Saved DRFP similarity matrix to {save_to}")

def calc_agg_mfp_cosine_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    '''
    Calculates consine similarity between abs(rct_mfp_sum - pdt_mfp_sum) reaction vectors
    '''
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_agg_mfp_cosine"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")

    rxn_vecs = []
    for entry in tqdm(rxns.values(), total=len(rxns), desc="Calculating aggregated Morgan fingerprints"):
        rxn = entry['smarts']
        mols = [[Chem.MolFromSmiles(smi) for smi in side.split(".")] for side in rxn.split(">>")]
        mfps = [[morgan_fingerprint(mol) for mol in side] for side in mols]
        rvec = abs(sum(mfps[0]) - sum(mfps[1]))
        
        n = np.linalg.norm(rvec)
        if n != 0:
            rvec /= np.linalg.norm(rvec)
        
        rxn_vecs.append(rvec)

    V = np.array(rxn_vecs)
    S = V @ V.T
    assert S.shape == (len(idx_feature), len(idx_feature))
    assert (S >= -0.001).all() and (S <= 1.001).all()

    save_sim_mat(S, save_to)
    print(f"Saved aggregated Morgan fingerprint cosine similarity matrix to {save_to}")

def calc_concat_mfp_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    '''
    Calculates Tanimoto similarity between concat(multiple rct mfp and multiple pdt mfp) reaction vectors
    '''
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_concat_mfp_tanimoto"

    rxns = load_json(data_filepath / args.dataset / f"{args.toc}.json")
    _, _, idx_feature = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")

    rct_vecs = []
    pdt_vecs = []
    for entry in tqdm(rxns.values(), total=len(rxns), desc="Calculating concatenated Morgan fingerprints"):
        rxn = entry['smarts']
        mols = [Chem.MolFromSmiles(side) for side in rxn.split(">>")]
        mfps = [morgan_fingerprint(side) for side in mols]
        rct_vecs.append(mfps[0])
        pdt_vecs.append(mfps[1])

    V = np.array([np.concatenate([r, p]) for r, p in zip(rct_vecs, pdt_vecs)])
    D = (V @ V.T)
    N = np.square(V).sum(axis=1).reshape(-1, 1)
    S = D / (N + N.T - D)

    assert S.shape == (len(idx_feature), len(idx_feature))
    assert (S >= -0.001).all() and (S <= 1.001).all()

    save_sim_mat(S, save_to)
    print(f"Saved concatenated Morgan fingerprint Tanimoto similarity matrix to {save_to}")

def calc_gsi(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):

    toc = pd.read_csv(
        filepath_or_buffer=data_filepath / args.dataset / f"{args.toc}.csv",
        sep='\t'
    ).set_index("Entry")

    aligner = Align.PairwiseAligner(
        mode="global",
        scoring="blastp"
    )
    aligner.open_gap_score = -1e6
    n_chunks = -(len(toc) // - args.chunk_size)
    sequences = {id: row["Sequence"] for id, row in toc.iterrows()}
    for i in range(n_chunks):
        print(f"Processing chunk {i+1} of {n_chunks}...")
        start = i * args.chunk_size
        end = (i + 1) * args.chunk_size
        S_chunk = homology_similarity_matrix(sequences, start, end, aligner)
        
        save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_gsi_chunk_{i}"
        parent = save_to.parent
        
        if not parent.exists():
            parent.mkdir(parents=True)

        sp.save_npz(save_to, S_chunk)

        del S_chunk

def calc_esm_sim(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):
    save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_esm"

    _, idx_sample, _ = construct_sparse_adj_mat(data_fp / args.dataset / f"{args.toc}.csv")

    proteins = []
    for i, id in tqdm(idx_sample.items(), total=len(idx_sample), desc="Loading ESM embeddings"):
        embed = load_embed(
            data_filepath / args.dataset / "esm" / f"{id}.pt",
            embed_key=33
        )[1]
        proteins.append(embed)

    V = np.array(proteins)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    assert S.shape == (len(idx_sample), len(idx_sample))
    assert (S >= -0.001).all() and (S <= 1.001).all()
    save_sim_mat(S, save_to)
    print(f"Saved ESM similarity matrix to {save_to}")

def calc_blosum62(args, data_filepath: Path = data_fp, sim_mats_dir: Path = sim_mats_dir):

    toc = pd.read_csv(
        filepath_or_buffer=data_filepath / args.dataset / f"{args.toc}.csv",
        sep='\t'
    ).set_index("Entry")

    # BLASTp defaults
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    n_chunks = -(len(toc) // - args.chunk_size)
    for i in range(n_chunks):
        sequences = {id: row["Sequence"] for id, row in toc.iterrows()}
        start = i * args.chunk_size
        end = (i + 1) * args.chunk_size
        S_chunk = blosum_similarity_matrix(sequences, start, end, aligner)
        
        save_to = sim_mats_dir / f"{args.dataset}_{args.toc}_blosum_chunk_{i}"
        parent = save_to.parent
        
        if not parent.exists():
            parent.mkdir(parents=True)

        sp.save_npz(save_to, S_chunk)

        del S_chunk

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
parser_rcmcs.add_argument("chunk_size", type=int, help="Breaks up rows of sim mat")
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

# DRFP similarity
parser_drfp = subparsers.add_parser("drfp", help="Calculate DRFP similarity")
parser_drfp.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_drfp.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_drfp.set_defaults(func=calc_drfp_sim)

# Global sequence identity
parser_gsi = subparsers.add_parser("gsi", help="Calculate global sequence identity")
parser_gsi.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_gsi.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_gsi.add_argument("chunk_size", type=int, help="Breaks up rows of sim mat")
parser_gsi.set_defaults(func=calc_gsi)

# ESM similarity
parser_esm = subparsers.add_parser("esm", help="Calculate ESM similarity")
parser_esm.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_esm.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_esm.set_defaults(func=calc_esm_sim)

# BLOSUM62 sequence similarity
parser_blosum = subparsers.add_parser("blosum", help="Calculate BLOSUM62 sequence similarity")
parser_blosum.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_blosum.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_blosum.add_argument("chunk_size", type=int, help="Breaks up rows of sim mat")
parser_blosum.set_defaults(func=calc_blosum62)

# Agg mfp cosine similarity
parser_agg_mfp_cosine = subparsers.add_parser("agg-mfp-cosine", help="Calculate cosine similarity of aggregated Morgan FPs")
parser_agg_mfp_cosine.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_agg_mfp_cosine.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_agg_mfp_cosine.set_defaults(func=calc_agg_mfp_cosine_sim)

# Concat mfp Tanimoto similarity
parser_concat_mfp = subparsers.add_parser("concat-mfp-tanimoto", help="Calculate Tanimoto similarity of concatenated Morgan FPs")
parser_concat_mfp.add_argument("dataset", help="Dataset name, e.g., 'sprhea'")
parser_concat_mfp.add_argument("toc", help="TOC name, e.g., 'v3_folded_pt_ns'")
parser_concat_mfp.set_defaults(func=calc_concat_mfp_sim)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()