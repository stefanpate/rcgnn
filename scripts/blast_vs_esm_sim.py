from Bio import pairwise2
from src.utils import load_design_matrix
import pandas as pd

def blast_fraction_identity(seq1, seq2):
    '''
    Runs global alignment between two sequences and
    returns identity fraction
    '''
    # Returns score that sums number of exact matches in global alignment
    score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
    percent_identity = score / max(len(seq1), len(seq2))
    return percent_identity

def load_embeds_seqs(ds_name, embed_type='esm'):
    df = pd.read_csv(f"../data/{ds_name}/{ds_name}.csv", delimiter='\t')
    df.set_index('Entry', inplace=True)
    sample_idx = {}
    seqs = []

    for i, elt in enumerate(df.index):
        seqs.append(df.loc[elt, 'Sequence'])
        sample_idx[elt] = i

    idx_sample = {v:k for k,v in sample_idx.items()}
    X = load_design_matrix(ds_name, embed_type, sample_idx, do_norm=True)

    return X, seqs, idx_sample


if __name__ == '__main__':
    import numpy as np
    rng = np.random.default_rng(seed=1234)
    n_samples = 1000

    # Load swissprot esm embeddings & AA seqs
    print("Loading swissprot")
    X_sp, seqs_sp, idx_sample_sp = load_embeds_seqs('swissprot')
    
    # Do pairwise global alignment for all sequences
    print("Computing similarity")
    idxs = rng.choice(np.arange(X_sp.shape[0]), size=(n_samples,), replace=False)
    entry1 = []
    entry2 = []
    blast_sim = []
    cosine_sim = []
    for n, i in enumerate(idxs):
        for j in idxs[n+1:]:
            seq1, seq2 = seqs_sp[i], seqs_sp[j]
            vec1, vec2 = X_sp[i, :].reshape(-1,), X_sp[j, :].reshape(-1,)
            blast_sim.append(blast_fraction_identity(seq1, seq2))
            cosine_sim.append(np.dot(vec1, vec2))
            entry1.append(idx_sample_sp[i])
            entry2.append(idx_sample_sp[j])

    # Save
    print("Saving")
    df = pd.DataFrame({"Entry 1": entry1,
                       "Entry 2": entry2,
                       "Embedding cosine similarity": cosine_sim,
                       "Percent Identity": blast_sim})
    
    df.to_csv(f"../artifacts/protein_dataset_compare/blast_esm_cosine_similarity_for_{int((n_samples**2)/2)}_random_swissprot_pairs.csv", sep='\t', index=False)

    print('Done')