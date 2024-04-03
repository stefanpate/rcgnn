from Bio import pairwise2
from src.utils import load_design_matrix
import pandas as pd
import queue
import numpy as np

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

def blast_all_refs(query_entry_seqs, entry_sim_scores, ref_seqs):
    while True:
        # Try to get query from input queue
        try:
            entry, query_seq = query_entry_seqs.get_nowait()
        except queue.Empty:
            break

        scores = []
        for ref_seq in ref_seqs:
            score = blast_fraction_identity(query_seq, ref_seq)
            scores.append(score)
        
        scores = np.array(scores)
        mean_score, max_score = scores.mean(), scores.max()
        entry_sim_scores.put((entry, max_score, mean_score)) # Put onto output queue
        
    return True

if __name__ == '__main__':
    import pandas as pd
    import multiprocessing

    ds_name = 'price'

    # Load swissprot esm embeddings & AA seqs
    print("Loading swissprot")
    X_sp, seqs_sp, idx_sample_sp = load_embeds_seqs('swissprot')
    
    # Load dataset esm embeddings & AA seqs
    print(f"Loading {ds_name}")
    X_ds, seqs_ds, idx_sample_ds = load_embeds_seqs(ds_name)
    sample_idx_ds = {v:k for k,v in idx_sample_ds.items()}
    
    # Matrix multiply embeds
    print("Computing cosine similarity")
    cosine_sim_max = (X_sp @ X_ds.T).max(axis=0)
    cosine_sim_mean = (X_sp @ X_ds.T).mean(axis=0)

    # Don't need embeddings anymore
    del X_sp
    del X_ds

    # Do pairwise global alignment for all sequences
    n_cores = max(multiprocessing.cpu_count(), 1)
    print(f"Computing percent sequence identity w/ {n_cores} processes")
    blast_sim = []
    align_times = []
    processes = []
    entry_sim_scores = multiprocessing.Queue() # Output queue
    query_entry_seqs = multiprocessing.Queue() # Input queue

    # Queue (entry, seq pairs) from query dataset
    for i, elt in enumerate(seqs_ds):
        query_entry_seqs.put((idx_sample_ds[i], elt))

    # Create processes
    for i in range(n_cores):
        p = multiprocessing.Process(target=blast_all_refs, args=(query_entry_seqs, entry_sim_scores, seqs_sp))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Unppack PI scores, making sure order of samples consistent w/ pre-multiprocessing
    max_blast_sims = [None for i in range(len(idx_sample_ds))]
    mean_blast_sims = [None for i in range(len(idx_sample_ds))]
    while not entry_sim_scores.empty():
        entry, max_score, mean_score = entry_sim_scores.get()
        max_blast_sims[sample_idx_ds[entry]] = max_score
        mean_blast_sims[sample_idx_ds[entry]] = mean_score

    # Save
    print("Saving")
    df = pd.DataFrame({"Entry": idx_sample_ds.values(),
                       "Max cosine similarity": cosine_sim_max,
                       "Mean cosine similarity": cosine_sim_mean,
                       "Max percent identity": max_blast_sims,
                       "Mean percent identity": mean_blast_sims})
    
    df.to_csv(f"../artifacts/protein_dataset_compare/swissprot_{ds_name}.csv", sep='\t', index=False)

    print('Done')