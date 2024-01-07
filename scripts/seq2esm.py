from src.CLEAN.utils import *

dataset = 'new'
outdir = f"../data/{dataset}/esm/"
csv_path = f"../data/{dataset}/{dataset}.csv" 
fasta_path = f"../data/{dataset}/{dataset}.fasta"


ensure_dirs(outdir)
csv_to_fasta(csv_path, fasta_path)
retrive_esm1b_embedding(fasta_path, outdir)
