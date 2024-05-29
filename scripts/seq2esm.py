from src.CLEAN.utils import *

dataset = 'sprhea'
toc = "sp_folded_pth"
outdir = f"/projects/p30041/spn1560/hiec/data/{dataset}/esm/"
csv_path = f"../data/{dataset}/{toc}.csv" 
fasta_path = f"../data/{dataset}/{toc}.fasta"


ensure_dirs(outdir)
# csv_to_fasta(csv_path, fasta_path)
retrive_esm1b_embedding(fasta_path, outdir)
