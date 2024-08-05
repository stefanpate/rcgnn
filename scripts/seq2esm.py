from catalytic_function.utils import ensure_dirs, retrive_esm1b_embedding

outdir = f"/projects/p30041/spn1560/hiec/data/sprhea/esm/"
fasta_path = f"/home/spn1560/hiec/data/sprhea/v3_incremental.fasta"

ensure_dirs(outdir)
retrive_esm1b_embedding(fasta_path, outdir)
