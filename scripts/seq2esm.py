from src.utils import ensure_dirs, retrive_esm1b_embedding

outdir = f"/projects/p30041/spn1560/hiec/data/sprhea/esm2/"
fasta_path = f"/projects/p30041/spn1560/hiec/data/sprhea/v3_folded_pt_ns.fasta"
model_loc = "/projects/p30041/spn1560/hiec/pretrained_models/esm2_t33_650M_UR50D.pt"
_include = "per_tok"

ensure_dirs(outdir)
retrive_esm1b_embedding(fasta_path, outdir, model_loc, _include)
