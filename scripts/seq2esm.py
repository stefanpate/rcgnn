from src.utils import ensure_dirs, retrive_esm1b_embedding
from hydra import initialize, compose

with initialize(config_path="../configs/filepaths", version_base=None):
    cfg = compose(config_name="base")

outdir = f"{cfg.data}/time_split/esm"
fasta_path = f"{cfg.data}/time_split/250915_time_split.fasta"
model_loc = f"{cfg.pretrained_models}/esm1b_t33_650M_UR50S.pt"
_include = "mean"

ensure_dirs(outdir)
retrive_esm1b_embedding(fasta_path, outdir, model_loc, _include)
