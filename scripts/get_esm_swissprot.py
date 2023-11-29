from src.CLEAN.utils import *

ensure_dirs("data/clean_data/esm_data")
ensure_dirs("data/clean_data/pretrained")
csv_to_fasta("data/clean_data/split100.csv", "data/clean_data/split100.fasta")
retrive_esm1b_embedding("split100")