import yaml
from pathlib import Path

project_dir = Path(__file__).parent.parent

with open(project_dir / "config.yml", 'r') as f:
    configs = yaml.safe_load(f)

filepaths = {}
for k, v in configs['dirs'].items():
    filepaths[k] = Path(v)
    if k in configs['subdirs']:
        for subdir in configs['subdirs'][k]:
            sub_k = f"{k}_{subdir}"
            filepaths[sub_k] = filepaths[k] / subdir