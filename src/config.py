import yaml
from pathlib import Path

project_dir = Path(__file__).parent.parent

with open(project_dir / "config.yaml", 'r') as f:
    configs = yaml.safe_load(f)

filepaths = {}
for k, v in configs['dirs'].items():
    filepaths[k] = Path(v)
    if k in configs['subdirs']:
        for subdir in configs['subdirs'][k]:
            
            if subdir == k:
                raise ValueError(f"Keys for dirs and subdirs should not be the same to avoid conflicts. Please rename {subdir}")

            sub_k = f"{subdir}"
            filepaths[sub_k] = filepaths[k] / subdir