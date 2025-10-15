import pandas as pd
import hydra
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
from sklearn.metrics import f1_score
from concurrent.futures import ProcessPoolExecutor
from src.ml_utils import downsample_negatives
from functools import partial
import json
from tqdm import tqdm

def tune_dt(n_thresholds: int, inner_split_path: str, n_bootstraps: int, seed: int, neg_multiples: list[int]) -> dict[int, float]:
    '''
    Tune decision threshold for several negative multiples / one inner split using bootstrapping.
    '''
    thresholds = np.linspace(0, 1, num=n_thresholds)
    rng = np.random.default_rng(seed=seed)
    bootstrapped_thresholds = {}
    for neg_multiple in neg_multiples:
        target_output = pd.read_parquet(inner_split_path)
        downsample_negatives(target_output, neg_multiple, rng)
        target_output.reset_index(drop=True, inplace=True)
        best_th = thresholds[0]
        best_f1 = 0
        for th in thresholds:
            f1s = []
            for _ in range(n_bootstraps):
                indices = rng.integers(0, len(target_output), len(target_output))
                y_sample = target_output['y'].iloc[indices]
                probas_sample = target_output['logits'].iloc[indices]
                y_pred = (probas_sample > th).astype(np.int32)
                f1 = f1_score(y_true=y_sample, y_pred=y_pred, average='binary')
                f1s.append(f1)

            mean_f1 = sum(f1s) / len(f1s)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_th = th
        
        bootstrapped_thresholds[neg_multiple] = best_th

    return bootstrapped_thresholds

@hydra.main(version_base=None, config_path="../configs", config_name="tune_decision_threshold")
def main(cfg: DictConfig):
    _tune_dt = partial(
        tune_dt,
        n_thresholds=cfg.n_thresholds,
        n_bootstraps=cfg.n_bootstraps,
        seed=cfg.seed,
        neg_multiples=cfg.neg_multiples
    )

    outer2inner = cfg.split_pairs
    inner2outer = {v: k for k, v in outer2inner.items()}
    tasks = [Path(cfg.filepaths.results) / "predictions" / elt / "target_output.parquet" for elt in inner2outer.keys()]

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
        results = tqdm(list(executor.map(_tune_dt, tasks)))

    for outer, best_ths in zip(outer2inner.values(), results):
        with open(Path(cfg.filepaths.results) / "predictions" / outer / f"best_thresholds.json", 'w') as f:
            json.dump(best_ths, f)