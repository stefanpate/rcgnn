import pandas as pd
from pathlib import Path
from src.cross_validation import sample_negatives
import numpy as np
import json

dir = "/home/stef/quest_data/hiec/scratch/sprhea_v3_folded_pt_ns"
strats = [
    'rcmcs',
    'homology',

]

rng = np.random.default_rng(seed=1234)

def up_sample(df: pd.DataFrame, rng) -> tuple[list[tuple[int]], list[int]]:
    X = list(zip(df['protein_idx'], df['reaction_idx']))
    y = df['y'].to_list()

    X, y = sample_negatives(
        X,
        y,
        neg_multiple=10,
        rng=rng
    )
    return X, y

def assemble_data(X, y, sprhea, df) -> pd.DataFrame:
    prots = {pidx: df.loc[df['protein_idx'] == pidx, ['protein_idx', 'pid', 'protein_embedding']].iloc[0].to_list() for pidx in df['protein_idx'].unique()}
    rxns = {ridx: df.loc[df['reaction_idx'] == ridx, ['reaction_idx', 'rid',]].iloc[0].to_list() for ridx in df['reaction_idx'].unique()}
    cols = ['protein_idx', 'reaction_idx', 'pid', 'rid', 'protein_embedding', 'smarts', 'am_smarts', 'reaction_center', 'y']

    data = []
    for (pidx, ridx), yi in zip(X, y):
        prot = prots[pidx]
        rxn = rxns[ridx]
        rxn_data = sprhea[rxn[1]]
        smarts = rxn_data['smarts']
        am_smarts = rxn_data['am_smarts']
        reaction_center = rxn_data['rcs']

        data.append(
            (
                prot[0],
                rxn[0],
                prot[1],
                rxn[1],
                list(prot[2]),
                smarts,
                am_smarts,
                reaction_center,
                yi
            )
        )
    return pd.DataFrame(data, columns=cols)

if __name__ == "__main__":

    with open("/home/stef/quest_data/hiec/data/sprhea/v3_folded_pt_ns.json", 'r') as f:
        sprhea = json.load(f)

    for strat in strats:
        print(f"Processing strategy: {strat}")
        old_test_fn = f"{dir}/{strat}/3fold/test.parquet"
        old_test = pd.read_parquet(old_test_fn)

        X, y = up_sample(old_test, rng)
        upsampled_test = assemble_data(
            X,
            y,
            sprhea,
            old_test
        )
        print(upsampled_test['y'].mean())
        upsampled_test.to_parquet(old_test_fn)
        print(f"Saved upsampled test set to: {old_test_fn}")