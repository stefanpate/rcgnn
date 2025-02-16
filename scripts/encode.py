import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
import mlflow
from itertools import chain
from rdkit import Chem

from src.cheminfo import get_r_hop_from_rc
from src.ml_utils import (
    featurize_data,
    construct_model,
    mlflow_to_omegaconf
)

def radially_mask_reactions(data: pd.DataFrame, radius: int):
    '''
    Args
    ----
    data:pd.DataFrame
        Reaction data
    radius:int
        Number of hops
    '''
    to_drop = []
    fragmented_data = []
    for i, row in data.iterrows():
            smarts = row['smarts']
            rc = row['reaction_center']
            rcts, pdts = [elt.split('.') for elt in smarts.split('>>')]
            rrcs, prcs = rc[0], rc[1]


            fragment_smarts = [[], []]
            fragment_rcs = [[], []]

            for (smiles, rc) in zip(rcts, rrcs):
                fragment_smiles, fragment_rc = get_r_hop_from_rc(smiles, rc, radius)

                if '.' in fragment_smiles:
                    to_drop.append(i)
                    break

                fragment_smarts[0].append(fragment_smiles)
                fragment_rcs[0].append(fragment_rc)
            
            for (smiles, rc) in zip(pdts, prcs):
                fragment_smiles, fragment_rc = get_r_hop_from_rc(smiles, rc, radius)
                
                if '.' in fragment_smiles:
                    to_drop.append(i)
                    break

                fragment_smarts[1].append(fragment_smiles)
                fragment_rcs[1].append(fragment_rc)

            fragment_smarts = '.'.join(fragment_smarts[0]) + '>>' + '.'.join(fragment_smarts[1])

            fragment_rcs = (fragment_rcs[0], fragment_rcs[1])
            fragmented_data.append((fragment_smarts, fragment_rcs))
    
    fragmented_data = pd.DataFrame(fragmented_data, columns=['smarts', 'reaction_center'])
    data[['smarts', 'reaction_center']] = fragmented_data

    # Drop those that cannot be sanitized; will run into chemprop issues
    for i, row in data.iterrows():
        smarts = row['smarts']
        smiles = chain(*[elt.split('.') for elt in smarts.split('>>')])
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is None:
                to_drop.append(i)
                break
    
    data.drop(to_drop, inplace=True)
    data.reset_index(drop=True, inplace=True)

@hydra.main(version_base=None, config_path="../configs", config_name="encode")
def main(outer_cfg: DictConfig):
    mlflow.set_tracking_uri(outer_cfg.tracking_uri)
    run_data = mlflow.get_run(run_id=outer_cfg.run_id)
    cfg, artifacts_path = mlflow_to_omegaconf(run_data)
    run_path = artifacts_path.parent

    rng = np.random.default_rng(seed=cfg.data.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    splits = []
    for i in range(cfg.data.n_splits):
        split = pd.read_parquet(
            Path(outer_cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet"
        )
        split['protein_embedding'] = split['protein_embedding'].apply(lambda x : np.array(x))
        splits.append(split)

    more_data = pd.read_parquet(
        Path(outer_cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet"
    )
    more_data['protein_embedding'] = more_data['protein_embedding'].apply(lambda x : np.array(x))
    splits.append(more_data)
    data = pd.concat(splits, ignore_index=True)
    data = data.loc[data['y'] == 1].reset_index(drop=True) # Take out negative datapoints

    # Radially mask reactions
    if outer_cfg.radial_mask:
        radially_mask_reactions(data, outer_cfg.r_hop)

    _, dataloader, featurizer = featurize_data(
        cfg=cfg,
        rng=rng,
        val_data=data,
        shuffle_val=False
    )

    # Construct model
    embed_dim = data.loc[0, 'protein_embedding'].shape[0]
    ckpt_dir = run_path / 'checkpoints' 
    ckpt = ckpt_dir / next(ckpt_dir.glob("*.ckpt"))
    model = construct_model(cfg, embed_dim, featurizer, device, ckpt=ckpt)

    print("Encoding")
    with torch.no_grad():
        fingerprints = [
            model.fingerprint(bmg=batch.bmg, X_d=batch.X_d)
            for batch in dataloader
        ]
        fingerprints = torch.cat(fingerprints, 0)

    fingerprints = fingerprints.detach().numpy().astype(np.float32)

    # Save
    data["output_reaction_embeddings"] = list(fingerprints[: , : cfg.model.d_h_encoder])
    data["output_protein_embeddings"] = list(fingerprints[: , cfg.model.d_h_encoder :])
    p_embeds = data.loc[:, ["protein_idx", "pid", "output_protein_embeddings"]]
    r_embeds = data.loc[:, ["reaction_idx", "rid", "output_reaction_embeddings"]]
    p_embeds.drop_duplicates(inplace=True, subset="pid")
    r_embeds.drop_duplicates(inplace=True, subset="rid")
    p_embeds.rename(columns={'output_protein_embeddings': "embedding"}, inplace=True)
    r_embeds.rename(columns={'output_reaction_embeddings': "embedding"}, inplace=True)
    p_embeds.to_parquet("protein_embeddings.parquet", index=False)
    r_embeds.to_parquet("reaction_embeddings.parquet", index=False)

if __name__ == '__main__':
    main()