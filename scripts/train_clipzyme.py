import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from src.clip import EnzymeReactionCLIP, ClipDataset, clip_collate, EnzymeReactionCLIPBN, create_protein_graph
from src.similarity import load_similarity_matrix
from src.utils import load_embed
import torch
from torch.utils.data import DataLoader
from logging import getLogger

def downsample_negatives(data: pd.DataFrame, neg_multiple: int, rng: np.random.Generator):
    neg_idxs = data[data['y'] == 0].index
    n_to_rm = len(neg_idxs) - (len(data[data['y'] == 1]) * neg_multiple)
    
    if n_to_rm <= 0:
        return
    
    idx_to_rm = rng.choice(neg_idxs, n_to_rm, replace=False)
    data.drop(axis=0, index=idx_to_rm, inplace=True)

def load_data(fp, use_prot_struct, blacklist):
    df = pd.read_parquet(fp)
    if use_prot_struct:
        df.drop(columns=['protein_embedding'], inplace=True)
        df = df[~df['pid'].isin(blacklist)].reset_index(drop=True)
    else:
        df['protein_embedding'] = df['protein_embedding'].apply(lambda x : np.array(x))

    return df

def get_protein_graphs(pids, cfg):
    protein_graphs = {}
    for pid in pids:
        esm_path = Path(cfg.filepaths.scratch) / "esm2" / f"{pid}.pt"
        cif_path = Path(cfg.filepaths.scratch) / "af2" / f"AF-{pid}-F1-model_v4.cif"
        esm_emb = load_embed(esm_path, embed_type='esm2')
        protein_graphs[pid] = create_protein_graph(cif_path, esm_emb)
    
    return protein_graphs

log = getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="train_clipzyme")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)

    # Could not get these structures
    af2_blacklist = set()
    with open(Path(cfg.filepaths.data) / "sprhea" / "af2_blacklist.txt", 'r') as f:
        for line in f:
            af2_blacklist.add(line.strip())

    # Clipzyme excludes greater than 650 residues
    toc = pd.read_csv(Path(cfg.filepaths.data) / cfg.data.dataset / f"{cfg.data.toc}.csv", sep='\t')
    toc['seq_len'] = toc['Sequence'].str.len()
    af2_blacklist.update(set(toc.loc[toc['seq_len'] > 650, 'Entry'].to_list()))

    # Load data
    log.info("Loading data...")
    train_val_splits = []
    for i in range(cfg.data.n_splits):
        split = load_data(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / f"train_val_{i}.parquet",
            cfg.model.use_protein_structure,
            af2_blacklist
        )
        train_val_splits.append(split)


    # Arrange data
    if cfg.data.split_idx == -2: # Train on full dataset
        if cfg.test_only:
            raise ValueError("Cannot do test only with full data training")
        
        version = f"{cfg.data.dataset}_{cfg.data.toc}_{cfg.data.split_strategy}_full_data"
        more_train_data = load_data(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet",
            cfg.model.use_protein_structure,
            af2_blacklist
        )
        train_data = pd.concat(train_val_splits + [more_train_data], ignore_index=True)
        val_data = None
    elif cfg.data.split_idx == -1: # Test on outer fold
        version = f"{cfg.data.dataset}_{cfg.data.toc}_{cfg.data.split_strategy}_outer_fold"
        val_data = load_data(
            Path(cfg.filepaths.scratch) / cfg.data.subdir_patt / "test.parquet",
            cfg.model.use_protein_structure,
            af2_blacklist
        )

        if not cfg.test_only:
            train_data = pd.concat(train_val_splits, ignore_index=True)
    else: # Test on inner fold
        version = f"{cfg.data.dataset}_{cfg.data.toc}_{cfg.data.split_strategy}_inner_fold_{cfg.data.split_idx + 1}_of_{cfg.data.n_splits}"
        val_data = train_val_splits[cfg.data.split_idx]
        
        if not cfg.test_only:
            train_data = pd.concat([train_val_splits[i] for i in range(cfg.data.n_splits) if i != cfg.data.split_idx], ignore_index=True)
            downsample_negatives(val_data, 1, rng) # Inner fold val are oversampled

    if not cfg.test_only:
        downsample_negatives(train_data, cfg.data.neg_multiple, rng) # Inner fold train are oversampled

    if cfg.model.use_protein_structure:
        log.info("Creating protein graphs...")
        val_pgs = get_protein_graphs(val_data['pid'].unique(), cfg) if val_data is not None else {}
        test_pgs = get_protein_graphs(train_data['pid'].unique(), cfg) if not cfg.test_only else {}
        pid2prot = {**val_pgs, **test_pgs}
        del val_pgs, test_pgs
        fmt_data = lambda df: (df['am_smarts'].tolist(), df['pid'].to_list(), torch.tensor(df['y'].to_numpy()).float().unsqueeze(1))

        # Just in case filter out entries without protein graphs
        pg_blacklist = [pid for pid in pid2prot if pid2prot[pid] is None]
        if val_data is not None:
            val_data = val_data[~val_data['pid'].isin(pg_blacklist)].reset_index(drop=True)
        if not cfg.test_only:
            train_data = train_data[~train_data['pid'].isin(pg_blacklist)].reset_index(drop=True)
    
    else:
        pid2prot = {}
        if val_data is not None:
            for pid in val_data['pid'].unique():
                pid2prot[pid] = torch.tensor(val_data.loc[val_data['pid'] == pid, 'protein_embedding'].iloc[0])
        
        if not cfg.test_only:
            for pid in train_data['pid'].unique():
                if pid not in pid2prot:
                    pid2prot[pid] = torch.tensor(train_data.loc[train_data['pid'] == pid, 'protein_embedding'].iloc[0])
        
        fmt_data = lambda df: (df['am_smarts'].tolist(), df['pid'].to_list(), torch.tensor(df['y'].to_numpy()).float().unsqueeze(1))
    
    # Prepare data
    val_reactions, val_pids, val_targets = (None, None, None) if val_data is None else fmt_data(val_data)

    if not cfg.test_only:
        train_reactions, train_pids, train_targets = fmt_data(train_data)

        train_dataset = ClipDataset(
            reactions=train_reactions,
            pids=train_pids,
            pid2prot=pid2prot,
            targets=train_targets
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=clip_collate,
            batch_size=cfg.training.batch_size,
        )

    val_dataset = None if val_data is None else ClipDataset(
        reactions=val_reactions,
        pids=val_pids,
        pid2prot=pid2prot,
        targets=val_targets
    )
    val_dataloader = None if val_data is None else DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=clip_collate,
        batch_size=cfg.training.batch_size,
    )

    del train_data

    exp = cfg.exp or "Default"
    if cfg.model.ckpt_fn is not None:
        ckpt = Path(cfg.filepaths.runs) / exp / version / 'checkpoints' / cfg.model.ckpt_fn.replace('_', '=')
    else:
        ckpt = None

    # Construct model
    model = EnzymeReactionCLIPBN(
        model_hps=cfg.model,
        negative_multiple=cfg.data.neg_multiple,
        positive_multiplier=cfg.training.pos_multiplier,
    )

    if not cfg.test_only:
        # Track
        logger = CSVLogger(
            name=exp,
            save_dir=cfg.filepaths.runs,
            version=version,
        )

        # Train
        trainer = pl.Trainer(
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=cfg.training.n_epochs,
            logger=logger,
            default_root_dir=Path(cfg.filepaths.runs) / exp / version,
            detect_anomaly=True,
            log_every_n_steps=1,
            precision='bf16' if torch.cuda.is_available() else 32,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt,
        )

    # Predict
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )
        val_preds = trainer.predict(
            model,
            val_dataloader,
            ckpt_path=ckpt if cfg.test_only else None, # Test only implies loaded from checkpoint / else follow gets best from training
        )

    logits = np.vstack(val_preds).reshape(-1,)
    
    # Save outputs
    target_output = val_data.loc[:, ["protein_idx", "reaction_idx", "pid", "rid", "y"]]
    target_output.loc[:, "logits"] = logits # These are not really logits but keep naming for consistency
    
    # Get max sims
    sim = cfg.data.split_strategy if cfg.data.split_strategy != 'homology' else 'gsi'
    try:
        S = load_similarity_matrix(
            sim_path=Path(cfg.filepaths.similarity_matrices),
            dataset=cfg.data.dataset,
            toc=cfg.data.toc,
            sim_metric=sim
        )
    except ValueError as e:
        print(e)
        target_output.to_parquet("target_output.parquet", index=False)
        return

    if sim in ['rcmcs', 'drfp']:
        val_idx = target_output.loc[:, 'reaction_idx'].to_list()
    elif sim in ['gsi', 'esm']:
        val_idx = target_output.loc[:, 'protein_idx'].to_list()

    train_idx = [i for i in range(S.shape[0]) if i not in val_idx]
    max_sims = S[:, val_idx][train_idx].max(axis=0)
    target_output.loc[:, "max_sim"] = max_sims

    subdir = Path(f"{exp}/{version}")
    if not subdir.exists():
        subdir.mkdir(parents=True)

    target_output.to_parquet(subdir / "target_output.parquet", index=False)

if __name__ == '__main__':
    main()