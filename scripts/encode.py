'''
Load fitted model and encode reaction and protein data
'''

from src.utils import load_json, load_precomputed_embeds, construct_sparse_adj_mat
from src.data import RxnRCDatapoint
from src.cross_validation import sample_negatives
from src.task import construct_model, construct_featurizer
from src.config import filepaths

import numpy as np
from argparse import ArgumentParser
import torch
import os
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument("model_name", type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    model_name = args.model_name
    chkpt_dir = filepaths['trained_models'] / "gnn" / args.model_name / "version_0" / "checkpoints"
    chkpt_path = next(chkpt_dir.glob("*.ckpt"))

    # Load model hyperparameters
    hps = load_json(filepaths['named_model_hps'] / f"{args.model_name}.json")

    # Load data
    print("Loading data")
    adj, idx_sample, idx_feature = construct_sparse_adj_mat(hps["dataset_name"], hps["toc"])
    sample_idx = {v: k for k, v in idx_sample.items()}
    positive_pairs = list(zip(*adj.nonzero()))
    X, y = sample_negatives(positive_pairs, hps["neg_multiple"], hps["seed"])
    embeds = load_precomputed_embeds(hps["dataset_name"], hps["toc"], hps["embed_type"], sample_idx, do_norm=False)
    embed_dim = embeds.shape[1]
    known_rxns = load_json(f"../data/{hps['dataset_name']}/{hps['toc']}.json") # Load reaction dataset

    dataset_base, generate_dataloader, featurizer = construct_featurizer(hps)
    model = construct_model(hps, featurizer, embed_dim, chkpt_path)
    datapoint_from_smi = RxnRCDatapoint.from_smi

    # Featurize data
    print("Constructing datasets & dataloaders")
    datapoints_train = []
    for (i, j), label in zip(X, y):
        rxn = known_rxns[idx_feature[j]]
        label = np.array([label], dtype=np.float32)
        prot_embed = embeds[i, :]
        datapoints_train.append(datapoint_from_smi(rxn, y=label, x_d=prot_embed))

    dataset_train = dataset_base(datapoints_train, featurizer=featurizer)
    data_loader_train = generate_dataloader(dataset_train, shuffle=False)

    print("Encoding")
    with torch.no_grad():
        fingerprints = [
            model.fingerprint(bmg=batch.bmg, X_d=batch.X_d)
            for batch in data_loader_train
        ]
        fingerprints = torch.cat(fingerprints, 0)

    fingerprints = fingerprints.detach()

    # Save embeds
    print("Saving embeds")
    save_dir = filepaths['embeddings'] /  args.model_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, "reactions"))
        os.mkdir(os.path.join(save_dir, "enzymes"))

    rxn_embeds = defaultdict(lambda: None)
    prot_embeds = defaultdict(lambda: None)
    for (p, r), concat_embed in zip(X, fingerprints):
        rid = idx_feature[r]
        pid = idx_sample[p]
        rxn_embeds[rid] = concat_embed[:hps["d_h_encoder"]].clone()
        prot_embeds[pid] = concat_embed[hps["d_h_encoder"]:].clone()

    for rid, embed in rxn_embeds.items():
        torch.save(embed, os.path.join(save_dir, "reactions", f"{rid}.pt"))

    for pid, embed in prot_embeds.items():
        torch.save(embed, os.path.join(save_dir, "proteins", f"{pid}.pt"))