from clipzyme.utils.screening import process_mapped_reaction
from clipzyme.utils.loading import default_collate
from clipzyme.models.chemprop import DMPNNEncoder
from clipzyme.models.egnn import EGNN_Sparse_Network
from clipzyme.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
)

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data
from torch_scatter import scatter
import copy
from pytorch_lightning import LightningModule
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from omegaconf import DictConfig
import math
from typing import Union
import Bio
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_3to1

def clip_collate(datapoints: list[dict[str, tuple[Data, Data] | torch.Tensor]]) -> dict[str, dict[str, Batch] | torch.Tensor]:
    rxn_batch, prot_batch, target_batch = [], [], []
    for elt in datapoints:
        rxns, prots, targets = elt['reaction'], elt['protein'], elt['target']
        rcts, pdts = rxns
        rxn_batch.append({'reactants': rcts, 'products': pdts})
        prot_batch.append(prots)
        target_batch.append(targets)
    
    rxn_batch = default_collate(rxn_batch)
    if type(prot_batch[0]) == torch.Tensor:
        prot_batch = torch.stack(prot_batch)
    else:
        prot_batch = default_collate([{"graph": g} for g in prot_batch])
    
    target_batch = torch.stack(target_batch)

    return {"reaction": rxn_batch, "protein": prot_batch, "target": target_batch}

class EnzymeReactionCLIP(LightningModule):
    '''
    Uses "pseudo transition state" representation of reactions and
    pre-trained protein embedding. Adapted from Mikhael, Chinn, & Barzilay 2024
    aka CLIPZyme. https://github.com/pgmikhael/clipzyme/
    '''
    def __init__(self, model_hps: DictConfig, negative_multiple: int = 1, positive_multiplier: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.args = model_hps
        pos_weight = torch.ones([1]) * negative_multiple * positive_multiplier
        self.register_buffer("pos_weight", pos_weight) # Adds non-trainable tensor to model state dict thus goes to right device

        if model_hps.use_protein_structure:
            self.protein_encoder = EGNN_Sparse_Network(model_hps)
            self.ln_final = nn.LayerNorm(model_hps.protein_dim)
        else:
            self.prot_linear_layer = torch.nn.Linear(
                in_features=1280,
                out_features=model_hps.chemprop_hidden_dim,
            )

        wln_diff_args = copy.deepcopy(model_hps)
        if model_hps.model_name != "enzyme_reaction_clip_wldnv1":
            self.wln = DMPNNEncoder(model_hps)  # WLN for mol representation
            wln_diff_args = copy.deepcopy(model_hps)
            wln_diff_args.chemprop_edge_dim = model_hps.chemprop_hidden_dim
            # wln_diff_args.chemprop_num_layers = 1 ## Sic clipzyme original code
            self.wln_diff = DMPNNEncoder(wln_diff_args)

    def encode_reaction(self, batch):
        '''
        Pseudo transition state reaction encoder
        '''
        reactant_edge_feats = self.wln(batch["reactants"])[
            "edge_features"
        ]  # N x D, where N is all the nodes in the batch
        product_edge_feats = self.wln(batch["products"])[
            "edge_features"
        ]  # N x D, where N is all the nodes in the batch

        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=reactant_edge_feats,
            batch=batch["reactants"].batch,
        )
        dense_product_edge_feats = to_dense_adj(
            edge_index=batch["products"].edge_index,
            edge_attr=product_edge_feats,
            batch=batch["products"].batch,
        )
        sum_vectors = dense_reactant_edge_feats + dense_product_edge_feats

        # undensify
        flat_sum_vectors = sum_vectors.sum(-1)
        new_edge_indices = [dense_to_sparse(E)[0] for E in flat_sum_vectors]
        new_edge_attr = torch.vstack(
            [sum_vectors[i, e[0], e[1]] for i, e in enumerate(new_edge_indices)]
        )
        cum_num_nodes = torch.cumsum(torch.bincount(batch["reactants"].batch), 0)
        new_edge_index = torch.hstack(
            [new_edge_indices[0]]
            + [ei + cum_num_nodes[i] for i, ei in enumerate(new_edge_indices[1:])]
        )
        reactants_and_products = batch["reactants"].clone()
        reactants_and_products.edge_attr = new_edge_attr
        reactants_and_products.edge_index = new_edge_index

        # apply a separate WLN to the difference graph
        wln_diff_output = self.wln_diff(reactants_and_products)

        if self.args.aggregate_over_edges:
            edge_feats = wln_diff_output["edge_features"]
            edge_batch = batch["reactants"].batch[new_edge_index[0]]
            graph_feats = scatter(edge_feats, edge_batch, dim=0, reduce="sum")
        else:
            sum_node_feats = wln_diff_output["node_features"]
            sum_node_feats, _ = to_dense_batch(
                sum_node_feats, batch["products"].batch
            )  # num_candidates x max_num_nodes x D
            graph_feats = torch.sum(sum_node_feats, dim=-2)
        return graph_feats
    
    def encode_protein(self, batch):
        if self.args.use_protein_structure:
            feats, coors = self.protein_encoder(batch)
            try:
                batch_idxs = batch["graph"]["receptor"].batch
            except:
                batch_idxs = batch["receptor"].batch
            protein_features = scatter(
                feats,
                batch_idxs,
                dim=0,
                reduce=self.args.pool_type,
            )

            # apply layer normalization
            protein_features = self.ln_final(protein_features)
        else:
            protein_features = self.prot_linear_layer(batch)
        
        return protein_features

    def dot_sig(self, R: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        logits = torch.mul(R, P).sum(dim=1).reshape(-1,1)
        return logits.sigmoid()

    def forward(self, batch) -> torch.Tensor:
        P = batch["protein"]
        P = self.encode_protein(P)
        P = P / P.norm(
            dim=1, keepdim=True
        )
        R = batch["reaction"]
        R = self.encode_reaction(R)
        R = R / R.norm(
            dim=1, keepdim=True
        )
        return self.dot_sig(R, P)
    
    def training_step(self, batch, batch_idx : int = 0):
        Y_hat = self.forward(batch)
        Y = batch["target"]
        mask = Y_hat.isfinite()
        loss = F.binary_cross_entropy_with_logits(Y_hat[mask], Y[mask], pos_weight=self.pos_weight)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["target"].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx: int = 0):
        Y_hat = self.forward(batch)
        Y = batch["target"]
        mask = Y_hat.isfinite()
        loss = F.binary_cross_entropy_with_logits(Y_hat[mask], Y[mask], pos_weight=self.pos_weight)
        self.log("val_loss", loss, batch_size=batch["target"].shape[0], prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx: int = 0):
        Y_hat = self.forward(batch)
        return Y_hat.detach().cpu().numpy()
    
    def configure_optimizers(self):
        """
        Obtain optimizers and hyperparameter schedulers for model
        cf. Clipzyme implementation
        """
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad], self.args
        )
        schedule = LinearWarmupCosineLRScheduler(
            optimizer, self.args
        )

        scheduler = {
            "scheduler": schedule,
            "monitor": self.args.monitor,
            "interval": self.args.scheduler_interval,
            "frequency": 1,
        }
        return [optimizer], [scheduler]

class EnzymeReactionCLIPBN(EnzymeReactionCLIP):
    '''
    Adds batch norm to protein embedding after linear layer
    '''
    def __init__(self, model_hps: DictConfig, negative_multiple: int = 1, positive_multiplier: int = 1):
        super().__init__(model_hps, negative_multiple, positive_multiplier)
        self.bn_rxn = nn.BatchNorm1d(model_hps.chemprop_hidden_dim)
        self.bn_prot = nn.BatchNorm1d(model_hps.protein_dim)

    def forward(self, batch) -> torch.Tensor:
        P = batch["protein"]
        P = self.encode_protein(P)
        P = self.bn_prot(P)
        R = batch["reaction"]
        R = self.encode_reaction(R)
        R = self.bn_rxn(R)
        return self.dot_sig(R, P)
    
class ClipDataset:
    def __init__(self, reactions: list[str], proteins: torch.Tensor | list[Data], targets: torch.Tensor):
        if len(reactions) != len(proteins) or len(reactions) != len(targets):
            raise ValueError("Length of reactions, protein_embeddings, and targets must be the same.")
        
        self.idx2smarts = {}
        self.smarts2rxn = {}
        for idx, sma in enumerate(reactions):
            self.idx2smarts[idx] = sma

            if sma not in self.smarts2rxn:
                self.smarts2rxn[sma] = process_mapped_reaction(sma)

        self.proteins = proteins
        self.targets = targets
    
    def __len__(self):
        return len(self.idx2smarts)
    
    def __getitem__(self, idx) -> dict[str, tuple[Data, Data] | torch.Tensor]:
        return {
            "reaction": self.smarts2rxn[self.idx2smarts[idx]],
            "protein": self.proteins[idx],
            "target": self.targets[idx],
        }

class AdamW(AdamW):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
    """

    def __init__(self, params, args):
        adam_betas = tuple(args.adam_betas)
        super().__init__(
            params=params,
            lr=args.lr,
            betas=adam_betas,
            weight_decay=args.weight_decay,
            amsgrad=args.use_amsgrad,
        )

class LinearWarmupCosineLRScheduler(_LRScheduler):
    '''
    cf. Clipzyme implementation
    '''
    def __init__(self, optimizer, args):
        max_epoch = args.max_epochs
        min_lr = args.warmup_min_lr
        init_lr = args.warmup_init_lr
        warmup_start_lr = args.warmup_start_lr
        warmup_steps = args.warmup_steps

        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        super().__init__(optimizer)

    def step(self, cur_epoch=0):
        # assuming the warmup iters less than one epoch
        self._step_count += 1
        cur_step = self._step_count
        if cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def build_NoamLike_LRSched(
    optimizer: Optimizer,
    warmup_steps: int,
    cooldown_steps: int,
    init_lr: float,
    max_lr: float,
    final_lr: float,
):
    r"""cf. chemprop
    """

    def lr_lambda(step: int):
        if step < warmup_steps:
            warmup_factor = (max_lr - init_lr) / warmup_steps
            return step * warmup_factor / init_lr + 1
        elif warmup_steps <= step < warmup_steps + cooldown_steps:
            cooldown_factor = (final_lr / max_lr) ** (1 / cooldown_steps)
            return (max_lr * (cooldown_factor ** (step - warmup_steps))) / init_lr
        else:
            return final_lr / init_lr

    return LambdaLR(optimizer, lr_lambda)

def create_protein_graph(cif_path: str, esm_embed: torch.Tensor) -> Union[Data, None]:
    """
    Create pyg protein graph from CIF file

    Parameters
    ----------
    cif_path : str
        Path to CIF file
    esm_path : str
        Path to ESM model (esm2_t33_650M_UR50D.pt)

    Returns
    -------
    data
        pygData object with protein graph
    """
    truncation_seq_length = 1022 # cf clipzyme
    try:
        raw_path = cif_path
        sample_id = "proteinX"
        protein_parser = Bio.PDB.MMCIFParser()
        protein_resolution = "residue"
        graph_edge_args = {"knn_size": 10}
        center_protein = True

        # parse pdb
        all_res, all_atom, all_pos = read_structure_file(
            protein_parser, raw_path, sample_id
        )
        # filter resolution of protein (backbone, atomic, etc.)
        atom_names, seq, pos = filter_resolution(
            all_res,
            all_atom,
            all_pos,
            protein_resolution=protein_resolution,
        )
        # generate graph
        data = build_graph(atom_names, seq, pos, sample_id)
        # kNN graph
        data = compute_graph_edges(data, **graph_edge_args)
        if center_protein:
            center = data["receptor"].pos.mean(dim=0, keepdim=True)
            data["receptor"].pos = data["receptor"].pos - center
            data.center = center

        # get sequence
        AA_seq = ""
        for char in seq:
            AA_seq += protein_letters_3to1[char]

        data.structure_sequence = AA_seq

        # compute embeddings
        data["receptor"].x = esm_embed[: min(len(AA_seq), truncation_seq_length) + 1]

        if len(data["receptor"].seq) != data["receptor"].x.shape[0]:
            return None

        if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
            data["receptor"].x = data.x

        if not hasattr(data, "structure_sequence"):
            data.structure_sequence = "".join(
                [protein_letters_3to1[char] for char in data["receptor"].seq]
            )

        coors = data["receptor"].pos
        feats = data["receptor"].x
        edge_index = data["receptor", "contact", "receptor"].edge_index
        assert (
            coors.shape[0] == feats.shape[0]
        ), f"Number of nodes do not match between coors ({coors.shape[0]}) and feats ({feats.shape[0]})"

        assert (
            max(edge_index[0]) < coors.shape[0] and max(edge_index[1]) < coors.shape[0]
        ), "Edge index contains node indices not present in coors"

        return data

    except Exception as e:
        print(f"Could not create protein graph because of the exception: {e}")
        return None


if __name__ == "__main__":
    from hydra import initialize, compose
    from time import perf_counter
    import pytorch_lightning as pl
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train_clipzyme")

    cif_path = "/home/stef/quest_data/hiec/data/sprhea/af2/AF-Q04JH4-F1-model_v4.cif"
    esm_embed = torch.load("/home/stef/quest_data/hiec/data/sprhea/esm2/Q04JH4.pt")['representations'][33]
    pg = create_protein_graph(cif_path, esm_embed)
    print(pg['receptor'].x.shape)
    print()
    # model = EnzymeReactionCLIP(cfg.model)
    # print(model)
    # print(model.pos_weight)

    # # model = EnzymeReactionCLIP()
    # # print(model)

    # ckpt = '/home/stef/quest_data/hiec/results/runs/debug/inner_fold_1_of_3/checkpoints/epoch=0-step=62-v1.ckpt'
    # trainer = pl.Trainer(max_epochs=1, logger=False)
    # trainer.fit(model, ckpt_path=ckpt)


    # model = EnzymeReactionCLIP.load_from_checkpoint(ckpt)
    # print(model)
    # print(model.pos_weight)

    reactions = [
        "[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][CH:6]=[O:7].[O:9]=[O:10].[OH2:8]>>[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][C:6](=[O:7])[OH:8].[OH:9][OH:10]",
        '[CH:1]([CH2:3][CH2:5][CH:6]([NH2:7])[C:8](=[O:9])[OH:10])=[O:4].[c:15]1([C:21]([NH2:22])=[O:23])[cH:16][n+:17]([CH:24]2[O:25][CH:27]([CH2:30][O:32][P:33](=[O:34])([OH:35])[O:36][P:37](=[O:38])([OH:39])[O:40][CH2:41][CH:42]3[O:43][CH:45]([n:48]4[cH:50][n:53][c:55]5[c:51]4[n:54][cH:59][n:61][c:60]5[NH2:62])[CH:46]([O:49][P:52](=[O:56])([OH:57])[OH:58])[CH:44]3[OH:47])[CH:28]([OH:31])[CH:26]2[OH:29])[cH:18][cH:19][cH:20]1.[OH:2][P:11](=[O:12])([OH:13])[OH:14]>>[C:1]([O:2][P:11](=[O:12])([OH:13])[OH:14])([CH2:3][CH2:5][CH:6]([NH2:7])[C:8](=[O:9])[OH:10])=[O:4].[C:15]1([C:21]([NH2:22])=[O:23])=[CH:16][N:17]([CH:24]2[O:25][CH:27]([CH2:30][O:32][P:33](=[O:34])([OH:35])[O:36][P:37](=[O:38])([OH:39])[O:40][CH2:41][CH:42]3[O:43][CH:45]([n:48]4[cH:50][n:53][c:55]5[c:51]4[n:54][cH:59][n:61][c:60]5[NH2:62])[CH:46]([O:49][P:52](=[O:56])([OH:57])[OH:58])[CH:44]3[OH:47])[CH:28]([OH:31])[CH:26]2[OH:29])[CH:18]=[CH:19][CH2:20]1',

    ]    
    proteins = [pg, pg]
    targets = torch.tensor([[1.0], [0.0]])
    dataset = ClipDataset(reactions, proteins, targets)
    batch = clip_collate([dataset[i] for i in range(len(dataset))])
    model = EnzymeReactionCLIP(cfg.model)
    out = model(batch)
    print(out)
    loss = model.training_step(batch, 0)
    print(loss)

    # tic = perf_counter()
    # print(dataset[0])
    # toc = perf_counter()
    # print(f"data retrieval time: {toc - tic:.4f} s")

    # tic = perf_counter()
    # print(dataset[0])
    # toc = perf_counter()
    # print(f"data retrieval time: {toc - tic:.4f} s")