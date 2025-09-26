from clipzyme.utils.screening import process_mapped_reaction
from clipzyme.utils.loading import default_collate
from clipzyme.models.protmol import EnzymeReactionCLIP
from clipzyme.models.abstract import AbstractModel
from clipzyme.models.chemprop import DMPNNEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data
from torch_scatter import scatter
import copy
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def clip_collate(datapoints: list[dict[str, tuple[Data, Data] | torch.Tensor]]) -> dict[str, dict[str, Batch] | torch.Tensor]:
    rxn_batch, prot_batch, target_batch = [], [], []
    for elt in datapoints:
        rxns, prots, targets = elt['reaction'], elt['protein_embedding'], elt['target']
        rcts, pdts = rxns
        rxn_batch.append({'reactants': rcts, 'products': pdts})
        prot_batch.append(prots)
        target_batch.append(targets)
    
    rxn_batch = default_collate(rxn_batch)
    prot_batch = torch.stack(prot_batch)
    target_batch = torch.stack(target_batch)

    return {"reaction": rxn_batch, "protein_embedding": prot_batch, "target": target_batch}

class EnzymeReactionCLIP(LightningModule):
    '''
    Uses "pseudo transition state" representation of reactions and
    pre-trained protein embedding. Adapted from Mikhael, Chinn, & Barzilay 2024
    aka CLIPZyme. https://github.com/pgmikhael/clipzyme/
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reaction_clip_model_path = args.reaction_clip_model_path
        self.use_as_protein_encoder = getattr(args, "use_as_protein_encoder", False)
        self.use_as_mol_encoder = getattr(args, "use_as_mol_encoder", False) or getattr(
            args, "use_as_reaction_encoder", False
        )  # keep mol for backward compatibility

        if args.reaction_clip_model_path is not None:
            state_dict = torch.load(args.reaction_clip_model_path)
            state_dict_copy = {
                k.replace("model.", "", 1): v
                for k, v in state_dict["state_dict"].items()
            }
            args = state_dict["hyper_parameters"]["args"]

        wln_diff_args = copy.deepcopy(args)
        if args.model_name != "enzyme_reaction_clip_wldnv1":
            self.wln = DMPNNEncoder(args)  # WLN for mol representation
            wln_diff_args = copy.deepcopy(args)
            wln_diff_args.chemprop_edge_dim = args.chemprop_hidden_dim
            # wln_diff_args.chemprop_num_layers = 1 ## Sic clipzyme original code
            self.wln_diff = DMPNNEncoder(wln_diff_args)

            # mol: attention pool
            self.final_linear = nn.Linear(
                args.chemprop_hidden_dim, args.chemprop_hidden_dim, bias=False
            )
            self.attention_fc = nn.Linear(args.chemprop_hidden_dim, 1, bias=False)

        if self.reaction_clip_model_path is not None:
            self.load_state_dict(state_dict_copy)

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

    def dot_sig(self, R: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        logits = torch.mul(R, P).sum(dim=1).reshape(-1,1)
        return logits.sigmoid()

    def forward(self, batch) -> torch.Tensor:
        P = batch["protein_embedding"]
        P = P / P.norm(
            dim=1, keepdim=True
        )
        R = batch["reaction"]
        R = self.encode_reaction(R)
        R = R / R.norm(
            dim=1, keepdim=True
        )
        return self.dot_sig(R, P)
    
    def training_step(self, batch, batch_idx):
        Y_hat = self.forward(batch)
        Y = batch["target"]
        mask = Y_hat.isfinite()
        loss = F.binary_cross_entropy(Y_hat[mask], Y[mask])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        print( f"train_loss: {loss}" )
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.args.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.args.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            print(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.args.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.args.init_lr, self.args.max_lr, self.args.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer

    
class ClipDataset:
    def __init__(self, reactions: list[str], protein_embeddings: torch.Tensor, targets: torch.Tensor):
        if len(reactions) != len(protein_embeddings) or len(reactions) != len(targets):
            raise ValueError("Length of reactions, protein_embeddings, and targets must be the same.")
        
        self.reactions = reactions
        self.protein_embeddings = protein_embeddings
        self.targets = targets
    
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, idx) -> dict[str, tuple[Data, Data] | torch.Tensor]:
        rcts, pdts = process_mapped_reaction(self.reactions[idx])
        return {
            "reaction": (rcts, pdts),
            "protein_embedding": self.protein_embeddings[idx],
            "target": self.targets[idx],
        }
    
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

if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train_clipzyme")

    model = EnzymeReactionCLIP(cfg.model)
    print(model)
    reactions = [
        "[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][CH:6]=[O:7].[O:9]=[O:10].[OH2:8]>>[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][C:6](=[O:7])[OH:8].[OH:9][OH:10]",
        '[CH:1]([CH2:3][CH2:5][CH:6]([NH2:7])[C:8](=[O:9])[OH:10])=[O:4].[c:15]1([C:21]([NH2:22])=[O:23])[cH:16][n+:17]([CH:24]2[O:25][CH:27]([CH2:30][O:32][P:33](=[O:34])([OH:35])[O:36][P:37](=[O:38])([OH:39])[O:40][CH2:41][CH:42]3[O:43][CH:45]([n:48]4[cH:50][n:53][c:55]5[c:51]4[n:54][cH:59][n:61][c:60]5[NH2:62])[CH:46]([O:49][P:52](=[O:56])([OH:57])[OH:58])[CH:44]3[OH:47])[CH:28]([OH:31])[CH:26]2[OH:29])[cH:18][cH:19][cH:20]1.[OH:2][P:11](=[O:12])([OH:13])[OH:14]>>[C:1]([O:2][P:11](=[O:12])([OH:13])[OH:14])([CH2:3][CH2:5][CH:6]([NH2:7])[C:8](=[O:9])[OH:10])=[O:4].[C:15]1([C:21]([NH2:22])=[O:23])=[CH:16][N:17]([CH:24]2[O:25][CH:27]([CH2:30][O:32][P:33](=[O:34])([OH:35])[O:36][P:37](=[O:38])([OH:39])[O:40][CH2:41][CH:42]3[O:43][CH:45]([n:48]4[cH:50][n:53][c:55]5[c:51]4[n:54][cH:59][n:61][c:60]5[NH2:62])[CH:46]([O:49][P:52](=[O:56])([OH:57])[OH:58])[CH:44]3[OH:47])[CH:28]([OH:31])[CH:26]2[OH:29])[CH:18]=[CH:19][CH2:20]1',

    ]    
    prot_embeddings = torch.randn((2, 1280))
    targets = torch.tensor([[1.0], [0.0]])
    dataset = ClipDataset(reactions, prot_embeddings, targets)
    batch = clip_collate([dataset[i] for i in range(len(dataset))])
    model = EnzymeReactionCLIP(cfg.model)
    out = model(batch)
    print(out)
    loss = model.training_step(batch, 0)
    print(loss)