'''
Neural net modules
'''

import torch
from torch import Tensor
from torch.nn import Module
from chemprop.nn import Aggregation, Predictor
from chemprop.nn.ffn import MLP
from chemprop.nn.hparams import HasHParams
from chemprop.nn.loss import BCELoss
from chemprop.nn.metrics import BinaryAUROCMetric
from chemprop.nn.message_passing.base import _MessagePassingBase, BondMessagePassing
from chemprop.data import BatchMolGraph
from lightning.pytorch.core.mixins import HyperparametersMixin

class LastAggregation(Aggregation):
    '''
    Takes hidden representation of last node
    which RCVNFeaturizer has constructed as virtual node
    linked to the reaction center atoms.

    Args
    ---
    H:Tensor - (# nodes in batch x d_h) node hidden representation matrix
    batch:Tensor - array of indices saying what sample each node in BatchMolGraph
        came from
    '''
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        batch_size = batch.max().int() + 1
        last_idxs = torch.tensor([torch.argwhere(batch == i)[-1,0] for i in range(batch_size)])
        return H[last_idxs, :]
    
class AttentionAggregation(Aggregation):
    '''
    Takes weighted average of each layer of hidden representation processing
    w/ weights as function of sequence of l layer-representations

    Args
    ----
    H:Tensor - (batch_size x d_h x MP depth) w/ l-layer hidden reps of 
        virtual node at each slice along dim=2

    Returns
    ---------
    Tensor - (batch_size x d_h) w/ attention-weighted sum of l virtual node
        hidden representations
    '''
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            hidden_dim: int = 300,
            n_layers: int = 1,
            dropout: float = 0.0,
            activation: str = "relu",

            ):
        super().__init__(None)

        self.ffn = MLP.build(
            input_dim,
            output_dim,
            hidden_dim,
            n_layers,
            dropout,
            activation
        )

    def forward(self, H:Tensor, batch:Tensor):
        return self._forward(H)

    def _forward(self, H:Tensor) -> Tensor:
       H = torch.transpose(H, 1, 2)
       S = self.ffn(H)
       A = torch.softmax(S, dim=1)
       H = torch.mul(H, A).sum(dim=1)
       return H
   
class DotSig(Predictor, HyperparametersMixin):
    '''
    Takes sigmoid(a . b) e.g., a is protein embed, b is reaction embed.
    Assumes d_prot == d_rxn

    Args
    ----
    input_dim:int - dimension of concatenated embeddings (cat(prot, rxn)) 
    '''
    n_targets = 1
    n_tasks = 1
    output_transform = None
    task_weights = None
    _T_default_criterion = BCELoss
    _T_default_metric = BinaryAUROCMetric

    def __init__(self, input_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.n_targets * self.n_tasks
        self.criterion = self._T_default_criterion()
        self.d_h  = self.input_dim // 2 # Dimension of one embedding

    def forward(self, H):
        R = H[:, :self.d_h]
        P = H[:, self.d_h:]
        logits = torch.mul(R, P).sum(dim=1).reshape(-1,1)
        return logits.sigmoid()
    
    def train_step(self, H):
        R = H[:, :self.d_h]
        P = H[:, self.d_h:]
        logits = torch.mul(R, P).sum(dim=1).reshape(-1,1)
        return logits
    
    def encode(self, H, i):
        return H

class LinDimRed(Module, HasHParams, HyperparametersMixin):
    def __init__(self, d_in:int, d_out:int) -> None:
        super().__init__()
        self.linear_layer = torch.nn.Linear(
            in_features=d_in,
            out_features=d_out
        )
        self.save_hyperparameters(ignore=['linear_layer'])
        self.hparams["cls"] = self.__class__


    def forward(self, X):
        return self.linear_layer(X)
    

class _MessagePassingDictBase(_MessagePassingBase):
    def extract_virtual_node(self, H):
        pass

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            a batch of :class:`BatchMolGraph`s to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
            molecule in the batch, depending on whether additional atom descriptors were provided
        """
        batch_size = bmg.batch.max().int() + 1
        last_idxs = torch.tensor([torch.argwhere(bmg.batch == i)[-1,0] for i in range(batch_size)])
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        VN = torch.zeros(size=(batch_size, self.output_dim, self.depth))
        H = self.tau(H_0)
        for l in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

            index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
            M_v = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
                0, index_torch, H, reduce="sum", include_self=False
            )
            H_v = self.finalize(M_v, bmg.V, V_d)
            VN[:, :, l] = H_v[last_idxs, :]

        return VN

class BondMessagePassingDict(_MessagePassingDictBase, BondMessagePassing):
    pass