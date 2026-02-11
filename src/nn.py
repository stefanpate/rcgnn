'''
Neural net modules
'''

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from chemprop.nn import Aggregation, Predictor, MeanAggregation, BinaryClassificationFFN # Don't remove unused objects
from chemprop.nn.ffn import MLP
from chemprop.nn.hparams import HasHParams
from chemprop.nn.loss import BCELoss
from chemprop.nn.metrics import BinaryAUROCMetric
from chemprop.nn.message_passing.base import _MessagePassingBase, BondMessagePassing
from chemprop.data import BatchMolGraph
from lightning.pytorch.core.mixins import HyperparametersMixin
from transformers import BertModel


class WeightedBCELoss(BCELoss):
    def __init__(self, pos_weight, task_weights = 1):
        super().__init__(task_weights)
        self.pos_weight = pos_weight

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none", pos_weight=self.pos_weight)

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
    output_transform = torch.nn.Identity()
    task_weights = None
    _T_default_criterion = BCELoss
    _T_default_metric = BinaryAUROCMetric

    def __init__(self, input_dim:int, criterion = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.n_targets * self.n_tasks
        self.criterion = criterion or self._T_default_criterion()
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

class BertRxnEncoder(Module):
    def __init__(self, base_model: BertModel, d_rxn: int, d_h: int):
        '''
        d_rxn: dimension of reaction embedding output by BERT
        d_h: dimension of hidden representation to feed into predictor
        '''
        super().__init__()
        self.model = base_model
        self.model.eval()
        self.linear_rxn_layer = torch.nn.Linear(in_features=d_rxn, out_features=d_h)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )['last_hidden_state'][:, 0, :] # [CLS] token embeddings in position 0

        rxn_embeddings = self.linear_rxn_layer(bert_output)
        return rxn_embeddings