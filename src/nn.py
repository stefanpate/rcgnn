'''
Neural net modules
'''

from chemprop.nn import Aggregation
import torch
from torch import Tensor
from torch.nn import Module
from chemprop.nn import Aggregation, Predictor
from chemprop.nn.hparams import HasHParams
from chemprop.nn.loss import BCELoss
from chemprop.nn.metrics import BinaryAUROCMetric
from lightning.pytorch.core.mixins import HyperparametersMixin

class LastAggregation(Aggregation):
    '''
    Takes hidden representation of last node
    which RCVNFeaturizer has constructed as virtual node
    linked to the reaction center atoms
    '''
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        batch_size = batch.max().int() + 1
        last_idxs = torch.tensor([torch.argwhere(batch == i)[-1,0] for i in range(batch_size)])
        return H[last_idxs, :]
    
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