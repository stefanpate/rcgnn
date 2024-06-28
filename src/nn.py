'''
Neural net modules
'''

from chemprop.nn import Aggregation
import torch
from torch import Tensor
from chemprop.nn import Aggregation, Predictor
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
    pass