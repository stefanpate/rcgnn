from chemprop.models import MPNN
import torch
from torch import Tensor
from chemprop.data import BatchMolGraph
from chemprop.nn import MessagePassing, Aggregation, Predictor

class MPNNDimRed(MPNN):
    def __init__(
            self,
            message_passing: MessagePassing,
            agg: Aggregation,
            predictor: Predictor,
            reduce_X_d,
            batch_norm: bool = True,
            warmup_epochs: int = 2,
            init_lr: float = 1e-4,
            max_lr: float = 1e-3,
            final_lr: float = 1e-4,
            ):

        super().__init__(
            message_passing=message_passing,
            agg=agg,
            predictor=predictor,
            batch_norm=batch_norm,
            warmup_epochs=warmup_epochs,
            init_lr=init_lr,
            max_lr=max_lr,
            final_lr=final_lr,
            )
        
        self.reduce_X_d = reduce_X_d # Linear layer to dim reduce protein embeds
        self.hparams.update({"reduce_X_d": self.reduce_X_d.hparams})

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.reduce_X_d(X_d)), 1)