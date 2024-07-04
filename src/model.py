from chemprop.models import MPNN
import torch
from torch import Tensor
from chemprop.data import BatchMolGraph

class MPNNDimRed(MPNN):
    def __init__(self, reduce_X_d, **kwargs):
        super().__init__(**kwargs)

        self.reduce_X_d = reduce_X_d # Linear layer to dim reduce protein embeds

        # self.save_hyperparameters(ignore=["message_passing", "agg", "predictor", "reduce_X_d"])
        # self.hparams.update(
        #     {
        #         "message_passing": self.message_passing.hparams,
        #         "agg": self.agg.hparams,
        #         "predictor": self.predictor.hparams,
        #     }
        # )

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.reduce_X_d(X_d)), 1)