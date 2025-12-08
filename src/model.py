
from typing import Iterable
import torch
from torch.optim import Adam
from torch import Tensor
import lightning as L
from chemprop.schedulers import NoamLR
from chemprop.models import MPNN
from chemprop.data import BatchMolGraph
from chemprop.nn import MessagePassing, Aggregation, Predictor, LossFunction, Metric
from chemprop.nn.ffn import MLP


class MPNNDimRed(MPNN):
    def __init__(
            self,
            message_passing: MessagePassing,
            agg: Aggregation,
            predictor: Predictor,
            reduce_X_d,
            batch_norm: bool = True,
            metrics: Iterable = [],
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
            metrics=metrics,
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
    
class TwoChannelLinear(L.LightningModule):
    def __init__(
            self,
            d_rxn: int,
            d_prot: int,
            d_h: int,
            predictor: Predictor,
            metrics: Iterable[Metric] | None = None,
            warmup_epochs: int = 2,
            init_lr: float = 1e-4,
            max_lr: float = 1e-3,
            final_lr: float = 1e-4,
            ):
        super().__init__()
        self.reaction_encoder = torch.nn.Linear(in_features=d_rxn, out_features=d_h)
        self.protein_encoder = torch.nn.Linear(in_features=d_prot, out_features=d_h)
        self.predictor = predictor
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self.metrics = (
            [*metrics, self.criterion]
            if metrics
            else [self.predictor._T_default_metric(), self.criterion]
        )

    @property
    def criterion(self) -> LossFunction:
        return self.predictor.criterion

    def training_step(self, batch, batch_idx):
        R, Y, P, weights, gt_mask, lt_mask = batch # reaction embeds, targets, protein embeds
        R = self.reaction_encoder(R)
        P = self.protein_encoder(P)
        Y_hat = self.predictor(torch.cat((R, P), dim=1))
        mask = Y_hat.isfinite()
        loss = self.criterion(Y_hat, Y, mask, weights, gt_mask, lt_mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))
        self.log("val_loss", losses[0], batch_size=len(batch[0]), prog_bar=True)
    
    def forward(self, R, P):
        R = self.reaction_encoder(R)
        P = self.protein_encoder(P)
        Y_hat = self.predictor(torch.cat((R, P), dim=1))
        return Y_hat
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        R, _, P, *_ = batch
        return self(R, P)
    
    def configure_optimizers(self):
        opt = Adam(self.parameters(), self.init_lr)

        lr_sched = NoamLR(
            opt,
            self.warmup_epochs,
            self.trainer.max_epochs,
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            self.init_lr,
            self.max_lr,
            self.final_lr,
        )
        lr_sched_config = {
            "scheduler": lr_sched,
            "interval": "step" if isinstance(lr_sched, NoamLR) else "batch",
        }

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}
    
    def _evaluate_batch(self, batch) -> list[Tensor]:
        R, Y, P, weights, gt_mask, lt_mask = batch

        mask = Y.isfinite()
        Y = Y.nan_to_num(nan=0.0)
        preds = self(R, P)

        return [
            metric(preds, Y, mask, None, lt_mask, gt_mask) for metric in self.metrics[:-1]
        ]

class TwoChannelFFN(TwoChannelLinear):
    def __init__(
            self,
            d_rxn: int,
            d_prot: int,
            d_h: int,
            encoder_depth: int,
            predictor: Predictor,
            metrics: Iterable[Metric] | None = None,
            warmup_epochs: int = 2,
            init_lr: float = 1e-4,
            max_lr: float = 1e-3,
            final_lr: float = 1e-4,
            ) -> None:
        super().__init__(
            d_rxn,
            d_prot,
            d_h,
            predictor,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
        )
        self.reaction_encoder = MLP.build(
            input_dim=d_rxn,
            output_dim=d_h,
            hidden_dim=d_h,
            n_layers=encoder_depth,
            )
        
