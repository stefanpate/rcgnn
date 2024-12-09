from torchmetrics import functional as F
from torch import Tensor
from chemprop.nn.metrics import (
    Metric,
    MetricRegistry,
    ThresholdedMixin,
    BinaryF1Metric,
    BinaryAccuracyMetric,
    BinaryAUPRCMetric,
    BinaryAUROCMetric
)

@MetricRegistry.register("binary_precision")
class BinaryPrecision(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.precision(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
        )

@MetricRegistry.register("binary_recall")
class BinaryRecall(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.recall(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
        )
    
@MetricRegistry.register("mcc")
class BinaryMCC(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.matthews_corrcoef(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
        )