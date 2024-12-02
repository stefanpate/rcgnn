from torchmetrics import functional as F
from torch import Tensor
from chemprop.nn.metrics import (
    Metric,
    MetricRegistry,
    ThresholdedMixin,
    BinaryF1Metric,
    BinaryMCCMetric,
    BinaryAccuracyMetric,
    BinaryAUPRCMetric,
    BinaryAUROCMetric
)

@MetricRegistry.register("f1")
class BinaryF1Metric(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.f1_score(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
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