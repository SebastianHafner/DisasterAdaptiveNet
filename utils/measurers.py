import abc
import torch
from utils.experiment_manager import CfgNode
import torchmetrics
from typing import Sequence


def get_measurer(cfg: CfgNode):
    return BaselineMeasurer()


class AbstractMeasurer(abc.ABC):
    def __init__(self, n_classes: int, threshold: float):

        self.n_classes = n_classes
        self.threshold = threshold

        self.metrics = []
        self.class_names = ['loc', 'nodmg', 'midmg', 'madmg', 'destr']

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError("add_sample method must be implemented in the subclass.")

    def xview2_score(self) -> torch.Tensor:
        raise NotImplementedError("add_sample method must be implemented in the subclass.")

    @staticmethod
    def harmonic_mean(tensor: torch.Tensor) -> torch.Tensor:
        reciprocal_sum = torch.sum(1 / tensor)
        if reciprocal_sum == 0:
            raise ValueError("Cannot calculate harmonic mean with zero in the denominator.")
        harmonic_mean_result = len(tensor) / reciprocal_sum
        return harmonic_mean_result


class BaselineMeasurer(AbstractMeasurer):
    def __init__(self):
        super().__init__(n_classes=5, threshold=0.38)
        for _ in range(self.n_classes):
            # self.metrics.append(torchmetrics.F1Score(num_labels=5, average='micro', threshold=0.5, task='multilabel'))
            self.metrics.append(torchmetrics.F1Score(average='none', task='binary'))

    def add_sample(self, logits: torch.Tensor, y: torch.Tensor):
        y_sigm = torch.sigmoid(logits)
        loc_pred = y_sigm[:, 0, ...]
        # loc mask in {0,1}
        loc_msk = (loc_pred > self.threshold)
        # dmg mask in {0,1,2,3,4}
        dmg_msk = y_sigm[:, 1:, ...].argmax(dim=1) + 1  # get 4-class ids per pixel
        dmg_msk = dmg_msk * loc_msk
        # hot encode dmg_msk shape (batch_size, 5, h, w)
        hot_dmg_msk = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device)
        for i in range(5):
            hot_dmg_msk[:, i, ...] = dmg_msk == i
        hot_dmg_msk[:, 0, ...] = loc_msk

        for i, metric in enumerate(self.metrics):
            pred = hot_dmg_msk[:, i, ...]
            target = y[:, i, ...]
            metric.update(pred, target)

    def xview2_score(self) -> torch.Tensor:
        dmg_scores = torch.tensor([metric.compute().item() for metric in self.metrics[1:]])
        f1_dmg = self.harmonic_mean(dmg_scores)
        f1_loc = self.metrics[0].compute()
        return 0.3 * f1_loc + 0.7 * f1_dmg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
