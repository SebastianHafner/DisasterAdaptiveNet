import torch

EPS = 10e-05
eps = 1e-6


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp + EPS)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn + EPS)


def rates(tp: int, fp: int, fn: int, tn: int) -> tuple:
    false_pos_rate = fp / (fp + tn + EPS)
    false_neg_rate = fn / (fn + tp + EPS)
    return false_pos_rate, false_neg_rate


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * p * r) / (p + r + EPS)


def iou(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fp + fn + EPS)


def oa(tp: int, fp: int, fn: int, tn: int) -> float:
    return (tp + tn) / (tp + tn + fp + fn + EPS)


def iou_tensors(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    tp = torch.sum(y & y_hat).float()
    fp = torch.sum(y_hat & ~y).float()
    fn = torch.sum(~y_hat & y).float()
    return tp / (tp + fp + fn + EPS)


def dice_round(outputs: torch.Tensor, targets: torch.tensor) -> torch.Tensor:
    """Convert predictions to float then calculate the dice loss.
    """
    outputs = outputs.float()
    return soft_dice_loss(outputs, targets)


def soft_dice_loss(outputs: torch.Tensor, targets: torch.Tensor, per_image: bool = False) -> torch.Tensor:
    """Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.

        Args:
            outputs: a tensor of shape [batch_size, num_classes, *spatial_dimensions]
            targets: a tensor of shape [batch_size, num_classes, *spatial_dimensions]
            per_image: if True, compute the dice loss per image instead of per batch

        Returns:
            dice_loss: the dice loss.
    """

    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss