from typing import Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

from legacy.losses import dice_round
from legacy.utils import get_class_weights, AverageMeter


def train_epoch(current_epoch: int, seg_loss: nn.Module, model: nn.Module, optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler, train_data_loader: DataLoader,
                class_weights_: str = "no1") -> None:
    """
    Trains model for one epoch.

    Args:
        seg_loss : segmentation loss
        model : model to train
        class_weights_ : Choose class weights to weight the classes separately in computing the loss.
            'Equal' assigns equal weights,
            'no1' uses the weights in the no.1 solution,
            'distr' uses the normalized inverse of the class distribution in the training dataset
    """

    losses = AverageMeter()
    dices = AverageMeter()

    class_weights = get_class_weights(class_weights_)

    model.train()
    for i, sample in enumerate(train_data_loader):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)

        out = model(imgs)

        seg_loss_val = torch.Tensor([0]).cuda()
        for c in range(out.size(1)):
            seg_loss_val = seg_loss_val + seg_loss(out[:, c, ...], msks[:, c, ...]) * class_weights[c]

        # Catch error that happens if focal and dice loss have weights 0
        seg_loss_val = torch.Tensor([seg_loss_val]).cuda() if (type(seg_loss_val) == float) else seg_loss_val
        loss = seg_loss_val

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(seg_loss_val.item(), imgs.size(0))
        dices.update(dice_sc, imgs.size(0))

        wandb.log({"train_loss_step": losses.val, "train_dice_step": dices.val})

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)
    wandb.log({"train_loss_epoch": losses.avg, "train_dice_epoch": dices.avg})
    print(f"epoch: {current_epoch}; lr {scheduler.get_lr()[-1]:.7f}; Loss {losses.avg:.4f}; Dice {dices.avg:.4f}")


def validate(model: nn.Module, data_loader: DataLoader, args: Dict = None, seg_loss: nn.Module = None) -> float:
    """Validate the classification model using the loss or the competition metric.

    Args:
        model: model to validate
        validate_on_score : Whether we validate on the competition score (mix of F1 and dice) or the loss
        args : arguments used to call the training script (see train34.sh)
        seg_loss : segmentation loss

    Returns:
        float: validation score (-loss or competition metric)

    """
    loss_vals = []

    _thr = 0.3

    for i, sample in enumerate(data_loader):
        msks = sample["msk"].numpy()
        imgs = sample["img"].cuda(non_blocking=True)
        with torch.no_grad():
            out = model(imgs)

        # Validate on loss function, don't filter by loc predictions of loc model
        class_weights = get_class_weights(args.class_weights)

        msks = torch.Tensor(msks).cuda()
        seg_loss_val = torch.Tensor([0]).cuda()
        for c in range(out.size(1)):
            seg_loss_val = seg_loss_val + seg_loss(out[:, c, ...], msks[:, c, ...]) * class_weights[c]

        loss = seg_loss_val
        loss_vals.append(loss.item())

    loss_mean = np.mean(loss_vals)
    wandb.log({"val_loss_epoch": loss_mean})
    return -loss_mean


def evaluate_val(data_val: DataLoader, best_score: float, model: nn.Module, snapshot_name: str, current_epoch: int,
                 models_folder: Path, args: Dict, seg_loss: nn.Module) -> float:
    """Evaluate the classification model on the validation set for one epoch and saves best model so far.

        Args:
            data_val : validation data loader
            best_score : initial value for best score on validation set (usually very small)
            model : model to evaluate
            snapshot_name : name of the snapshot from the last best model
            current_epoch : current epoch
            models_folder : folder to save best model
            args : arguments used to call the training script (see train34.sh)
            seg_loss : segmentation loss
        Returns:
            best_score : best score on validation set so far (-loss or competition metric)
    """

    model = model.eval()

    d = validate(model, data_loader=data_val, args=args, seg_loss=seg_loss)
    if d > best_score:
        save_file = models_folder / f'{snapshot_name}_best.pt'
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, str(save_file))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score
