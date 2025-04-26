import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, loss_factory
import numpy as np

EPS = 10e-05


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int):
    ds = datasets.xBDDataset(cfg, run_type, disable_augmentations=True)

    class_weights = loss_factory.loss_class_weights(cfg.TRAINER.LOSS.CLASS_WEIGHTS, ds.get_class_counts())
    criterion = loss_factory.get_criterion(cfg)

    net.to(device)
    net.eval()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE // 2, num_workers=0, shuffle=False,
                                       drop_last=False)

    loss_vals = []

    for step, item in enumerate(dataloader):
        x = item['img'].to(device)
        y = item['msk'].to(device)

        with torch.no_grad():
            if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
                x_cond = item['cond_id'].to(device)
                logits = net(x, x_cond)
            else:
                logits = net(x)

        loss = torch.Tensor([0]).cuda()
        for c in range(logits.size(1)):
            loss = loss + criterion(logits[:, c], y[:, c]) * class_weights[c]

        loss_vals.append(loss.item())

    loss_mean = np.mean(loss_vals)
    wandb.log({
        "val_loss_epoch": loss_mean,
        'step': step, 'epoch': epoch,
    })
    return -loss_mean


def model_evaluation_changeos(net, cfg, device, run_type: str, epoch: float, step: int):
    ds = datasets.xBDDataset(cfg, run_type, disable_augmentations=True)

    net.to(device)
    net.eval()

    dataloader = torch_data.DataLoader(ds, batch_size=max(1, cfg.TRAINER.BATCH_SIZE // 2), num_workers=0, shuffle=False,
                                       drop_last=False)

    loss_vals = []

    for step, item in enumerate(dataloader):
        x = item['img'].to(device)
        y = item['msk'].to(device)

        with torch.no_grad():
            if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
                x_cond = item['cond_id'].to(device)
                logits = net(x, x_cond)
            else:
                logits = net(x)

        loss_loc = loss_factory.loc_loss_changeos(logits[:, 0], y[:, 0].long())

        y_dmg = torch.zeros_like(y[:, 0]).long()
        for c in range(1, 5):
            y_dmg[y[:, c] == 1] = c
        loss_dmg = torch.Tensor([0]).to(device)
        if y_dmg.nonzero().nelement() > 0:
            loss_dmg = loss_factory.damage_loss_changeos(logits[:, 1:], y_dmg.long(),
                                                         ignore_index=cfg.MODEL.CHANGEOS.IGNORE_INDEX,
                                                         weighted=cfg.MODEL.CHANGEOS.WEIGHTED_LOSS)
        loss = loss_loc + loss_dmg
        loss_vals.append(loss.item())

    loss_mean = np.mean(loss_vals)
    wandb.log({
        "val_loss_epoch": loss_mean,
        'step': step, 'epoch': epoch,
    })
    return -loss_mean


def model_evaluation_damformer(net, cfg, device, run_type: str, epoch: float, step: int):
    ds = datasets.xBDDataset(cfg, run_type, disable_augmentations=True)

    net.to(device)
    net.eval()

    batch_size = max(1, cfg.TRAINER.BATCH_SIZE // 2)
    dataloader = torch_data.DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

    loss_vals = []

    for step, item in enumerate(dataloader):
        x = item['img'].to(device)
        y = item['msk'].to(device)

        with torch.no_grad():
            if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
                x_cond = item['cond_id'].to(device)
                logits = net(x, x_cond)
            else:
                logits = net(x)

        loss_loc = loss_factory.loc_loss_damformer(logits[:, 0], y[:, 0].long())

        y_dmg = torch.full_like(y[:, 0], fill_value=cfg.MODEL.CHANGEOS.IGNORE_INDEX).long()
        for c in range(1, 5):
            y_dmg[y[:, c] == 1] = c - 1
        loss_dmg = torch.Tensor([0]).to(device)
        if torch.sum(torch.ne(y_dmg, cfg.MODEL.CHANGEOS.IGNORE_INDEX)) > 0:
            loss_dmg = loss_factory.damage_loss_damformer(logits[:, 1:], y_dmg.long(),
                                                          ignore_index=cfg.MODEL.CHANGEOS.IGNORE_INDEX)
        loss = loss_loc + loss_dmg
        loss_vals.append(loss.item())

    loss_mean = np.mean(loss_vals)
    wandb.log({
        "val_loss_epoch": loss_mean,
        'step': step, 'epoch': epoch,
    })
    return -loss_mean


