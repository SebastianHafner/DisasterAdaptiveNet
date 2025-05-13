import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, losses
import numpy as np

EPS = 10e-05


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int):
    ds = datasets.xBDDataset(cfg, run_type, disable_augmentations=True)

    class_weights = losses.loss_class_weights(cfg.TRAINER.LOSS.CLASS_WEIGHTS, ds.get_class_counts())
    criterion = losses.ComboLoss(weights=cfg.TRAINER.LOSS.WEIGHTS)

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


