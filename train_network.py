import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np
import random

from utils import datasets, losses, evaluation, experiment_manager, parsers, schedulers, measurers, metrics, models
from utils.experiment_manager import CfgNode


def run_training(cfg: CfgNode):
    net = models.create_network(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LEARNING_RATE, weight_decay=0.01)
    scheduler = schedulers.get_scheduler(cfg, optimizer)
    criterion = losses.ComboLoss(weights=cfg.TRAINER.LOSS.WEIGHTS)

    m_total, m_loc, m_dmg = measurers.AverageMeter(), measurers.AverageMeter(), measurers.AverageMeter()
    m_dice = measurers.AverageMeter()
    measurers_list = [m_total, m_loc, m_dmg, m_dice]

    # reset the generators
    dataset = datasets.xBDDataset(cfg, run_type='train')
    print(dataset)
    class_weights = losses.loss_class_weights(cfg.TRAINER.LOSS.CLASS_WEIGHTS, dataset.get_class_counts())

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    best_val_loss = -10e6
    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LEARNING_RATE,
            'epoch': epoch
        })
        start = timeit.default_timer()
        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x, msk = batch['img'].to(device), batch['msk'].to(device)

            if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
                x_cond = batch['cond_id'].to(device)
                logits = net(x, x_cond)
            else:
                logits = net(x)

            loss_loc = criterion(logits[:, 0], msk[:, 0].long()) * class_weights[0]

            loss_dmg = torch.Tensor([0]).cuda()
            for c in range(1, logits.size(1)):
                loss_dmg = loss_dmg + criterion(logits[:, c], msk[:, c].long()) * class_weights[c]
            loss = loss_loc + loss_dmg

            with torch.no_grad():
                y_hat = torch.sigmoid(logits[:, 0])
                dice_sc = 1 - metrics.dice_round(y_hat, msk[:, 0])

            m_loc.update(loss_loc.item(), x.size(0))
            m_dmg.update(loss_dmg.item(), x.size(0))
            m_total.update(loss.item(), x.size(0))
            m_dice.update(dice_sc, x.size(0))

            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                time = timeit.default_timer() - start
                wandb.log({
                    'train_loss_loc': m_loc.avg,
                    'train_loss_dmg': m_dmg.avg,
                    'train_loss': m_total.avg,
                    'train_dice': m_dice.avg,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                for measurer in measurers_list:
                    measurer.reset()
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if scheduler is not None:
            scheduler.step()
        # evaluation at the end of an epoch
        val_loss = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)
        if val_loss > best_val_loss:
            models.save_checkpoint(net, optimizer, epoch_float, cfg)
            best_val_loss = val_loss
            wandb.log({
                "best_val_loss": best_val_loss,
                'step': global_step,
                'epoch': epoch_float,
            })


if __name__ == '__main__':
    args = parsers.argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    print(cfg.NAME)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if cfg.RANDOM_SEED:
        random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        project='disasteradaptivenet',
        tags=['building localization', 'damage detection', 'xBD', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
