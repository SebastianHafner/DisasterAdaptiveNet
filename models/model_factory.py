import torch
import torch.nn as nn
from pathlib import Path
from utils.experiment_manager import CfgNode
from models import unet, baseline, changeos, siamdiff, damformer, dualhrnet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_network(cfg) -> torch.nn.Module:
    if cfg.MODEL.TYPE == 'unet':
        net = unet.UNet(cfg)
    elif cfg.MODEL.TYPE == 'baseline':
        net = baseline.StrongBaselineNet(cfg)
    elif cfg.MODEL.TYPE == 'disasterfilmnet':
        net = baseline.DisasterAdaptiveNet(cfg)
    elif cfg.MODEL.TYPE == 'disastertypenet':
        net = baseline.DisasterTypeNet(cfg)
    elif cfg.MODEL.TYPE == 'changeoscond':
        net = changeos.ChangeOSConditioned(cfg)
    elif cfg.MODEL.TYPE == 'changeos':
        net = changeos.ChangeOS(cfg)
    elif cfg.MODEL.TYPE == 'changeoscondv2':
        net = changeos.ChangeOSConditionedV2(cfg)
    elif cfg.MODEL.TYPE == 'disasterspecificnets':
        net = baseline.DisasterSpecificNets(cfg)
    elif cfg.MODEL.TYPE == 'siamdiff':
        net = siamdiff.SiamUnet_diff(cfg)
    elif cfg.MODEL.TYPE == 'damformer':
        net = damformer.DamFormer(cfg)
    elif cfg.MODEL.TYPE == 'baseline_swin':
        net = baseline.StrongBaselineNetSwin(cfg)
    elif cfg.MODEL.TYPE == 'baseline_convnext':
        net = baseline.StrongBaselineNetConvNeXt(cfg)
    elif cfg.MODEL.TYPE == 'dualhrnet':
        net = dualhrnet.get_model(cfg)
    else:
        raise Exception(f'Unknown network ({cfg.MODEL.TYPE}).')
    return nn.DataParallel(net)


def save_checkpoint(network, optimizer, epoch: int, cfg: CfgNode, save_file: Path = None):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt' if save_file is None else save_file
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(cfg: CfgNode, device: torch.device, net_file: Path = None) -> torch.nn.Module:
    net = create_network(cfg)
    net.to(device)

    if net_file is None:
        net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'

    checkpoint = torch.load(net_file, map_location=device)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    net.load_state_dict(checkpoint['network'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    return net

