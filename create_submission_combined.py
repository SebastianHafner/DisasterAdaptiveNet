import timeit
from tqdm.auto import tqdm
from typing import Tuple
import torch
import random
import cv2
from skimage import measure
from pathlib import Path
from utils import parsers, experiment_manager, datasets, xview2_metrics
from models import model_factory

from skimage.morphology import square, dilation

import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def object_based_infer(loc: np.ndarray, dam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # original input dimensions (1, 1, H, W) (1, 5, H, W)
    # loc = (pre_logit > 0.).cpu().squeeze(1).numpy()
    # dam = post_logit.argmax(dim=1).cpu().squeeze(1).numpy()
    loc, dam = loc[None], dam[None]
    refined_dam = np.zeros_like(dam)
    for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
        refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

    return loc.squeeze(), refined_dam.squeeze()


def _object_vote(loc: np.ndarray, dam: np.ndarray) -> np.ndarray:
    damage_cls_list = [1, 2, 3, 4]
    local_mask = loc
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, [8., 38., 25., 11.])]
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()
    return new_dam


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parsers.prediction_argument_parser().parse_known_args()[0]
    disable_cond = parsers.str2bool(args.disable_cond)
    cfg = experiment_manager.setup_cfg(args)

    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_factory.load_checkpoint(cfg, device)
    net.eval()

    ds = datasets.xBDDataset(cfg, 'test', disable_augmentations=True)

    sub_folder_name = f'submission_{cfg.NAME}_discond' if disable_cond else f'submission_{cfg.NAME}'
    sub_folder = Path(cfg.PATHS.OUTPUT) / 'predictions' / cfg.NAME / sub_folder_name
    sub_folder.mkdir(exist_ok=True, parents=True)

    for index in tqdm(range(len(ds))):
        item = ds.__getitem__(index)
        event, patch_id = item['event'], item['patch_id']
        torch.cuda.empty_cache()

        disaster_lookups = None
        if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
            if disable_cond:
                disaster_lookups = None
            elif parsers.str2bool(args.random_cond):
                disaster_lookups = torch.randint(len(cfg.DATASET.CONDITIONING_KEY), (1,)).to(device)
            else:
                disaster_lookups = item['cond_id'].to(device)

        x = item['img'].to(device)
        inp = [x]
        if cfg.INFERENCE.USE_ALL_FLIPS:
            inp.append(torch.flip(x, dims=[1]))
            inp.append(torch.flip(x, dims=[2]))
            inp.append(torch.flip(x, dims=[1, 2]))
            if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION and not disable_cond:
                disaster_lookups = disaster_lookups.repeat(4)

        inp = torch.stack(inp)

        with torch.no_grad():
            logits = net(inp, disaster_lookups) if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION else net(inp)
        y_hat = torch.sigmoid(logits)
        y_hat = y_hat.cpu().detach()

        pred = [y_hat[0]]
        if cfg.INFERENCE.USE_ALL_FLIPS:
            pred.append(torch.flip(y_hat[1], dims=[1]))
            pred.append(torch.flip(y_hat[2], dims=[2]))
            pred.append(torch.flip(y_hat[3], dims=[1, 2]))

        pred_full = torch.stack(pred).numpy()
        preds = pred_full.mean(axis=0).transpose(1, 2, 0)
        loc_preds = preds[..., 0]

        msk_dmg = preds[..., 1:].argmax(axis=2) + 1
        _thr = [0.38, 0.13, 0.14]
        if cfg.INFERENCE.USE_TRICKS:
            msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | (
                    (loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')
        else:
            msk_loc = (loc_preds > _thr[0]).astype('uint8')

        if cfg.INFERENCE.OBJECT_BASED:
            msk_loc, msk_dmg = object_based_infer(msk_loc, msk_dmg)
        else:
            # why is this one needed -> dmg is assessed separately to localization
            msk_dmg = msk_dmg * msk_loc

        _msk = (msk_dmg == 2)
        if cfg.INFERENCE.USE_TRICKS and (_msk.sum() > 0):
            _msk = dilation(_msk, square(5))
            msk_dmg[_msk & msk_dmg == 1] = 2

        msk_dmg = msk_dmg.astype('uint8')

        loc_file = sub_folder / f'{event}_{patch_id}_localization_disaster_prediction.png'
        cv2.imwrite(str(loc_file), msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cls_file = sub_folder / f'{event}_{patch_id}_damage_disaster_prediction.png'
        cv2.imwrite(str(cls_file), msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    xview2_metrics.XviewMetrics.get_score(cfg)
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))


