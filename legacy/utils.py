import numpy as np
import cv2
import argparse
import gc
from distutils import util
from pathlib import Path

import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class_weights", type=str, default="no1",
                        choices=["no1", "equal", "distr", "distr_no_overlap"],
                        help="Choose class weights to weigh the classes separately in computing the loss. "
                             "'Equal' assigns equal weights, 'no1' uses the weights in the no.1 solution, "
                             "'distr' uses the normalized inverse of the class distribution in the training dataset.")
    parser.add_argument("--dir_prefix", type=str,
                        help="Prefix to use when creating the different subfolders, e.g. for weights, predictions etc.")
    parser.add_argument("--debug", type=lambda x: bool(util.strtobool(x)), default=False,
                        help="Only one epoch per subscript to make sure that everything is working as intended.")
    parser.add_argument("--wandb_group", type=str, default="",
                        help="Group argument to be set in wandb. "
                             "Used to identify all partial runs within one experiment.")
    return parser


def load_snapshot(model, snap_to_load, models_folder: Path):
    print("=> loading checkpoint '{}'".format(snap_to_load))
    file = models_folder / f'{snap_to_load}.pt'
    state_dict = torch.load(str(file), map_location='cpu')["state_dict"]
    if 'module.' in list(state_dict.keys())[0] and 'module.' not in list(model.state_dict().keys())[0]:
        loaded_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        print("modifying keys so that they match the current model, normal if you're using loc weights for cls")
    else:
        loaded_dict = state_dict

    # loaded_dict = checkpoint['state_dict']
    sd = model.state_dict().copy()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
        else:
            print("skipping key: {}".format(k))
    model.load_state_dict(sd)

    del loaded_dict
    del sd
    gc.collect()
    torch.cuda.empty_cache()


def get_class_weights(cw):
    if cw == "no1":
        class_weights = [0.05, 0.2, 0.8, 0.7, 0.4]
    elif cw == "distr":
        class_weights = [1 / 0.030342701529957234, 1 / 0.023289585196743044, 1 / 0.002574037714986485,
                         1 / 0.002682490082519425, 1 / 0.0017965885357082826]
        sum_of_weights = sum(class_weights)
        class_weights = [w / sum_of_weights for w in class_weights]
    elif cw == "distr_no_overlap":
        class_weights = [32.20398121025673, 41.516691841904844, 406.2242790072747, 319.5142994620793, 727.8449005124751]
        sum_of_weights = sum(class_weights)
        class_weights = [w / sum_of_weights for w in class_weights]
    elif cw == "equal":
        class_weights = [0.2] * 5
    else:
        raise ValueError(f"Not implemented for class weight choice: {cw}")

    return class_weights


def rotate_image(image, angle, scale, rot_pnt):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)  # INTER_NEAREST
    return result


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


def normalize_image(img):
    img = np.asarray(img, dtype='float32')
    img /= 127
    img -= 1
    return img


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def iou(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    union = np.logical_or(im1, im2)
    im_sum = union.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / im_sum