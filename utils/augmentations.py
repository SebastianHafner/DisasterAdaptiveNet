from torchvision import transforms
import random
import numpy as np
import cv2
from utils.experiment_manager import CfgNode


def compose_transformations(cfg: CfgNode, augs_enabled: bool = True):
    transformations = []
    if augs_enabled:
        if cfg.AUGMENTATION.FLIP:
            transformations.append(RandomFlip(p=0.5))
        if cfg.AUGMENTATION.ROTATE:
            transformations.append(RandomRotate(p=0.95))
        if cfg.AUGMENTATION.AFFINE:
            transformations.append(RandomAffine(p=0.4))
        if cfg.AUGMENTATION.SMART_CROP:
            transformations.append(
                RandomSmartCrop(p=0.8, input_shape=(cfg.AUGMENTATION.CROP_SIZE, cfg.AUGMENTATION.CROP_SIZE),
                                split=cfg.DATASET.SPLIT, dynamic_weights=cfg.AUGMENTATION.SMART_CROP_WEIGHTS)
            )
        if cfg.AUGMENTATION.SIMPLE_CROP:
            transformations.append(RandomSimpleCrop(p=0.8, input_shape=(cfg.AUGMENTATION.CROP_SIZE,
                                                                        cfg.AUGMENTATION.CROP_SIZE)))

    if cfg.DATALOADER.NORMALIZE_IMAGES:
        transformations.append(Normalize())
    if cfg.DATALOADER.DAMAGE_ONEHOTENCODING:
        transformations.append(DamageOneHotEncoding(4))  # TODO: get this from the config
    if cfg.DATALOADER.NUMPY2TORCH:
        pass
    return transforms.Compose(transformations)


class DamageOneHotEncoding(object):
    def __init__(self, n: int):
        self.n = n  # number of damage classes

    def __call__(self, args):
        img, msk = args
        msk_loc, msk_dmg = msk[:, :, 0], msk[:, :, 1]
        msk_dmg_hot = np.zeros((msk.shape[0], msk.shape[1], self.n), dtype=np.float32)
        for i in range(self.n):
            msk_dmg_hot[msk_dmg == i + 1, i] = 1
        if msk.shape[-1] == 2:
            msk = np.concatenate([msk_loc[..., None], msk_dmg_hot], axis=-1)
        else:
            assert msk.shape[-1] == 3
            msk_buildings = msk[:, :, 2]
            msk = np.concatenate([msk_loc[..., None], msk_dmg_hot, msk_buildings[..., None]], axis=-1)
        return img, msk


class Normalize(object):
    def __call__(self, args):
        img, msk = args
        img = np.asarray(img, dtype='float32')
        img /= 127
        img -= 1
        return img, msk


class RandomFlip(object):
    def __init__(self, p: float):
        self.p = 1 - p

    def __call__(self, args):
        img, msk = args
        if random.random() > self.p:
            img = np.flip(img, axis=0)
            msk = np.flip(msk, axis=0)
        img, msk = img.copy(), msk.copy()
        return img, msk


class RandomRotate(object):
    def __init__(self, p: float):
        self.p = 1 - p

    def __call__(self, args):
        img, msk = args
        if random.random() > self.p:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot).copy()
                msk = np.rot90(msk, k=rot).copy()
        return img, msk


class RandomRotate(object):
    def __init__(self, p: float):
        self.p = 1 - p

    def __call__(self, args):
        img, msk = args
        if random.random() > self.p:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot).copy()
                msk = np.rot90(msk, k=rot).copy()
        return img, msk


class RandomAffine(object):
    def __init__(self, p: float):
        self.p = 1 - p

    def __call__(self, args):
        img, msk = args
        if random.random() > self.p:
            rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img, msk = self.rotate_data(img, msk, angle, scale, rot_pnt)
        return img, msk

    @staticmethod
    def rotate_data(img, mask, angle, scale, rot_pnt):
        rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
        result_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
        result_msk = cv2.warpAffine(mask, rot_mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return result_img, result_msk


class RandomSmartCrop(object):
    def __init__(self, p: float, input_shape: tuple, split: str = 'legacy', dynamic_weights: bool = True):
        self.p = 1 - p
        self.input_shape = input_shape
        # these weights are computed based on the damage class distribution
        if split == 'no_overlap' and dynamic_weights:
            self.weights = [1.0, 9.784601349119356, 7.696044296563575, 17.5313799876951]
        elif split == 'legacy' or split == 'changeos' or not dynamic_weights:
            self.weights = [1.0, 9.047880324809197, 8.682076906271057, 12.963227101725556]
        else:
            raise NotImplementedError()

    def __call__(self, args):
        img, msk = args
        crop_size = self.input_shape[0]
        if random.random() > self.p:
            crop_size = random.randint(int(self.input_shape[0] / 1.15), int(self.input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for _ in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = 0
            for c in range(4):
                _sc += (msk[y0:y0 + crop_size, x0:x0 + crop_size, 1] == c + 1).sum() * self.weights[c]
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk = msk[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != self.input_shape[0]:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, self.input_shape, interpolation=cv2.INTER_LINEAR)

        return img, msk


class RandomSimpleCrop(object):
    def __init__(self, p: float, input_shape: tuple):
        self.p = 1 - p
        self.input_shape = input_shape

    def __call__(self, args):
        img, msk = args
        crop_size = self.input_shape[0]
        if random.random() > self.p:
            crop_size = random.randint(int(self.input_shape[0] / 1.15), int(self.input_shape[0] / 0.85))

        x0 = random.randint(0, img.shape[1] - crop_size)
        y0 = random.randint(0, img.shape[0] - crop_size)
        img = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk = msk[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != self.input_shape[0]:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, self.input_shape, interpolation=cv2.INTER_LINEAR)

        return img, msk