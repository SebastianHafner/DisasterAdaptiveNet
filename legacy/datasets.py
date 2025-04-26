from pathlib import Path
import random
from typing import Sequence, Dict, Tuple, List
import cv2

from imgaug import augmenters as iaa
import numpy as np
from sklearn.model_selection import train_test_split
from legacy.utils import rotate_image
import torch
from torch.utils.data import Dataset

np.random.seed(1)
random.seed(1)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def normalize_image(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


class ClassificationDataset(Dataset):
    """Dataset for training classification model."""

    def __init__(self, idxs: Sequence[int], all_files: Sequence[Path], input_shape: Sequence[int], task: str) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.all_files = all_files
        self.idxs = idxs
        self.task = task

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _idx = self.idxs[idx]

        fn = self.all_files[_idx]

        img_pre = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)

        msk_pre = cv2.imread(fn.replace('images', 'masks'), cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(fn.replace('images', 'masks').replace('_pre_disaster', '_post_disaster'),
                              cv2.IMREAD_UNCHANGED)

        if self.task == "val":
            msk0 = msk_pre
            n_lbls = 4
            msk14 = np.zeros((msk_post.shape[0], msk_post.shape[1], n_lbls), dtype=np.float32)
            for i in range(n_lbls):
                msk14[msk_post == i + 1, i] = 255

            msk = np.concatenate([msk0[..., np.newaxis], msk14], axis=2)
            msk = (msk > 127)

        else:
            assert self.task == 'train'
            img_pre, img_post, msk = self.augmentations_train(img_pre, img_post, msk_pre, msk_post)

        msk = msk * 1
        lbl_msk = msk[..., 1:].argmax(axis=2) if self.task == "val" else msk.argmax(axis=2)
        img = np.concatenate([img_pre, img_post], axis=2)
        img = normalize_image(img)

        # Reshaping tensors from (H, W, C) to (C, H, W)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        return {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}

    def augmentations_train(self, img, img2, msk0, lbl_msk1):
        n_lbls = 4
        msk14 = np.zeros((lbl_msk1.shape[0], lbl_msk1.shape[1], n_lbls), dtype=np.float32)
        for i in range(n_lbls):
            msk14[lbl_msk1 == i + 1, i] = 255

        if random.random() > 0.5:
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk14 = msk14[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk14 = np.rot90(msk14, k=rot)

        if random.random() > 0.6:
            rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk14 = rotate_image(msk14, angle, scale, rot_pnt)

        crop_size = self.input_shape[0]
        if random.random() > 0.2:
            crop_size = random.randint(
                int(self.input_shape[0] / 1.15), int(self.input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk14[y0:y0 + crop_size, x0:x0 + crop_size, 1].sum() * 9.047880324809197 + \
                  msk14[y0:y0 + crop_size, x0:x0 + crop_size, 2].sum() * 8.682076906271057 + \
                  msk14[y0:y0 + crop_size, x0:x0 + crop_size, 3].sum() * 12.963227101725556 + \
                  msk14[y0:y0 + crop_size, x0:x0 + crop_size, 0].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
        img2 = img2[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk0 = msk0[y0:y0 + crop_size, x0:x0 + crop_size]
        msk14 = msk14[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != self.input_shape[0]:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, self.input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, self.input_shape, interpolation=cv2.INTER_LINEAR)
            msk14 = cv2.resize(msk14, self.input_shape, interpolation=cv2.INTER_LINEAR)

        msk = np.concatenate([msk0[..., np.newaxis], msk14], axis=2)
        msk = (msk > 127)

        return img, img2, msk


def get_stratified_train_val_split(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get train/val split stratified by disaster name."""
    train_dirs = ['train', 'tier3']

    all_files = []
    for d in train_dirs:
        img_folder = dataset_dir / d / 'images'
        files = sorted([str(f) for f in img_folder.glob('**/*') if '_pre_disaster.png' in f.name])
        all_files.extend(files)

    # Fixed stratified sample to split data into train/val
    disaster_names = list(map(lambda path: Path(path).name.split("_")[0], all_files))
    train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=23,
                                            stratify=disaster_names)
    return train_idxs, val_idxs, all_files


def get_train_val_datasets(dataset_dir: Path, input_shape: int, debug: bool = False) -> Tuple[Dataset, Dataset]:

    train_idxs, val_idxs, all_files = get_stratified_train_val_split(dataset_dir)

    if not debug:
        # Oversample images that contain buildings.
        # This should lead to roughly 50-50 distribution between images with and without buildings.
        file_classes = []
        for fn in all_files:
            fl = np.zeros((4,), dtype=bool)
            msk1 = cv2.imread(fn.replace('images', 'masks').replace('_pre_disaster', '_post_disaster'),
                              cv2.IMREAD_UNCHANGED)
            for c in range(1, 5):
                fl[c - 1] = c in msk1
            file_classes.append(fl)
        file_classes = np.asarray(file_classes)

        new_train_idxs = []
        for i in train_idxs:
            new_train_idxs.append(i)
            if file_classes[i, 1:].max():
                new_train_idxs.append(i)
        train_idxs = np.asarray(new_train_idxs)

    data_train = ClassificationDataset(idxs=train_idxs, all_files=all_files, input_shape=input_shape, task='train')
    val_train = ClassificationDataset(idxs=val_idxs, all_files=all_files, input_shape=input_shape, task="val")

    return data_train, val_train


if __name__ == "__main__":
    train_data, val_data = get_train_val_datasets(1, (512, 512), False, False, None, False)