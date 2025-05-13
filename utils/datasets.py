from typing import Sequence, Dict, Any, Union

import torch
from abc import abstractmethod

from utils import augmentations, helpers
from utils.experiment_manager import CfgNode
import cv2

from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy as np


class AbstractxBDDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.metadata = helpers.load_json(self.root_path / 'metadata.json')

        # for new dataset split
        self.train_events = list(cfg.DATASET.TRAIN_EVENTS)
        self.test_events = list(cfg.DATASET.TEST_EVENTS)
        self.exclude_events = list(cfg.DATASET.EXCLUDE_EVENTS)

        # split legacy/new
        self.split = cfg.DATASET.SPLIT
        if self.split == 'xview2':
            self.samples_splits = self.get_samples_legacy()
        elif self.split == 'event':
            self.samples_splits = self.get_samples()
        else:
            raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def load_images(self, subset: str, event: str, patch_id: str) -> Sequence[np.ndarray]:
        img_pre_file = self.root_path / subset / 'images' / f'{event}_{patch_id}_pre_disaster.png'
        img_pre = cv2.imread(str(img_pre_file), cv2.IMREAD_COLOR)
        img_post_file = self.root_path / subset / 'images' / f'{event}_{patch_id}_post_disaster.png'
        img_post = cv2.imread(str(img_post_file).replace('_pre_', '_post_'), cv2.IMREAD_COLOR)
        return img_pre, img_post

    def load_masks(self, subset: str, event: str, patch_id: str) -> Sequence[np.ndarray]:
        # 0: background, 1: building
        msk_pre_file = self.root_path / subset / 'masks' / f'{event}_{patch_id}_pre_disaster.png'
        msk_pre = cv2.imread(str(msk_pre_file), cv2.IMREAD_UNCHANGED)
        msk_pre = msk_pre.astype(np.float32) / 255
        # 0: background, 1: no damage, 2: minor damage, 3: major damage, 4: destroyed
        msk_post_file = self.root_path / subset / 'masks' / f'{event}_{patch_id}_post_disaster.png'
        msk_post = cv2.imread(str(msk_post_file), cv2.IMREAD_UNCHANGED)
        return msk_pre, msk_post

    def get_samples_legacy(self) -> Dict:
        """Get train/val split stratified by disaster name."""
        subsets = ['train', 'tier3']
        all_samples = []
        for subset in subsets:
            subset_samples = sorted(self.metadata[subset]['patches'], key=lambda s: f'{s["event"]}_{s["patch_id"]}')
            all_samples.extend(subset_samples)

        if self.cfg.DATASET.EXCLUDE_UNDAMAGED:
            all_samples = [s for s in all_samples if (s['cls_2'] or s['cls_3'] or s['cls_4'])]

        disaster_names = [s['event'] for s in all_samples]

        # Fixed stratified sample to split data into train/val
        train_indices, val_indices = train_test_split(np.arange(len(all_samples)), test_size=0.1,
                                                      random_state=self.cfg.SEED, stratify=disaster_names)

        if self.cfg.DATASET.OVERSAMPLE_BUILDINGS:
            # Oversample images that contain buildings. But seems to oversample damage > minor damage?
            # This should lead to roughly 50-50 distribution between images with and without buildings.
            train_indices_new = list(train_indices)
            for i in train_indices:
                fl = np.zeros(4, dtype=bool)
                for c in range(1, 5):
                    fl[c - 1] = all_samples[i][f'cls_{c}'] > 0
                if fl[1:].max():
                    train_indices_new.append(i)
            train_indices = train_indices_new

        samples = {
            'train': [all_samples[i] for i in train_indices],
            'val': [all_samples[i] for i in val_indices],
            'test': self.metadata['test']['patches'],
        }
        return samples

    def get_samples(self) -> Dict:
        """Get train/val split stratified by disaster name."""
        subsets = ['train', 'tier3', 'test', 'hold']
        all_samples = []
        for subset in subsets:
            all_samples.extend(self.metadata[subset]['patches'])

        if self.cfg.DATASET.EXCLUDE_UNDAMAGED:
            all_samples = [s for s in all_samples if (s['cls_2'] or s['cls_3'] or s['cls_4'])]

        trainval_samples = [s for s in all_samples if s['event'] in self.train_events]
        trainval_event_names = [s['event'] for s in trainval_samples]

        # Fixed stratified sample to split data into train/val
        train_indices, val_indices = train_test_split(np.arange(len(trainval_samples)), test_size=0.1,
                                                      random_state=self.cfg.SEED, stratify=trainval_event_names)

        if self.cfg.DATASET.OVERSAMPLE_BUILDINGS:
            # Oversample images that contain buildings. But seems to oversample damage > minor damage?
            # This should lead to roughly 50-50 distribution between images with and without buildings.
            train_indices_new = list(train_indices)
            for i in train_indices:
                fl = np.zeros(4, dtype=bool)
                for c in range(1, 5):
                    fl[c - 1] = trainval_samples[i][f'cls_{c}'] > 0
                if fl[1:].max():
                    train_indices_new.append(i)
            train_indices = train_indices_new

        samples = {
            'train': [trainval_samples[i] for i in train_indices],
            'val': [trainval_samples[i] for i in val_indices],
            'test': [s for s in all_samples if s['event'] in self.test_events],
        }
        return samples


# dataset for urban extraction with building footprints
class xBDDataset(AbstractxBDDataset):

    def __init__(self, cfg, run_type: str, disable_augmentations: bool = False):
        super().__init__(cfg)
        self.run_type = run_type
        augs = True if (run_type == 'train' and not disable_augmentations) else False
        self.transforms = augmentations.compose_transformations(cfg, augs_enabled=augs)
        self.n_dmg_classes = 4

        self.samples = list(self.samples_splits[run_type])
        self.length = len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Any, str]]:
        sample = self.samples[index]
        event, patch_id, subset = sample['event'], sample['patch_id'], sample['subset']

        # load images and masks
        img_pre, img_post = self.load_images(subset, event, patch_id)
        img = np.concatenate([img_pre, img_post], axis=2)

        msk_loc, msk_dmg = self.load_masks(subset, event, patch_id)
        msk = np.stack((msk_loc, msk_dmg), axis=-1)
        img, msk = self.transforms((img, msk))

        # Reshaping tensors from (H, W, C) to (C, H, W)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).bool()

        item = {'img': img, 'msk': msk, 'event': event, 'patch_id': patch_id, 'subset': subset}

        if self.cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
            cond_attr = self.cfg.DATASET.EVENT_CONDITIONING[event]
            cond_id = int(self.cfg.DATASET.CONDITIONING_KEY[cond_attr])
            item['cond_id'] = torch.tensor([cond_id]).long()

        return item

    def get_class_counts(self) -> Sequence[int]:
        class_counts = [0, 0, 0, 0, 0]
        for sample in self.samples:
            class_counts[0] += sample['loc']
            for i in range(1, 5):
                class_counts[i] += sample[f'cls_{i}']
        return class_counts

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
