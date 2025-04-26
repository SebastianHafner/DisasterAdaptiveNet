# xview2 metrics
#
# the total score is calculated a weighted average of the localization f1 score (lf1) and the damage f1 score (df1)
# score = .3 * lf1 + .7 * df1
#
# the df1 is calculated by taking the harmonic mean of the 4 damage f1 scores
# (no damage, minor damage, major damage, and destroyed)
# df1 = 4 / sum((f1+epsilon)**-1 for f1 in [no_damage_f1, minor_damage_f1, major_damage_f1, destroyed_f1]),
# where epsilon = 1e-6
#
# Abbreviations used in this file:
# l: localization
# d: damage
# p: prediction
# t: target (ground truth)
# x: usually a numpy array
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd
import sklearn.metrics as sm

from PIL import Image

from utils import parsers, experiment_manager, datasets
from utils.experiment_manager import CfgNode


class PathHandler:
    def __init__(self, cfg: CfgNode, subset: str, img_id: str, disaster_name: str):
        """
        Args:
            pred_dir  (Path): directory of localization and damage predictions
            targ_dir  (Path): directory of localization and damage targets
            img_id    (str) : # 5 digit string of image id
            disaster_name (str) :
        """
        self.cfg = cfg

        self.patch = (disaster_name, img_id, subset)

        # predictions
        self.pred_dir = Path(cfg.PATHS.OUTPUT) / 'predictions' / cfg.NAME / f'submission_{cfg.NAME}'
        self.lp = self.pred_dir / f"{disaster_name}_{img_id}_localization_disaster_prediction.png"
        self.dp = self.pred_dir / f"{disaster_name}_{img_id}_damage_disaster_prediction.png"

        # targets
        self.dataset_dir = Path(cfg.PATHS.DATASET)
        self.lt = self.dataset_dir / subset / 'targets' / f"{disaster_name}_{img_id}_pre_disaster_target.png"
        self.dt = self.dataset_dir / subset / 'targets' / f"{disaster_name}_{img_id}_post_disaster_target.png"

        self.paths = (self.lp, self.dp, self.lt, self.dt)

    def load_and_validate_image(self, path):
        assert path.is_file(), f"file '{path}' does not exist or is not a file"
        img = np.array(Image.open(path))
        assert img.dtype == np.uint8, f"{path.name} is of wrong format {img.dtype} - should be np.uint8"
        # assert set(np.unique(img)) <= {0, 1, 2, 3, 4}, f"values must ints 0-4, found {np.unique(img)}, path: {path}"
        assert img.shape == (1024, 1024), f"{path} must be a 1024x1024 image"
        return img

    def load_images(self):
        return [self.load_and_validate_image(path) for path in self.paths]


class RowPairCalculator:
    """
    Contains all the information and functions necessary to calculate the true positives (TPs),
    false negatives (FNs), and false positives (FPs), for a pair of localization/damage predictions
    """

    @staticmethod
    def extract_buildings(x: np.ndarray):
        """ Returns a mask of the buildings in x """
        buildings = x.copy()
        buildings[x > 0] = 1
        return buildings

    @staticmethod
    def compute_tp_fn_fp(pred: np.ndarray, targ: np.ndarray, c: int) -> List[int]:
        """
        Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)

        Args:
            pred (np.ndarray): prediction
            targ (np.ndarray): target
            c (int): positive class
        """
        TP = np.logical_and(pred == c, targ == c).sum()
        FN = np.logical_and(pred != c, targ == c).sum()
        FP = np.logical_and(pred == c, targ != c).sum()
        return [TP, FN, FP]

    @classmethod
    def get_row_pair(cls, ph: PathHandler):
        """
        Builds a row of TPs, FNs, and FPs for both the localization dataframe and the damage dataframe.
        This pair of rows are built in the same function as damages are only assessed where buildings are predicted.


        Args:
            ph (PathHandler): used to load the required prediction and target images
        """

        # loc prediction, dmg prediction, loc target, dmg target
        lp, dp, lt, dt = ph.load_images()

        if ph.cfg.INFERENCE.REGRESSION:
            # convert damage regression to damage classes
            dp = dp / 255  # rescale [0, 1]

            dp_conv = np.zeros_like(dp)
            thresholds = ph.cfg.INFERENCE.DAMAGE_THRESHOLDS
            for i in range(1, 5):
                low_thr, up_thr = thresholds[i - 1], thresholds[i]
                # Since we initialize with with 0s "<" for low_thr = 0 will result in 0
                d_bool = np.logical_and(low_thr < dp, dp <= up_thr)
                dp_conv[d_bool] = i
            dp = dp_conv.copy()

        # convert all damage scores 1-4 to 1
        lp_b, lt_b, dt_b = map(cls.extract_buildings, (lp, lt, dt))

        dp = dp * lp_b  # only give credit to damages where buildings are predicted
        # only score damage where there exist buildings in target damage
        dp, dt = dp[dt_b == 1], dt[dt_b == 1]

        lrow = cls.compute_tp_fn_fp(lp_b, lt_b, 1)
        drow = []
        for i in range(1, 5):
            drow += cls.compute_tp_fn_fp(dp, dt, i)

        conf_mat = sm.confusion_matrix(dt.flatten(), dp.flatten(), labels=[1, 2, 3, 4])

        return (lrow, drow, ph.patch), conf_mat


class F1Recorder:
    """
    Records the precision and recall when calculating the f1 score.
    Read about the f1 score here: https://en.wikipedia.org/wiki/F1_score
    """

    def __init__(self, TP, FP, FN, name='', return_nan=False):
        """
        Args:
            TP (int): true positives
            FP (int): false positives
            FN (int): false negatives
            name (str): optional name when printing
            return_nan (bool): if TP and FP are 0, return NaN instead of 0
        """
        self.TP, self.FN, self.FP, self.name, self.return_nan = TP, FN, FP, name, return_nan
        self.P = self.precision()
        self.R = self.recall()
        self.f1 = self.f1()

    def __repr__(self):
        return f'{self.name} | f1: {self.f1:.4f}, precision: {self.P:.4f}, recall: {self.R:.4f}'

    def precision(self):
        """ calculates the precision using the true positives (self.TP) and false positives (self.FP)"""
        assert self.TP >= 0 and self.FP >= 0
        if self.TP == 0:
            if self.FN == 0 and self.return_nan:
                return float("NaN")
            else:
                return 0
        else:
            return self.TP / (self.TP + self.FP)

    def recall(self):
        """ calculates recall using the true positives (self.TP) and false negatives (self.FN) """
        assert self.TP >= 0 and self.FN >= 0
        if self.TP == 0:
            if self.FN == 0 and self.return_nan:
                return float("NaN")
            else:
                return 0
        return self.TP / (self.TP + self.FN)

    def f1(self):
        """ calculates the f1 score using precision (self.P) and recall (self.R) """
        if self.return_nan and (np.isnan(self.P) or np.isnan(self.R)):
            return float("NaN")

        assert 0 <= self.P <= 1 and 0 <= self.R <= 1
        if self.P == 0 or self.R == 0:
            return 0
        return (2 * self.P * self.R) / (self.P + self.R)


class XviewMetrics:
    """
    Calculates the xview2 metrics given a directory of predictions and a directory of targets

    Directory of predictions and directory of targets must be two separate directories. These
    could be structured as followed:
        .
        ├── predictions
        │   ├── test_damage_00000_prediction.png
        │   ├── test_damage_00001_prediction.png
        │   ├── test_localization_00000_prediction.png
        │   ├── test_localization_00001_prediction.png
        │   └── ...
        └── targets
            ├── test_damage_00000_target.png
            ├── test_damage_00001_target.png
            ├── test_localization_00000_target.png
            ├── test_localization_00001_target.png
            └── ...
    """

    def __init__(self, cfg: CfgNode, event_keyword):
        self.cfg = cfg
        self.samples = datasets.xBDDataset(cfg, run_type='test').samples
        self.event_keyword = event_keyword

        self.dmg2str = {
            1: f'No damage     (1) ',
            2: f'Minor damage  (2) ',
            3: f'Major damage  (3) ',
            4: f'Destroyed     (4) ',
        }

        self.get_path_handlers()
        self.get_type_error()
        self.get_lf1r()
        self.get_df1rs()
        self.get_per_imgs_metrics()

    def __repr__(self):
        s = 'Localization:\n'
        s += f'    {self.lf1r}\n'

        s += '\nDamage:\n'
        for F1Rec in self.df1rs:
            s += f'    {F1Rec}\n'
        s += f'    Harmonic mean dmgs | f1: {self.df1:.4f}\n'

        s += '\nScore:\n'
        s += f'    Score | f1: {self.score:.4f}\n'
        return s.rstrip()

    def get_path_handlers(self):
        self.path_handlers = []
        for s in self.samples:
            disaster_name, img_id, subset = s['event'], s['patch_id'], s['subset']
            if self.event_keyword is None or self.event_keyword == disaster_name:
                self.path_handlers.append(PathHandler(self.cfg, subset, img_id, disaster_name))

    def get_type_error(self):
        """
        builds the localization dataframe (self.ldf) and damage dataframe (self.ddf) from
        path handlers (self.path_handlers) and computes the damage confusion matrix (averaged over the whole dataset);
        dataframes consist purely of rows of TP, FN, FP for loc and cls (per class)
        """
        with Pool() as p:
            all_rows, dmg_confs = list(zip(*p.map(RowPairCalculator.get_row_pair, self.path_handlers)))

        patch_columns = ['disaster_name', 'img_id', 'subset']
        self.patches = pd.DataFrame([patch for _, _, patch in all_rows], columns=patch_columns)

        lcolumns = ['lTP', 'lFN', 'lFP']
        self.ldf = pd.DataFrame([lrow for lrow, _, _ in all_rows], columns=lcolumns)

        dcolumns = ['dTP1', 'dFN1', 'dFP1', 'dTP2', 'dFN2', 'dFP2', 'dTP3', 'dFN3', 'dFP3', 'dTP4', 'dFN4', 'dFP4']
        self.ddf = pd.DataFrame([drow for _, drow, _ in all_rows], columns=dcolumns)

        self.dmg_conf = np.sum(dmg_confs, axis=0)

    def get_lf1r(self):
        """ localization f1 recorder """
        TP = self.ldf['lTP'].sum()
        FP = self.ldf['lFP'].sum()
        FN = self.ldf['lFN'].sum()
        self.lf1r = F1Recorder(TP, FP, FN, 'Buildings')

    def get_metrics_for_single_img(self, row: pd.Series):
        TP = row['lTP']
        FP = row['lFP']
        FN = row['lFN']
        lf1 = F1Recorder(TP, FP, FN, 'Buildings', return_nan=True).f1

        df1s = []
        for i in range(1, 5):
            TP = row[f'dTP{i}']
            FP = row[f'dFP{i}']
            FN = row[f'dFN{i}']
            df1s.append(F1Recorder(TP, FP, FN, self.dmg2str[i], return_nan=True).f1)

        df1s_clean = [x for x in df1s if not np.isnan(x)]
        if len(df1s_clean) == 0:
            df1 = float("NaN")
            score = float("NaN")
        else:
            df1 = self.harmonic_mean(df1s_clean)
            score = 0.3 * lf1 + 0.7 * df1

        patch_cols = ['disaster_name', 'img_id', 'subset']
        df = pd.Series(pd.concat([pd.Series({
            'score': score,
            'localization_f1': lf1,
            'damage_f1': df1,
            'damage_f1_no_damage': df1s[0],
            'damage_f1_minor_damage': df1s[1],
            'damage_f1_major_damage': df1s[2],
            'damage_f1_destroyed': df1s[3]}),
            row[patch_cols].map(str), row.drop(patch_cols),
        ]))
        return df

    def get_per_imgs_metrics(self):
        source_df = pd.concat([self.ldf, self.ddf, self.patches], axis=1)
        self.per_img_metrics_df = source_df.apply(self.get_metrics_for_single_img, axis=1)

    @property
    def lf1(self):
        """ localization f1 """
        return self.lf1r.f1

    def get_df1rs(self):
        """ damage f1 recorders """
        self.df1rs = []
        for i in range(1, 5):
            TP = self.ddf[f'dTP{i}'].sum()
            FP = self.ddf[f'dFP{i}'].sum()
            FN = self.ddf[f'dFN{i}'].sum()
            self.df1rs.append(F1Recorder(TP, FP, FN, self.dmg2str[i]))

    @property
    def df1s(self):
        """ damage f1s """
        return [F1.f1 for F1 in self.df1rs]

    @staticmethod
    def harmonic_mean(xs):
        return len(xs) / sum((x + 1e-6) ** -1 for x in xs)

    @property
    def df1(self):
        """ damage f1. Computed using harmonic mean of damage f1s """
        # remove 0s so the harmonic mean doesn't vanish when there is no example of one type of damage
        return self.harmonic_mean(self.df1s)

    @property
    def score(self):
        """ xview2 score computed as a weighted average of the localization f1 and damage f1 """
        return 0.3 * self.lf1 + 0.7 * self.df1

    @classmethod
    def get_score(cls, cfg: CfgNode, evaluation_strategy='full'):
        """Computes the xview2 scores at an event level or at the dataset level.
            Args:
                cfg : Config
                evaluation_strategy: 'both', 'full' or 'event'. If 'full' the score is computed
                    for the full dataset. If 'event' the score is computed for each event separately.
        """
        if evaluation_strategy in ('both', 'full'):
            """computing score for the full dataset"""
            cls.compute_score(cfg, event_keyword=None)
        if evaluation_strategy in ('both', 'event'):
            raise NotImplementedError()
            # for event in events:
            #     print(f"computing score for event {event}")
            #     cls.compute_score(cfg, event_keyword=event)
        if evaluation_strategy not in ('both', 'event', 'full'):
            raise ValueError(
                f"evaluation_strategy must be either 'both', 'full' or 'event' but is {evaluation_strategy}")

    @classmethod
    def compute_score(cls, cfg: CfgNode, event_keyword):
        """Computes the xview2 scores, saves it to a json file and push it to wandb.
        If a event_keyword is provided, the score is computed only for the images
        of the corresponding event.

        Args:
            cfg : Config
            event_keyword : If provided, the score is computed only for the images
            of the corresponding event. if None, the score is computed for the full dataset.
        """
        print(f"Calculating metrics using {cpu_count()} cpus...")

        self = cls(cfg, event_keyword)

        d = {
            'localization_f1': self.lf1,
            'damage_f1_no_damage': self.df1s[0],
            'damage_f1_minor_damage': self.df1s[1],
            'damage_f1_major_damage': self.df1s[2],
            'damage_f1_destroyed': self.df1s[3],
        }

        if event_keyword is None:
            d = {'score': self.score, 'damage_f1': self.df1, **d}
            scores_file = Path(cfg.PATHS.OUTPUT) / 'scores' / f'scores_{cfg.NAME}.json'
        else:
            scores_file = Path(cfg.PATHS.OUTPUT) / 'scores' / f'scores_{cfg.NAME}_{event_keyword}.json'

        with open(scores_file, 'w') as json_file:
            json.dump(d, json_file, indent=2)
        print(f"Wrote summary metrics to {scores_file.name}")
        print(d)

        out_file = Path(cfg.PATHS.OUTPUT) / 'scores' / f'{scores_file.stem}.csv'
        self.per_img_metrics_df.to_csv(out_file, index=False, sep=';')


if __name__ == '__main__':
    args = parsers.prediction_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    XviewMetrics.get_score(cfg)
