import json
from shapely.wkt import loads
from multiprocessing import Pool
import sys
from pathlib import Path
import timeit
import cv2
import random
import numpy as np
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

np.random.seed(1)
random.seed(1)
sys.setrecursionlimit(10000)

###To be changed####
dataset_dir = Path('C:/Users/shafner/datasets/xview2')
# dirs = ['train', 'tier3']
dirs = ['hold']


def mask_for_polygon(poly, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)

    def int_coords(x): return np.array(x).round().astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1  # ?
}


def process_image(json_file: Path):
    buildings = json.load(open(str(json_file)))

    building_ids = np.zeros((1024, 1024), dtype='uint16')

    building_features = buildings['features']['xy']
    assert len(building_features) < 65535
    for i, feat in enumerate(building_features):
        poly = loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        building_ids[_msk > 0] = i + 1

    event, patch_id, *_ = json_file.stem.split('_')
    buildings_file = json_file.parent.parent / 'buildings' / f'{event}_{patch_id}_buildings.png'
    cv2.imwrite(str(buildings_file), building_ids, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    all_files = []
    for d in dirs:
        buildings_dir = dataset_dir / d / 'buildings'
        buildings_dir.mkdir(exist_ok=True)

        labels_dir = dataset_dir / d / 'labels'
        labels_files = list([f for f in labels_dir.glob('*.json')])
        for f in labels_files:
            if '_pre_disaster' in f.stem:
                all_files.append(f)

    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
