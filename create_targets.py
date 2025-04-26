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
    subset_dir = json_file.parent.parent
    event, patch_id = json_file.stem.split('_')[:2]

    js1 = json.load(open(str(json_file)))
    json2_file = subset_dir / 'labels' / f'{event}_{patch_id}_post_disaster.json'
    js2 = json.load(open(str(json2_file)))

    msk = np.zeros((1024, 1024), dtype='uint8')
    msk_damage = np.zeros((1024, 1024), dtype='uint8')

    for feat in js1['features']['xy']:
        poly = loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        msk[_msk > 0] = 1  # only difference to msk

    for feat in js2['features']['xy']:
        poly = loads(feat['wkt'])
        subtype = feat['properties']['subtype']
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict[subtype]

    loc_file = subset_dir / 'targets' / f'{event}_{patch_id}_pre_disaster_target.png'
    cv2.imwrite(str(loc_file), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    dmg_file = subset_dir / 'targets' / f'{event}_{patch_id}_post_disaster_target.png'
    cv2.imwrite(str(dmg_file), msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    all_files = []
    targets_dir = dataset_dir / 'tier3' / 'targets'
    assert not targets_dir.exists()
    targets_dir.mkdir(exist_ok=True)

    images_dir = dataset_dir / 'tier3' / 'images'
    image_files = list([f for f in images_dir.glob('*.png')])
    for f in image_files:
        if '_pre_disaster.png' in f.name:
            all_files.append(dataset_dir / 'tier3' / 'labels' / f'{f.stem}.json')

    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
