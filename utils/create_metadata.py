import numpy as np
from pathlib import Path
import cv2
from typing import Sequence
import json
from utils import parsers, geofiles

DISASTER_TYPES = {
    '0': 'fire',
    '1': 'volcano',
    '2': 'earthquake',
    '3': 'tornado',
    '4': 'flood',
}

PERILS = {
    '0': 'wildfire',
    '1': 'volcano',
    '2': 'earthquake',
    '3': 'storm',
    '4': 'flood',
    '5': 'tsunami',
}


EVENT_DICT_PERILS = {
    'santa-rosa-wildfire': 0,
    'woolsey-fire': 0,
    'pinery-bushfire': 0,
    'socal-fire': 0,
    'portugal-wildfire': 0,
    'guatemala-volcano': 1,
    'lower-puna-volcano': 1,
    'sunda-tsunami': 5,
    'palu-tsunami': 5,
    'mexico-earthquake': 2,
    'hurricane-matthew': 3,
    'moore-tornado': 3,
    'joplin-tornado': 3,
    'tuscaloosa-tornado': 3,
    'hurricane-harvey': 4,
    'midwest-flooding': 4,
    'hurricane-florence': 4,
    'hurricane-michael': 3,
    'nepal-flooding': 4,
}

EVENT_DICT = {
    'santa-rosa-wildfire': 0,
    'woolsey-fire': 0,
    'pinery-bushfire': 0,
    'socal-fire': 0,
    'portugal-wildfire': 0,
    'guatemala-volcano': 1,
    'lower-puna-volcano': 1,
    'sunda-tsunami': 2,
    'palu-tsunami': 2,
    'mexico-earthquake': 2,
    'hurricane-matthew': 3,
    'moore-tornado': 3,
    'joplin-tornado': 3,
    'tuscaloosa-tornado': 3,
    'hurricane-harvey': 4,
    'midwest-flooding': 4,
    'hurricane-florence': 4,
    'hurricane-michael': 4,
    'nepal-flooding': 4,
}


def create_metadata_file(dataset_path: str, subsets: Sequence[str] = ['train', 'tier3', 'test', 'hold']):
    metadata = {'subsets': subsets, 'disaster_types': DISASTER_TYPES, 'events': EVENT_DICT, 'perils': PERILS,
                'event_perils': EVENT_DICT_PERILS}
    for subset in subsets:
        metadata[subset] = {'events': [], 'patches': []}

        masks_folder = Path(dataset_path) / subset / 'masks'
        assert masks_folder.exists()
        # pre disaster file names
        files = [f for f in masks_folder.glob('*.png') if 'pre_disaster' in f.stem]
        for f in files:
            event, patch_id, *_ = f.stem.split('_')
            if event not in metadata[subset]['events']:
                metadata[subset]['events'].append(event)
            assert event in EVENT_DICT.keys(), event
            patch = {
                'event': event,
                'patch_id': patch_id,
                'subset': subset,
                'disaster_type': EVENT_DICT[event],
                'peril': EVENT_DICT_PERILS[event],
            }

            pre_mask_file = masks_folder / f'{event}_{patch_id}_pre_disaster.png'
            pre_mask = cv2.imread(str(pre_mask_file), cv2.IMREAD_UNCHANGED)
            patch[f'loc'] = int(np.sum(pre_mask == 255))

            post_mask_file = masks_folder / f'{event}_{patch_id}_post_disaster.png'
            post_mask = cv2.imread(str(post_mask_file), cv2.IMREAD_UNCHANGED)
            for class_ in range(1, 5):
                patch[f'cls_{class_}'] = int(np.sum(post_mask == class_))

            label_file = Path(dataset_path) / subset / 'labels' / f'{event}_{patch_id}_pre_disaster.json'
            building_labels = json.load(open(str(label_file)))
            n_buildings = len(building_labels['features']['xy'])
            patch['n_buildings'] = n_buildings

            metadata[subset]['patches'].append(patch)

    metadata_file = Path(dataset_path) / 'metadata.json'
    geofiles.write_json(metadata_file, metadata)


if __name__ == '__main__':
    args = parsers.dataset_path_parser().parse_known_args()[0]
    create_metadata_file(args.dataset_dir)
