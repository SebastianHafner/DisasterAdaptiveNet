from pathlib import Path
import numpy as np
from PIL import ImageColor
import json

DAMAGE_COLORS = ['#FFFFFF', '#00C600', '#FFB966', '#F10000']


def dmg2img(pred: np.ndarray) -> np.ndarray:
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for i in range(4):
        color = ImageColor.getcolor(DAMAGE_COLORS[i], "RGB")
        dmg_msk = pred == i + 1
        for c in range(3):
            img[dmg_msk, 2 - c] = color[c]
    return img


def load_json(file: Path):
    with open(str(file)) as f:
        d = json.load(f)
    return d


def write_json(file: Path, data):
    with open(str(file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
