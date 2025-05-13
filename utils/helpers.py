import json
from pathlib import Path


def load_json(file: Path):
    with open(str(file)) as f:
        d = json.load(f)
    return d


def write_json(file: Path, data):
    with open(str(file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
