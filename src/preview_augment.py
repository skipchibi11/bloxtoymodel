"""水増しの結果を outputs/ に書き出して、ビューアで確認できるようにする。

使い方:
    .venv/bin/python -m src.preview_augment bee.json
"""

import json
import sys

import numpy as np

from . import config as C
from .augment import random_augment, rotate_mirror
from .vocab import MaterialRegistry, blocks_to_json, decode, encode, json_to_blocks


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "bee.json"
    materials = MaterialRegistry.load()
    doc = json.loads((C.DATA_DIR / name).read_text())
    blocks, _ = json_to_blocks(doc, materials)
    stem = name.removesuffix(".json")
    C.OUTPUTS_DIR.mkdir(exist_ok=True)

    variants = {
        "roundtrip": decode(encode(blocks)),  # トークン化→復元（丸めた色）
        "rot90": rotate_mirror(blocks, 1, False),
        "mirror": rotate_mirror(blocks, 0, True),
        "random": random_augment(blocks, np.random.default_rng(0)),
    }
    for tag, b in variants.items():
        out = C.OUTPUTS_DIR / f"preview_{stem}_{tag}.json"
        out.write_text(json.dumps(blocks_to_json(b, materials, f"{doc.get('title', stem)} ({tag})"), indent=1))
        print(f"書き出し: {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
