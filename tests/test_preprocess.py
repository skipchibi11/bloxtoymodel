"""前処理の動作確認。  .venv/bin/python -m tests.test_preprocess"""

import json

import numpy as np

from src import config as C
from src.augment import random_augment, rotate_mirror
from src.vocab import (MaterialRegistry, blocks_to_json, decode, dequantize, encode,
                       json_to_blocks, quantize_hex, sort_blocks)


def as_set(b):
    return {tuple(r) for r in b.tolist()}


def main():
    # 色の丸め
    assert quantize_hex("#000000") == (0, 0, 0)
    assert quantize_hex("#ffffff") == (15, 15, 15)
    assert dequantize(*quantize_hex("#ffffff")) == "#ffffff"
    for h in ["#6fa57a", "#f8b4b4", "#2b8a3e"]:
        back = dequantize(*quantize_hex(h))
        assert all(abs(int(h[i:i + 2], 16) - int(back[i:i + 2], 16)) <= 9 for i in (1, 3, 5)), (h, back)

    files = sorted(C.DATA_DIR.glob("*.json"))
    assert files, "data/ が空"
    for path in files:
        mats = MaterialRegistry.load()
        doc = json.loads(path.read_text())
        blocks, _ = json_to_blocks(doc, mats, add_materials=True)

        # トークン化 → 復元で元に戻る
        assert as_set(decode(encode(blocks))) == as_set(blocks), path.name

        # JSON → ブロック → JSON → ブロック で元に戻る
        again, _ = json_to_blocks(blocks_to_json(blocks, mats), mats)
        assert as_set(again) == as_set(blocks), path.name

        # 回転・反転の 8 通り：数・範囲・重複なし。4回回すと元通り
        for k in range(4):
            for m in (False, True):
                b = rotate_mirror(blocks, k, m)
                assert len(as_set(b[:, :3])) == len(blocks)
                assert b[:, :3].min() >= 0 and b[:, :3].max() < C.GRID
        b = blocks
        for _ in range(4):
            b = rotate_mirror(b, 1, False)
        assert as_set(b) == as_set(blocks)

        # ランダム水増し：はみ出さない・高さは変わらない
        rng = np.random.default_rng(1)
        for _ in range(50):
            b = random_augment(blocks, rng)
            assert b[:, :3].min() >= 0 and b[:, :3].max() < C.GRID, path.name
            assert sorted(b[:, 1].tolist()) == sorted(blocks[:, 1].tolist())

        # 並び順は下から上
        s = sort_blocks(blocks)
        assert (np.diff(s[:, 1]) >= 0).all()

    # 壊れたトークン列でも落ちない
    assert len(decode([1, 5, 999, 3, 2])) == 0
    print(f"OK: {len(files)} ファイルで全テスト通過")


if __name__ == "__main__":
    main()
