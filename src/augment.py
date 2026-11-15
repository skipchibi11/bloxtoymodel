"""データの水増し。学習時にランダムに適用する（ファイルは増やさない）。

- 回転：縦軸（y）まわりに 0/90/180/270 度
- 反転：左右反転（あり/なし）
  → 回転×反転で 8 通り
- 平行移動：グリッドからはみ出さない範囲で、水平方向（x, z）にずらす。
  高さ（y）は地面との関係が意味を持つのでずらさない。

色は変えない（「赤いバラ」が青くなるとラベルと合わなくなるため）。
"""

import numpy as np

from . import config as C

G1 = C.GRID - 1


def rotate_mirror(blocks, k, mirror):
    """k: 90度回転の回数（0〜3）, mirror: 左右反転するか"""
    b = blocks.copy()
    x, z = b[:, 0].copy(), b[:, 2].copy()
    for _ in range(k % 4):
        x, z = G1 - z, x
    if mirror:
        x = G1 - x
    b[:, 0], b[:, 2] = x, z
    return b


def shift(blocks, dx, dz):
    b = blocks.copy()
    b[:, 0] += dx
    b[:, 2] += dz
    return b


def shift_range(blocks):
    """はみ出さずに動かせる (dx の最小, 最大, dz の最小, 最大)"""
    return (-int(blocks[:, 0].min()), G1 - int(blocks[:, 0].max()),
            -int(blocks[:, 2].min()), G1 - int(blocks[:, 2].max()))


def random_augment(blocks, rng, max_shift=2):
    b = rotate_mirror(blocks, int(rng.integers(4)), bool(rng.integers(2)))
    lo_x, hi_x, lo_z, hi_z = shift_range(b)
    dx = int(rng.integers(max(lo_x, -max_shift), min(hi_x, max_shift) + 1))
    dz = int(rng.integers(max(lo_z, -max_shift), min(hi_z, max_shift) + 1))
    return shift(b, dx, dz)
