"""BloxToy JSON ⇔ ブロック配列 ⇔ トークン列 の変換。

ブロック配列: int の (N, 7) 配列。列は [x, y, z, r, g, b, m]
  - x, y, z : 0〜15
  - r, g, b : 0〜15（16段階に丸めた色）
  - m       : 素材番号（0 = なし）

トークン列: 1ブロック = 3トークン（位置・色・素材）。種類ごとに番号帯を分けている。
  [BOS] (位置 色 素材) (位置 色 素材) ... [EOS]
  - 位置 = y*256 + z*16 + x（4096種類）
  - 色   = r*256 + g*16 + b（4096種類）
  ブロックは下から上へ（y → z → x の順）＝ 位置番号の小さい順に並べる。積み木を下から組む順番。
  ※ 1ブロック7トークンだと Mac の GPU では系列が長すぎて遅い（約50倍）ため、3トークンにまとめている。
"""

import json

import numpy as np

from . import config as C

PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3

FIELDS = ["p", "c", "m"]  # 位置・色・素材（トークンを出す順番）
FIELD_SIZE = {"p": C.GRID ** 3, "c": C.COLOR_LEVELS ** 3, "m": C.MATERIAL_SLOTS}
OFFSET = {}
_o = N_SPECIAL
for _f in FIELDS:
    OFFSET[_f] = _o
    _o += FIELD_SIZE[_f]
VOCAB_SIZE = _o
TOKENS_PER_BLOCK = len(FIELDS)
COL = {"x": 0, "y": 1, "z": 2, "r": 3, "g": 4, "b": 5, "m": 6}  # ブロック配列の列番号


def pos_index(x, y, z):
    return (y * C.GRID + z) * C.GRID + x


def pos_xyz(p):
    return p % C.GRID, p // (C.GRID * C.GRID), (p // C.GRID) % C.GRID  # (x, y, z)


def color_index(r, g, b):
    L = C.COLOR_LEVELS
    return (r * L + g) * L + b


def color_rgb(c):
    L = C.COLOR_LEVELS
    return c // (L * L), (c // L) % L, c % L


# ---------- 素材の一覧（追加のみ） ----------

class MaterialRegistry:
    """素材名 ⇔ 番号。一度振った番号は変えない。"""

    def __init__(self, names=None):
        self.names = list(names) if names else ["none"]
        assert self.names[0] == "none"

    @classmethod
    def load(cls, path=C.MATERIALS_FILE):
        if path.exists():
            return cls(json.loads(path.read_text())["materials"])
        return cls()

    def save(self, path=C.MATERIALS_FILE):
        path.write_text(json.dumps({"materials": self.names}, ensure_ascii=False, indent=2) + "\n")

    def id(self, name, add=False):
        name = name or "none"
        if name not in self.names:
            if not add:
                raise KeyError(f"未登録の素材: {name}")
            if len(self.names) >= C.MATERIAL_SLOTS:
                raise ValueError(f"素材の枠（{C.MATERIAL_SLOTS}）を超えました。config.MATERIAL_SLOTS を増やしてください")
            self.names.append(name)
        return self.names.index(name)

    def name(self, i):
        return self.names[i] if 0 <= i < len(self.names) else None


# ---------- 色 ----------

def quantize_hex(hex_color):
    """'#6fa57a' → (r, g, b) 各 0〜15"""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    v = int(h[:6], 16)
    return tuple(min(C.COLOR_LEVELS - 1, round(((v >> s) & 255) / C.COLOR_STEP)) for s in (16, 8, 0))


def dequantize(r, g, b):
    return "#" + "".join(f"{int(c) * C.COLOR_STEP:02x}" for c in (r, g, b))


# ---------- JSON ⇔ ブロック配列 ----------

def json_to_blocks(doc, materials, add_materials=False):
    """BloxToy JSON → (ブロック配列, 警告リスト)"""
    warns = []
    grid = doc.get("data", {}).get("grid", C.GRID)
    if grid != C.GRID:
        warns.append(f"grid が {grid}（想定は {C.GRID}）")
    rows, seen = [], set()
    for b in doc.get("data", {}).get("blocks", []):
        x, y, z = int(b["x"]), int(b["y"]), int(b["z"])
        if not all(0 <= v < C.GRID for v in (x, y, z)):
            warns.append(f"範囲外のブロックを除外: ({x},{y},{z})")
            continue
        if (x, y, z) in seen:
            warns.append(f"重複位置のブロックを除外: ({x},{y},{z})")
            continue
        if b.get("s", "cube") != "cube":
            warns.append(f"cube 以外の形: {b.get('s')}")
        seen.add((x, y, z))
        r, g, bl = quantize_hex(b.get("c", "#cccccc"))
        rows.append([x, y, z, r, g, bl, materials.id(b.get("m"), add=add_materials)])
    return np.array(rows, dtype=np.int16).reshape(-1, 7), warns


def blocks_to_json(blocks, materials, title="generated"):
    out = []
    for x, y, z, r, g, b, m in sort_blocks(blocks).tolist():
        blk = {"x": x, "y": y, "z": z, "c": dequantize(r, g, b), "s": "cube"}
        name = materials.name(m)
        if name and name != "none":
            blk["m"] = name
        out.append(blk)
    return {"app": "3dworld", "type": "blocks", "title": title,
            "data": {"version": 1, "grid": C.GRID, "blocks": out}}


# ---------- ブロック配列 ⇔ トークン列 ----------

def sort_blocks(blocks):
    """下から上へ：y → z → x の順"""
    if len(blocks) == 0:
        return blocks
    order = np.lexsort((blocks[:, COL["x"]], blocks[:, COL["z"]], blocks[:, COL["y"]]))
    return blocks[order]


def encode(blocks):
    b = sort_blocks(blocks).astype(np.int64)
    toks = np.empty((len(b), TOKENS_PER_BLOCK), dtype=np.int64)
    toks[:, 0] = OFFSET["p"] + pos_index(b[:, 0], b[:, 1], b[:, 2])
    toks[:, 1] = OFFSET["c"] + color_index(b[:, 3], b[:, 4], b[:, 5])
    toks[:, 2] = OFFSET["m"] + b[:, 6]
    return np.concatenate([[BOS], toks.reshape(-1), [EOS]])


def decode(tokens):
    """トークン列 → ブロック配列。壊れたブロック（番号帯の不一致・重複位置）は捨てる。"""
    toks = [int(t) for t in tokens if t not in (PAD, BOS)]
    if EOS in toks:
        toks = toks[:toks.index(EOS)]
    rows, seen = [], set()
    for i in range(0, len(toks) - TOKENS_PER_BLOCK + 1, TOKENS_PER_BLOCK):
        vals = [t - OFFSET[f] for f, t in zip(FIELDS, toks[i:i + TOKENS_PER_BLOCK])]
        if not all(0 <= v < FIELD_SIZE[f] for f, v in zip(FIELDS, vals)):
            continue
        p, c, m = vals
        if p in seen:
            continue
        seen.add(p)
        rows.append([*pos_xyz(p), *color_rgb(c), m])
    return np.array(rows, dtype=np.int16).reshape(-1, 7)
