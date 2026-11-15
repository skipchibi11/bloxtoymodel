"""データ前処理。

使い方:
    .venv/bin/python -m src.preprocess

やること:
  1. data/*.json を読み、色を丸めてブロック配列にする
  2. 見つかった素材を materials.json に追加する（番号は追加のみ）
  3. ラベルを集める（title・ファイル名・labels.json）
  4. ラベルをテキストエンコーダでベクトルにする
  5. processed/dataset.pt に保存し、集計を表示する
"""

import json
import re
from collections import Counter

import numpy as np
import torch

from . import config as C
from .vocab import TOKENS_PER_BLOCK, VOCAB_SIZE, MaterialRegistry, json_to_blocks


def auto_labels(path, doc):
    labels = []
    title = (doc.get("title") or "").strip()
    if title:
        labels.append(title)
    stem = re.sub(r"[_-]?\d+$", "", path.stem)        # cat_01 → cat
    stem = re.sub(r"[_-]+", " ", stem).strip()         # present-box → present box
    if stem:
        labels.append(stem)
    return labels


def load_extra_labels():
    if not C.LABELS_FILE.exists():
        return {}
    d = json.loads(C.LABELS_FILE.read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def unique(seq):
    seen, out = set(), []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.TEXT_ENCODER, revision=C.TEXT_ENCODER_REVISION, device="cpu")
    vecs = model.encode([C.TEXT_PREFIX + t for t in texts], normalize_embeddings=True,
                        convert_to_tensor=True, show_progress_bar=False)
    return vecs.float().cpu()


def main():
    materials = MaterialRegistry.load()
    n_mat_before = len(materials.names)
    extra = load_extra_labels()

    files = sorted(C.DATA_DIR.glob("*.json"))
    if not files:
        print("data/ に JSON がありません")
        return

    samples, all_warns = [], []
    for path in files:
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            all_warns.append(f"{path.name}: JSON を読めません（{e}）→ スキップ")
            continue
        blocks, warns = json_to_blocks(doc, materials, add_materials=True)
        all_warns += [f"{path.name}: {w}" for w in warns]
        if len(blocks) == 0:
            all_warns.append(f"{path.name}: ブロックが 0 個 → スキップ")
            continue
        labels = unique(auto_labels(path, doc) + extra.get(path.name, []))
        samples.append({"file": path.name, "title": doc.get("title", path.stem),
                        "blocks": torch.from_numpy(blocks), "labels": labels})

    for k in extra:
        if not (C.DATA_DIR / k).exists():
            all_warns.append(f"labels.json: {k} に対応するファイルがありません")

    materials.save()

    # ラベルをまとめてベクトル化
    texts = [t for s in samples for t in s["labels"]]
    print(f"テキストエンコーダでラベル {len(texts)} 件をベクトル化中…")
    vecs = embed(texts)
    i = 0
    for s in samples:
        n = len(s["labels"])
        s["emb"] = vecs[i:i + n]
        i += n

    C.PROCESSED_DIR.mkdir(exist_ok=True)
    meta = {"grid": C.GRID, "color_levels": C.COLOR_LEVELS, "vocab_size": VOCAB_SIZE,
            "tokens_per_block": TOKENS_PER_BLOCK, "material_slots": C.MATERIAL_SLOTS,
            "materials": materials.names, "text_encoder": C.TEXT_ENCODER,
            "text_encoder_revision": C.TEXT_ENCODER_REVISION, "emb_dim": int(vecs.shape[1])}
    torch.save({"meta": meta, "samples": samples}, C.PROCESSED_DIR / "dataset.pt")

    # ---------- 集計 ----------
    n_blocks = [len(s["blocks"]) for s in samples]
    mat_count = Counter()
    for s in samples:
        for m in s["blocks"][:, 6].tolist():
            if m:
                mat_count[materials.name(m)] += 1
    print()
    print(f"素材: {len(samples)} 件 / ブロック数 {min(n_blocks)}〜{max(n_blocks)}（平均 {np.mean(n_blocks):.0f}）")
    print(f"トークン長: 最大 {max(n_blocks) * TOKENS_PER_BLOCK + 2}（語彙 {VOCAB_SIZE}）")
    new_mats = materials.names[n_mat_before:]
    print(f"素材の一覧: {materials.names}" + (f"（新規: {new_mats}）" if new_mats else ""))
    print(f"素材の使用数: {dict(mat_count) or 'なし'}")
    print()
    for s in samples:
        print(f"  {s['file']:<20} {len(s['blocks']):>4} ブロック  ラベル: {', '.join(s['labels'])}")
    if all_warns:
        print("\n⚠️ 警告:")
        for w in all_warns:
            print("  " + w)
    print(f"\n保存しました: {C.PROCESSED_DIR / 'dataset.pt'}")


if __name__ == "__main__":
    main()
