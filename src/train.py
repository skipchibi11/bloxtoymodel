"""学習。

使い方:
    .venv/bin/python -m src.train --steps 3000              # 新規に学習
    .venv/bin/python -m src.train --steps 2000 --resume     # 前回の続きから追加学習（素材を増やしたとき）

事前に .venv/bin/python -m src.preprocess を実行しておくこと。
"""

import argparse
import csv
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .augment import random_augment
from .model import BloxModel, ModelConfig
from .vocab import PAD, encode

CKPT_DIR = C.ROOT / "checkpoints"


def device():
    """NVIDIA GPU（クラウド）→ Mac の GPU → CPU の順に使えるものを選ぶ"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch(samples, batch_size, rng, max_blocks):
    toks, conds = [], []
    for _ in range(batch_size):
        s = samples[rng.integers(len(samples))]
        blocks = random_augment(s["blocks"].numpy(), rng)
        if len(blocks) > max_blocks:
            blocks = blocks[:max_blocks]
        toks.append(torch.from_numpy(encode(blocks)))
        conds.append(s["emb"][rng.integers(len(s["emb"]))])  # ラベルをランダムに1つ選ぶ
    return toks, conds


def micro_batches(toks, conds, token_budget):
    """長さの近いもの同士でまとめ、(件数 × 最大長) が token_budget を超えないように分ける。
    Mac の GPU は長い系列を一度にたくさん処理すると極端に遅くなるため。"""
    order = sorted(range(len(toks)), key=lambda i: len(toks[i]))
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(toks[i]) > token_budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    for g in groups:
        T = max(len(toks[i]) for i in g)
        batch = torch.full((len(g), T), PAD, dtype=torch.long)
        for j, i in enumerate(g):
            batch[j, :len(toks[i])] = toks[i]
        yield batch, torch.stack([conds[i] for i in g])


def check_compat(meta, ckpt_meta):
    for k in ("text_encoder", "text_encoder_revision", "vocab_size", "material_slots"):
        if meta[k] != ckpt_meta[k]:
            raise SystemExit(f"前回の学習と {k} が違うため追加学習できません: {ckpt_meta[k]} → {meta[k]}")
    if meta["materials"][:len(ckpt_meta["materials"])] != ckpt_meta["materials"]:
        raise SystemExit("materials.json の番号が前回から変わっています（追加のみ可）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--token-budget", type=int, default=6000, help="一度に GPU に送るトークン数の上限")
    ap.add_argument("--cond-drop", type=float, default=0.1, help="条件を「なし」にする確率")
    ap.add_argument("--resume", action="store_true", help="checkpoints/last.pt から続ける")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = torch.load(C.PROCESSED_DIR / "dataset.pt", weights_only=False)
    meta, samples = data["meta"], data["samples"]
    dev = device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    start_step = 0
    if args.resume:
        ck = torch.load(CKPT_DIR / "last.pt", weights_only=False, map_location="cpu")
        check_compat(meta, ck["meta"])
        cfg = ModelConfig(**ck["model_config"])
        model = BloxModel(cfg).to(dev)
        model.load_state_dict(ck["model"])
        start_step = ck["step"]
        print(f"前回の続きから学習します（{start_step} ステップ目から）")
    else:
        cfg = ModelConfig(emb_dim=meta["emb_dim"], vocab_size=meta["vocab_size"])
        model = BloxModel(cfg).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    if args.resume and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])

    print(f"素材 {len(samples)} 件 / モデル {model.n_params() / 1e6:.1f}M パラメータ / 端末 {dev}")
    CKPT_DIR.mkdir(exist_ok=True)
    log_path = CKPT_DIR / "train_log.csv"
    log_new = not log_path.exists() or not args.resume
    log_f = open(log_path, "w" if log_new else "a", newline="")
    log = csv.writer(log_f)
    if log_new:
        log.writerow(["step", "loss", "lr", "sec"])

    def save(step, name):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "model_config": cfg.to_dict(), "meta": meta}, CKPT_DIR / name)

    model.train()
    t0, avg = time.time(), None
    end = start_step + args.steps
    for step in range(start_step + 1, end + 1):
        # 学習率：最初はゆっくり上げ、その後なめらかに下げる（今回の学習の範囲で）
        k = step - start_step
        lr = args.lr * min(1.0, k / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * k / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr

        toks, conds = make_batch(samples, args.batch, rng, cfg.max_blocks)
        n_tok = sum(len(t) - 1 for t in toks)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for tokens, cond in micro_batches(toks, conds, args.token_budget):
            tokens, cond = tokens.to(dev), cond.to(dev)
            drop = torch.rand(len(tokens), device=dev) < args.cond_drop
            logits = model(tokens[:, :-1], cond, drop)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1),
                                   ignore_index=PAD, reduction="sum") / n_tok
            loss.backward()
            loss_sum += loss.item()
        loss = torch.tensor(loss_sum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        lv = loss.item()
        avg = lv if avg is None else 0.98 * avg + 0.02 * lv
        if step % 50 == 0 or step == end:
            el = time.time() - t0
            print(f"step {step:>6}/{end}  loss {avg:.3f}  lr {lr:.1e}  {el / (k):.2f}s/step  残り約{el / k * (end - step) / 60:.0f}分", flush=True)
            log.writerow([step, f"{avg:.4f}", f"{lr:.2e}", f"{el:.0f}"])
            log_f.flush()
        if step % args.save_every == 0 or step == end:
            save(step, "last.pt")
    log_f.close()
    print(f"保存しました: {CKPT_DIR / 'last.pt'}")


if __name__ == "__main__":
    main()
