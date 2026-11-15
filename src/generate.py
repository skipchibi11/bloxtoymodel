"""テキストからブロックモデルを生成して outputs/ に保存する。

使い方:
    .venv/bin/python -m src.generate ネズミ
    .venv/bin/python -m src.generate "赤いバラ" -n 3 --temp 0.9 --guidance 3

生成時のルール（文法マスク）で、必ず正しい BloxToy JSON になるようにしている:
  - 各位置で、その項目（位置/色/素材）に使える番号だけを選ぶ
  - ブロックは下から上への順番を守る → 同じ位置に2個置かれることがない
  - 素材は登録済みのものだけ
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from . import config as C
from .model import BloxModel, ModelConfig
from .train import CKPT_DIR, device
from .vocab import BOS, EOS, FIELD_SIZE, FIELDS, OFFSET, TOKENS_PER_BLOCK, MaterialRegistry, blocks_to_json, decode


def allowed_mask(field, prev_pos, n_blocks, max_blocks, n_materials, vocab):
    """次のトークンとして選んでよい番号（bool 配列）。prev_pos: 直前のブロックの位置番号 または -1"""
    m = torch.zeros(vocab, dtype=torch.bool)
    f = FIELDS[field]
    if f == "p":
        if n_blocks > 0:
            m[EOS] = True
        if n_blocks < max_blocks:
            m[OFFSET["p"] + prev_pos + 1:OFFSET["p"] + FIELD_SIZE["p"]] = True  # 下から上への順番を守る
    elif f == "m":
        m[OFFSET["m"]:OFFSET["m"] + n_materials] = True
    else:
        m[OFFSET[f]:OFFSET[f] + FIELD_SIZE[f]] = True
    return m


@torch.no_grad()
def sample(model, cond, n, n_materials, temp=1.0, top_p=0.95, guidance=3.0, max_blocks=None, dev="cpu"):
    cfg = model.cfg
    max_blocks = max_blocks or cfg.max_blocks
    cond = cond.to(dev).expand(n, -1)
    seqs = torch.full((n, 1), BOS, dtype=torch.long, device=dev)
    done = [False] * n
    prev = [-1] * n
    n_blocks = [0] * n

    for step in range(max_blocks * TOKENS_PER_BLOCK + 1):
        field = step % TOKENS_PER_BLOCK
        if guidance != 1.0:
            logits = model(seqs.repeat(2, 1), torch.cat([cond, cond]),
                           torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool)]).to(dev))[:, -1]
            lc, lu = logits[:n], logits[n:]
            logits = lu + guidance * (lc - lu)
        else:
            logits = model(seqs, cond)[:, -1]
        logits = logits.float().cpu() / temp

        nxt = []
        for i in range(n):
            if done[i]:
                nxt.append(EOS)
                continue
            mask = allowed_mask(field, prev[i], n_blocks[i], max_blocks, n_materials, logits.size(-1))
            lg = logits[i].masked_fill(~mask, float("-inf"))
            probs = torch.softmax(lg, -1)
            sp, si = probs.sort(descending=True)
            keep = sp.cumsum(0) - sp < top_p
            sp = sp * keep
            t = int(si[torch.multinomial(sp / sp.sum(), 1)])
            nxt.append(t)
            if t == EOS:
                done[i] = True
                continue
            if FIELDS[field] == "p":
                prev[i] = t - OFFSET["p"]
                n_blocks[i] += 1
        seqs = torch.cat([seqs, torch.tensor(nxt, device=dev)[:, None]], dim=1)
        if all(done):
            break
    return [decode(s.tolist()) for s in seqs.cpu()]


class Generator:
    """モデルとテキストエンコーダを一度だけ読み込んで使い回す（ビューアのサーバーからも使う）。
    チェックポイントが更新されたら（追加学習の後など）自動で読み直す。"""

    def __init__(self, ckpt=CKPT_DIR / "last.pt"):
        self.ckpt = Path(ckpt)
        self.dev = device()
        self._mtime = None
        self.encoder = None

    def _load(self):
        mtime = self.ckpt.stat().st_mtime
        if mtime == self._mtime:
            return
        ck = torch.load(self.ckpt, weights_only=False, map_location="cpu")
        meta = ck["meta"]
        if meta["text_encoder_revision"] != C.TEXT_ENCODER_REVISION:
            raise RuntimeError("チェックポイントとテキストエンコーダの版が違います")
        model = BloxModel(ModelConfig(**ck["model_config"])).to(self.dev).eval()
        model.load_state_dict(ck["model"])
        self.model, self.step = model, ck["step"]
        self.materials = MaterialRegistry(meta["materials"])
        self._mtime = mtime

    def embed(self, text):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(C.TEXT_ENCODER, revision=C.TEXT_ENCODER_REVISION, device="cpu")
        return self.encoder.encode([C.TEXT_PREFIX + text], normalize_embeddings=True,
                                   convert_to_tensor=True).float().cpu()

    def generate(self, text, n=2, temp=0.7, top_p=0.95, guidance=3.0, seed=None):
        """生成して outputs/ に保存する。戻り値: [(保存先パス, ブロック数), ...]"""
        if not self.ckpt.exists():
            raise RuntimeError("学習済みモデルがありません。先に src.train を実行してください")
        self._load()
        if seed is not None:
            torch.manual_seed(seed)
        results = sample(self.model, self.embed(text), n, len(self.materials.names),
                         temp, top_p, guidance, dev=self.dev)
        C.OUTPUTS_DIR.mkdir(exist_ok=True)
        slug = re.sub(r"[^\w]+", "_", text).strip("_")[:30] or "gen"
        stamp = time.strftime("%m%d-%H%M%S")
        saved = []
        for k, blocks in enumerate(results, 1):
            out = C.OUTPUTS_DIR / f"gen_{slug}_{stamp}_{k}.json"
            out.write_text(json.dumps(blocks_to_json(blocks, self.materials, f"{text} #{k}"),
                                      ensure_ascii=False, indent=1))
            saved.append((out, len(blocks)))
        return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-n", type=int, default=2, help="作る個数")
    ap.add_argument("--temp", type=float, default=0.7, help="ランダムさ（小さいほど無難）")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--guidance", type=float, default=3.0, help="指示に従う強さ（1 = 強調なし）")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "last.pt"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    gen = Generator(args.ckpt)
    t0 = time.time()
    try:
        saved = gen.generate(args.text, args.n, args.temp, args.top_p, args.guidance, args.seed)
    except RuntimeError as e:
        raise SystemExit(str(e))
    for out, nb in saved:
        print(f"{out.relative_to(C.ROOT)}  （{nb} ブロック）")
    print(f"生成時間 {time.time() - t0:.0f} 秒（学習 {gen.step} ステップのモデル）")


if __name__ == "__main__":
    main()
