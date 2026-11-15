"""ブロック生成部：テキストのベクトルを条件にして、ブロックのトークンを1つずつ生成する Transformer。

  [条件トークン × n_prefix] [BOS] 位置 色 素材  位置 色 素材 ... [EOS]
   ↑ テキストのベクトルを変換したもの

- 系列中の場所は「何番目のブロックか」＋「ブロック内のどの項目か（位置/色/素材）」で与える
- 位置トークンの埋め込みには x・y・z それぞれの埋め込みを、色トークンには r・g・b の埋め込みを足す。
  → 「(3,5,7) と (3,5,8) は隣同士」「似た赤同士は近い」ことを最初から構造として持てる
- 出力層は同じ埋め込み表を共有する（重み共有）
- 学習中は一定の確率で条件を「なし」に置き換える。
  生成時に「条件あり」と「条件なし」の差を強調して、指示に沿った形にしやすくする（classifier-free guidance）
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .vocab import FIELD_SIZE, OFFSET, TOKENS_PER_BLOCK, VOCAB_SIZE, color_rgb, pos_xyz


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    emb_dim: int = 768        # テキストエンコーダの出力次元
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_prefix: int = 4         # 条件トークンの数
    max_blocks: int = 512     # 1作品の最大ブロック数（16³ の 1/8）
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.n_heads = cfg.n_heads
        self.p = cfg.dropout

    def forward(self, x, attn_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class BloxModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        # 構造埋め込み：x, y, z, r, g, b（値 16 = 該当なし → 0ベクトル）
        n = C.GRID
        self.part_emb = nn.ModuleList(nn.Embedding(n + 1, d, padding_idx=n) for _ in range(6))
        parts = torch.full((cfg.vocab_size, 6), n, dtype=torch.long)
        p = torch.arange(FIELD_SIZE["p"])
        parts[OFFSET["p"] + p, 0], parts[OFFSET["p"] + p, 1], parts[OFFSET["p"] + p, 2] = pos_xyz(p)
        c = torch.arange(FIELD_SIZE["c"])
        parts[OFFSET["c"] + c, 3], parts[OFFSET["c"] + c, 4], parts[OFFSET["c"] + c, 5] = color_rgb(c)
        self.register_buffer("parts", parts, persistent=False)
        self.block_emb = nn.Embedding(cfg.max_blocks + 2, d)    # 0 = BOS, 1.. = ブロック番号+1
        self.field_emb = nn.Embedding(TOKENS_PER_BLOCK + 1, d)  # 最後 = BOS
        self.cond_proj = nn.Sequential(nn.Linear(cfg.emb_dim, d), nn.GELU(), nn.Linear(d, d * cfg.n_prefix))
        self.prefix_pos = nn.Parameter(torch.zeros(cfg.n_prefix, d))
        self.null_cond = nn.Parameter(torch.zeros(cfg.emb_dim))  # 「条件なし」
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.apply(self._init)
        for e in self.part_emb:
            with torch.no_grad():
                e.weight[C.GRID].zero_()

    def emb_table(self):
        E = self.tok_emb.weight
        for k, e in enumerate(self.part_emb):
            E = E + e(self.parts[:, k])
        return E

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, cond, drop_cond=None):
        """
        tokens: (B, T)  BOS から始まるトークン列
        cond:   (B, emb_dim) テキストのベクトル
        drop_cond: (B,) bool。True の行は条件を「なし」に置き換える
        戻り値: (B, T, vocab) 各位置の「次のトークン」の予測
        """
        B, T = tokens.shape
        P = self.cfg.n_prefix
        if drop_cond is not None:
            cond = torch.where(drop_cond[:, None], self.null_cond.expand_as(cond), cond)
        prefix = self.cond_proj(cond).view(B, P, -1) + self.prefix_pos

        i = torch.arange(T, device=tokens.device)
        block_id = torch.where(i == 0, 0, (i - 1) // TOKENS_PER_BLOCK + 1).clamp(max=self.cfg.max_blocks + 1)
        field_id = torch.where(i == 0, TOKENS_PER_BLOCK, (i - 1) % TOKENS_PER_BLOCK)
        E = self.emb_table()
        x = E[tokens] + self.block_emb(block_id) + self.field_emb(field_id)
        x = torch.cat([prefix, x], dim=1)

        # 条件トークン同士は互いに見える。トークン部分は前だけを見る（因果マスク）
        L = P + T
        mask = torch.ones(L, L, dtype=torch.bool, device=tokens.device).tril()
        mask[:P, :P] = True
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln_f(x[:, P:]) @ E.t() + self.head_bias
