"""
transformer.py
==============

GPT wrapper that plugs into the project‑wide Time2Vec + context
embedding defined in embedding.py and the batch structure produced
by dataset.py.

Usage
-----
>>> from embedding import EMREmbedding
>>> from transformer import GPT
>>> embedder = EMREmbedding(...)
>>> gpt      = GPT(MODEL_CONFIG, embedder)
>>> logits, loss = gpt(token_ids, time_deltas, context_vec, targets)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────── project modules ─────────────────────────────────────────────────── #
from embedding import EMREmbedding
from dataset   import EMRDataset, collate_emr       # optional: needed only in training
from config.model_config import MODEL_CONFIG        # base hyper‑params

# ───────── helpers ─────────────────────────────────────────────────────────── #
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi‑head causal self‑attention (no rotary/ALiBi; same math as GPT‑2)."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]

        self.qkv   = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.proj  = nn.Linear(cfg["n_embd"], cfg["n_embd"],    bias=cfg["bias"])
        self.attn_dropout  = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])

        # pre‑built causal mask (triangular) – trimmed in forward
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"])
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=2)
        # (B, head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y   = att @ v                              # (B, head, T, head_dim)
        y   = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=cfg["bias"]),
            nn.GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"]),
            nn.Dropout(cfg["dropout"])
        )

    def forward(self, x):  return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg["n_embd"], bias=cfg["bias"])
        self.att = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg["n_embd"], bias=cfg["bias"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ───────── the GPT wrapper that consumes EMREmbedding ─────────────────────── #
class GPT(nn.Module):
    """
    GPT‑style decoder that takes an *external* EMREmbedding instead of its own
    token/positional embeddings.

    Parameters
    ----------
    cfg       : dict – hyper‑parameters (block_size, n_layer, n_head, dropout, ...)
    embedder  : EMREmbedding – fully initialised shared embedding module
    """

    def __init__(self, cfg: dict, embedder: EMREmbedding):
        super().__init__()

        # allow the config file to use 'embed_dim' instead of 'n_embd'
        if "n_embd" not in cfg and "embed_dim" in cfg:
            cfg["n_embd"] = cfg.pop("embed_dim")

        assert cfg["n_embd"] == embedder.output_dim, (
            "Config n_embd must equal EMREmbedding.output_dim"
        )

        self.cfg      = cfg
        self.embedder = embedder

        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f  = LayerNorm(cfg["n_embd"], bias=cfg["bias"])

        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.embedder.token_embed.weight  # weight tying

        self.apply(self._init_weights)
        # slightly smaller init for res projections as in gpt‑2
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

        print(f"Total params: {self.get_num_params()/1e6:.2f} M")

    # -------------------------------------------------------- helpers ------- #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay,    "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate, betas=betas)

    # ---------------------------------------------------- forward & loss ---- #
    def forward(
        self,
        token_ids,          # (B, T)
        time_deltas,        # (B, T)
        context_vec=None,   # (B, C)
        targets=None        # (B, T)
    ):
        """
        All tensors come straight from `collate_emr`:
            token_ids   – padded token ids
            time_deltas – relative start times (days)
            context_vec – age/gender or [] if not used
            targets     – same size as token_ids (for next‑token loss)
        """
        x = self.drop(self.embedder(token_ids, time_deltas, context_vec))  # (B, T+1, D)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))                                # (B, T+1, V)

        loss = None
        if targets is not None:
            # Skip [CTX] token (first position) when computing language‑model loss
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.embedder.token_embed.padding_idx
            )
        return logits, loss
