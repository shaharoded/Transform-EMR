"""
two_phase_pretrain.py
=====================
Phase‑1 : call embedding.train()  ------------>  pretrained_embedder.pt
Phase‑2 : GPT( pretrained_embedder, frozen )  ->  best.pt
"""

import json, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np
from pathlib import Path

# --- local modules ---------------------------------------------------------
from dataset   import EMRDataset, collate_emr          # your existing dataset.py
from embedding import EMREmbedding, train as train_embedder
from transformer import GPT                           # transformer.py (after the tweaks)
from config.model_config import MODEL_CONFIG as BASE_MODEL_CFG

# ---------------------------------------------------------------------------#
# 0.  helpers                                                                #
# ---------------------------------------------------------------------------#
def save_ckpt(model, optim, epoch, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict()}, path)

@torch.no_grad()
def evaluate(model, loader, device, pad_id):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        _, loss = model(**b)
        tot += loss.item() * b["targets"].numel()
        n   += b["targets"].numel()
    return tot / n

# ---------------------------------------------------------------------------#
# 1.  DATA (comes from dataset_config.py)                                     #
# ---------------------------------------------------------------------------#
from config.dataset_config import DATA_FILE, STATES   

NUMERIC_CONC = STATES            # rename for readability inside this file
CTX_COLS     = ["age", "gender"] # still generating these synthetically
PAD_ID       = 0

raw = pd.read_csv(DATA_FILE)
pids = raw["PatientID"].unique()
train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)

# random patient demographics (only for demo)
ctx_df = pd.DataFrame({
    "PatientID": pids,
    "age": np.random.randint(18, 66, size=len(pids)),
    "gender": np.random.choice([0, 1], size=len(pids)),
})

train_df = raw[raw.PatientID.isin(train_ids)].copy()
val_df   = raw[raw.PatientID.isin(val_ids)].copy()
train_ctx = ctx_df[ctx_df.PatientID.isin(train_ids)]
val_ctx   = ctx_df[ctx_df.PatientID.isin(val_ids)]

train_ds  = EMRDataset(train_df, train_ctx, numeric_concepts=NUMERIC_CONC,
                       context_columns=CTX_COLS)
val_ds    = EMRDataset(val_df,   val_ctx,   numeric_concepts=NUMERIC_CONC,
                       context_columns=CTX_COLS)

train_ld = DataLoader(train_ds, batch_size=16, shuffle=True,
                      collate_fn=collate_emr)
val_ld   = DataLoader(val_ds,  batch_size=16, shuffle=False,
                      collate_fn=collate_emr)

# ---------------------------------------------------------------------------#
# 2.  PHASE‑1  – use embedding.train() exactly as‑is                          #
# ---------------------------------------------------------------------------#
device = "cuda" if torch.cuda.is_available() else "cpu"

embedder, decoder, tr_losses, v_losses = train_embedder(
    train_loader = train_ld,
    val_loader   = val_ld,
    vocab_size   = len(train_ds.token2id),
    ctx_dim      = len(CTX_COLS),
    time2vec_dim = 8,
    embed_dim    = 128,
    lr           = 1e-4,
    n_epochs     = 150,
    patience     = 5,
    pad_token_id = PAD_ID,
    device       = device,
)

BEST_EMB_PATH = Path("checkpoints/phase1/best_embedder.pt")
BEST_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(embedder.state_dict(), BEST_EMB_PATH)

# ---------------------------------------------------------------------------#
# 3.  PHASE‑2  – Transformer on top (embedder frozen)                         #
# ---------------------------------------------------------------------------#
# freeze embedder
for p in embedder.parameters():
    p.requires_grad = False
embedder.eval()                        # just in case

MODEL_CONFIG = BASE_MODEL_CFG.copy()           # don’t mutate the import

# auto‑fill fields that depend on data
MODEL_CONFIG["vocab_size"] = len(train_ds.token2id)
MODEL_CONFIG["block_size"] = max(train_ds.max_len,
                                 val_ds.max_len) + 1        # +[CTX]

# keep n_embd consistent
if "n_embd" not in MODEL_CONFIG:
    MODEL_CONFIG["n_embd"] = MODEL_CONFIG.pop("embed_dim")
assert MODEL_CONFIG["n_embd"] == embedder.output_dim, \
       "embed_dim in config must match EMREmbedding.embed_dim"
       
model = GPT(MODEL_CONFIG, embedder).to(device)
optim = model.configure_optimizers(
    weight_decay = 1e-2,
    learning_rate= 3e-4,
    betas        = (0.9, 0.95),
    device_type  = "cuda" if device=="cuda" else "cpu"
)

best_val, patience, wait = float("inf"), 5, 0
OUTDIR = Path("checkpoints/phase2"); OUTDIR.mkdir(parents=True, exist_ok=True)

for epoch in range(30):
    # ---- train ------------------------------------------------------------
    model.train(); running = 0.0
    for batch in train_ld:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, loss = model(**batch)
        loss.backward()
        optim.step(); optim.zero_grad()
        running += loss.item()
    tr_loss = running / len(train_ld)

    # ---- val --------------------------------------------------------------
    val_loss = evaluate(model, val_ld, device, PAD_ID)
    print(f"[PH‑2] epoch {epoch:02d}  train={tr_loss:.4f}  val={val_loss:.4f}")

    save_ckpt(model, optim, epoch, OUTDIR / f"ckpt_epoch{epoch:02d}.pt")
    if val_loss < best_val - 1e-3:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), OUTDIR / "best.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early‑stop in phase 2.")
            break

print("Two‑phase pre‑training completed.")
