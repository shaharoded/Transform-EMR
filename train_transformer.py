"""
train_joint.py
==============

End‑to‑end language‑model training in **one phase**:

    ┌──────────────┐      ┌──────────────┐
    │ EMREmbedding │ ───► │   GPT body   │ ───► lm_head
    └──────────────┘      └──────────────┘

Both parts are optimised together.
"""

import torch, math, json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np
from pathlib import Path

# project modules ------------------------------------------------------------
from dataset     import EMRDataset, collate_emr
from embedding   import EMREmbedding
from transformer import GPT                       # <-- already rewritten to accept an embedder
from config.model_config   import MODEL_CONFIG as BASE_MODEL_CFG
from config.dataset_config import DATA_FILE, STATES

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def save_ckpt(model, optim, epoch, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
    }, path)

@torch.no_grad()
def evaluate(model, loader, device, pad_id):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch_t = {
            "token_ids":  batch["token_ids"].to(device),
            "time_deltas":batch["time_deltas"].to(device),
            "context_vec":batch["context_vector"].to(device),
            "targets":     batch["token_ids"].to(device),
        }
        _, loss = model(**batch_t)
        tot += loss.item() * batch_t["targets"].numel()
        n   += batch_t["targets"].numel()
    return tot / n

# --------------------------------------------------------------------------- #
# 1.  data prep                                                               #
# --------------------------------------------------------------------------- #
NUMERIC_CONC = STATES
CTX_COLS     = ["age", "gender"]         # simple synthetic demographics
PAD_ID       = 0

raw = pd.read_csv(DATA_FILE)
pids = raw["PatientID"].unique()
train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)

rng = np.random.default_rng(42)
ctx_df = pd.DataFrame({
    "PatientID": pids,
    "age":    rng.integers(18, 66, size=len(pids)),
    "gender": rng.integers(0,  2,  size=len(pids)),   # 0=male 1=female
})

train_df  = raw[raw.PatientID.isin(train_ids)]
val_df    = raw[raw.PatientID.isin(val_ids)]
train_ctx = ctx_df[ctx_df.PatientID.isin(train_ids)]
val_ctx   = ctx_df[ctx_df.PatientID.isin(val_ids)]

train_ds = EMRDataset(train_df, train_ctx, states=NUMERIC_CONC,
                      context_columns=CTX_COLS)
val_ds   = EMRDataset(val_df,   val_ctx,   states=NUMERIC_CONC,
                      context_columns=CTX_COLS)

train_ld = DataLoader(train_ds, batch_size=16, shuffle=True,
                      collate_fn=collate_emr)
val_ld   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                      collate_fn=collate_emr)

# --------------------------------------------------------------------------- #
# 2.  build embedder + GPT wrapper                                            #
# --------------------------------------------------------------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

embed_dim = BASE_MODEL_CFG.get("embed_dim", BASE_MODEL_CFG.get("n_embd"))
embedder  = EMREmbedding(
    token_vocab_size = len(train_ds.token2id),
    ctx_dim          = len(CTX_COLS),
    time2vec_dim     = 8,
    embed_dim        = embed_dim,
).to(device)

MODEL_CONFIG           = BASE_MODEL_CFG.copy()               # don't mutate import
MODEL_CONFIG["vocab_size"] = len(train_ds.token2id)
MODEL_CONFIG["block_size"] = max(train_ds.tokens_df.groupby("PatientID").size().max(),
                                 val_ds.tokens_df.groupby("PatientID").size().max()) + 1
if "n_embd" not in MODEL_CONFIG:
    MODEL_CONFIG["n_embd"] = MODEL_CONFIG.pop("embed_dim")

assert MODEL_CONFIG["n_embd"] == embedder.output_dim, \
       "embed_dim mismatch between model_config and EMREmbedding"

model = GPT(MODEL_CONFIG, embedder).to(device)

# optimiser ------------------------------------------------------------------
optim = model.configure_optimizers(
    weight_decay = 1e-2,
    learning_rate= 3e-4,
    betas        = (0.9, 0.95),
)

# --------------------------------------------------------------------------- #
# 3.  joint training loop                                                     #
# --------------------------------------------------------------------------- #
OUTDIR = Path("checkpoints/joint"); OUTDIR.mkdir(parents=True, exist_ok=True)
best_val, patience, wait = float("inf"), 5, 0
EPOCHS  = 40

for epoch in range(EPOCHS):
    # --- train --------------------------------------------------------------
    model.train()
    run_loss = 0.0
    for batch in train_ld:
        batch_t = {
            "token_ids":  batch["token_ids"].to(device),
            "time_deltas":batch["time_deltas"].to(device),
            "context_vec":batch["context_vector"].to(device),
            "targets":     batch["token_ids"].to(device),   # next‑token target
        }
        _, loss = model(**batch_t)
        loss.backward()
        optim.step(); optim.zero_grad()
        run_loss += loss.item()

    tr_loss = run_loss / len(train_ld)

    # --- val ---------------------------------------------------------------
    val_loss = evaluate(model, val_ld, device, PAD_ID)
    print(f"[joint] epoch {epoch:02d}  train={tr_loss:.4f}  val={val_loss:.4f}")

    # --- checkpoint & early‑stop ------------------------------------------
    save_ckpt(model, optim, epoch, OUTDIR / f"ckpt_epoch{epoch:02d}.pt")
    if val_loss < best_val - 1e-3:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), OUTDIR / "best.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early‑stopping triggered.")
            break

print("Joint pre‑training finished.  Best val loss:", best_val)
