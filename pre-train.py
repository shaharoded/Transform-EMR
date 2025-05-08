"""
pre-train.py
=====================
A two-phase transformer training process:
Phase‑1 : call embedding.train()  ------------>  pretrained_embedder.pt
Phase‑2 : GPT( pretrained_embedder, frozen )  ->  best.pt
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from dataset import EMRDataset, collate_emr
from embedding import EMREmbedding, train_embedder
from transformer import GPT
from config.model_config import MODEL_CONFIG, TRAINING_SETTINGS
from config.dataset_config import DATA_FILE, STATES

CTX_COLS = ["age", "gender"]

def prepare_data():
    raw = pd.read_csv(DATA_FILE)
    pids = raw["PatientID"].unique()
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)

    ctx_df = pd.DataFrame({
        "PatientID": pids,
        "age": np.random.randint(18, 66, size=len(pids)),
        "gender": np.random.choice([0, 1], size=len(pids)),
    })

    train_df, val_df = raw[raw.PatientID.isin(train_ids)].copy(), raw[raw.PatientID.isin(val_ids)].copy()
    train_ctx, val_ctx = ctx_df[ctx_df.PatientID.isin(train_ids)], ctx_df[ctx_df.PatientID.isin(val_ids)]

    train_ds = EMRDataset(train_df, train_ctx, states=STATES, context_columns=CTX_COLS)
    val_ds   = EMRDataset(val_df,  val_ctx,   states=STATES, context_columns=CTX_COLS)
    MODEL_CONFIG['vocab_size'] = len(set(train_ds.token2id.keys()) | set(val_ds.token2id.keys())) # Dinamically updating vocab

    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_emr)
    val_ld   = DataLoader(val_ds,  batch_size=16, shuffle=False, collate_fn=collate_emr)
    return train_ds, val_ds, train_ld, val_ld

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        _, loss = model(**b)
        total += loss.item() * b["targets"].numel()
        count += b["targets"].numel()
    return total / count

def phase_one(train_ld, val_ld, embedder, device):
    return train_embedder(
        embedder=embedder,
        train_loader=train_ld,
        val_loader=val_ld,
        vocab_size=MODEL_CONFIG.get('vocab_size'),
        lr=TRAINING_SETTINGS.get('learning_rate'),
        n_epochs=TRAINING_SETTINGS.get('learning_rate'),
        patience=TRAINING_SETTINGS.get('patience'),
        device=device
    )

def phase_two(train_ld, val_ld, train_ds, val_ds, embedder, device):
    for p in embedder.parameters(): p.requires_grad = False
    embedder.eval()

    model_cfg = MODEL_CONFIG.copy()
    model_cfg["vocab_size"] = len(train_ds.token2id)
    model_cfg["block_size"] = max(train_ds.max_len, val_ds.max_len) + 1
    model_cfg["n_embd"] = model_cfg.get("n_embd", model_cfg.get("embed_dim"))
    assert model_cfg["n_embd"] == embedder.output_dim

    model = GPT(model_cfg, embedder).to(device)
    optimizer = model.configure_optimizers(1e-2, 3e-4, (0.9, 0.95))

    best_val, wait, patience = float("inf"), 0, 5
    outdir = Path("checkpoints/phase2")
    outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for batch in train_ld:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, loss = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        train_loss = total_loss / len(train_ld)

        val_loss = evaluate(model, val_ld, device)
        print(f"[Training Transformer]: Epoch {epoch:02d}  Train={train_loss:.4f}  Val={val_loss:.4f}")

        torch.save({"model_state": model.state_dict(), "optim_state": optimizer.state_dict()}, outdir / f"ckpt_{epoch:02d}.pt")
        if val_loss < best_val - 1e-3:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), outdir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                print("[Training Transformer]: Early stopping in phase 2!")
                break

def run_two_phase_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, train_ld, val_ld = prepare_data()

    embedder, decoder, _, _ = phase_one(train_ld, val_ld, len(train_ds.token2id), len(CTX_COLS), device)
    torch.save(embedder.state_dict(), "checkpoints/phase1/best_embedder.pt")

    phase_two(train_ld, val_ld, train_ds, val_ds, embedder, device)

if __name__ == "__main__":
    run_two_phase_training()
