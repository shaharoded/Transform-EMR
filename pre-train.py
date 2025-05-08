"""
pre-train.py
=====================
A two-phase transformer training process:
Phase‑1 : call embedding.train()  ------------>  pretrained_embedder.pt
Phase‑2 : GPT( pretrained_embedder, fine-tuned during training )  ->  best.pt

TO-DO: 
- Add loss monitoring here
- Why is the evaluate function seperate from the training loop - looks weird.
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from dataset import EMRDataset, collate_emr
from embedding import EMREmbedding, train_embedder
from transformer import GPT
from utils import plot_losses
from config.model_config import MODEL_CONFIG, TRAINING_SETTINGS
from config.dataset_config import TEMPORAL_DATA_FILE, CTX_DATA_FILE, STATES


def prepare_data():
    temporal_df = pd.read_csv(TEMPORAL_DATA_FILE)
    ctx_df = pd.read_csv(CTX_DATA_FILE)
    pids = temporal_df["PatientID"].unique()
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)


    train_df, val_df = temporal_df[temporal_df.PatientID.isin(train_ids)].copy(), temporal_df[temporal_df.PatientID.isin(val_ids)].copy()
    train_ctx, val_ctx = ctx_df[ctx_df.PatientID.isin(train_ids)], ctx_df[ctx_df.PatientID.isin(val_ids)]

    train_ds = EMRDataset(train_df, train_ctx, states=STATES)
    val_ds   = EMRDataset(val_df,  val_ctx,   states=STATES, scaler=train_ds.scaler)    
    MODEL_CONFIG['vocab_size'] = len(set(train_ds.token2id.keys()) | set(val_ds.token2id.keys())) # Dinamically updating vocab
    MODEL_CONFIG['ctx_dim'] = train_ds.context_df.shape[1] # Dinamically updating shape

    train_dl = DataLoader(train_ds, batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=True, collate_fn=collate_emr)
    val_dl   = DataLoader(val_ds,  batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=False, collate_fn=collate_emr)
    return train_dl, val_dl

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

def phase_one(train_ld, val_ld, embedder):
    return train_embedder(
        embedder=embedder,
        train_loader=train_ld,
        val_loader=val_ld
    )

def phase_two(train_dl, val_dl, embedder, tune_embedder=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if tune_embedder:
        embedder.train()  # let it fine-tune
    else:
        for p in embedder.parameters(): p.requires_grad = False
        embedder.eval()

    assert MODEL_CONFIG.get("embed_dim") == embedder.output_dim

    model = GPT(MODEL_CONFIG, embedder).to(device)

    # AdamW with weight decay
    optimizer = model.configure_optimizers(
        weight_decay=TRAINING_SETTINGS.get("weight_decay"),
        learning_rate=TRAINING_SETTINGS.get("phase2_learning_rate"),
        betas=(0.9, 0.95)
    )

    # Optional: Reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    best_val, wait, patience = float("inf"), 0, TRAINING_SETTINGS.get("patience", 5)
    outdir = Path("checkpoints/phase2")
    outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(TRAINING_SETTINGS.get("phase2_n_epochs")):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, loss = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        train_loss = total_loss / len(train_dl)
        val_loss = evaluate(model, val_dl, device)

        scheduler.step(val_loss)  # adjust LR

        print(f"[Training Transformer]: Epoch {epoch:02d} | Train={train_loss:.4f} | Val={val_loss:.4f}")

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
    train_dl, val_dl = prepare_data()

    embedder = EMREmbedding(
        vocab_size=MODEL_CONFIG['vocab_size'],
        ctx_dim=MODEL_CONFIG['ctx_dim'],
        time2vec_dim=MODEL_CONFIG.get('time2vec_dim'),
        embed_dim=MODEL_CONFIG['embed_dim']
    )
    embedder, _, _ = phase_one(train_dl, val_dl, embedder)
    torch.save(embedder.state_dict(), "checkpoints/phase1/best_embedder.pt")

    phase_two(train_dl, val_dl, embedder)

if __name__ == "__main__":
    run_two_phase_training()
