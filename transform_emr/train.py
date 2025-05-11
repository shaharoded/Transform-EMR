"""
pre-train.py
=====================
A two-phase transformer training process:
Phase‑1 : call embedding.train()  ------------>  pretrained_embedder.pt
Phase‑2 : GPT( pretrained_embedder, fine-tuned during training )  ->  best.pt
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.dataset import EMRDataset, collate_emr
from transform_emr.embedding import EMREmbedding, train_embedder
from transform_emr.transformer import GPT
from transform_emr.utils import plot_losses
from transform_emr.config.model_config import MODEL_CONFIG, TRAINING_SETTINGS, TRANSFORMER_CHECKPOINT
from transform_emr.config.dataset_config import TRAIN_TEMPORAL_DATA_FILE, TRAIN_CTX_DATA_FILE


def prepare_data():
    print(f"[Pre-processing]: Building dataset...")
    temporal_df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE)
    ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)
    pids = temporal_df["PatientID"].unique()
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)


    train_df, val_df = temporal_df[temporal_df.PatientID.isin(train_ids)].copy(), temporal_df[temporal_df.PatientID.isin(val_ids)].copy()
    train_ctx, val_ctx = ctx_df[ctx_df.PatientID.isin(train_ids)], ctx_df[ctx_df.PatientID.isin(val_ids)]

    train_ds = EMRDataset(train_df, train_ctx)
    val_ds   = EMRDataset(val_df,  val_ctx, scaler=train_ds.scaler)    
    MODEL_CONFIG['vocab_size'] = len(set(train_ds.token2id.keys()) | set(val_ds.token2id.keys())) # Dinamically updating vocab
    MODEL_CONFIG['ctx_dim'] = train_ds.context_df.shape[1] # Dinamically updating shape

    train_dl = DataLoader(train_ds, batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=True, collate_fn=collate_emr)
    val_dl   = DataLoader(val_ds,  batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=False, collate_fn=collate_emr)
    return train_dl, val_dl, train_ds.scaler

def phase_one(train_dl, val_dl, embedder, resume=True, scaler=None):
    return train_embedder(
        embedder=embedder,
        train_loader=train_dl,
        val_loader=val_dl,
        resume=resume,
        scaler=scaler
    )

def phase_two(train_dl, val_dl, embedder, tune_embedder=True, resume=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not tune_embedder:
        for p in embedder.parameters(): p.requires_grad = False
        embedder.eval()
    else:
        embedder.train()

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

    outdir = Path(TRANSFORMER_CHECKPOINT).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load latest checkpoint if resume is requested
    ckpt_last_path = outdir / "ckpt_last.pt"
    if resume and ckpt_last_path.exists():
        ckpt = torch.load(ckpt_last_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        print(f"[Phase 2]: Resumed from checkpoint: {ckpt_last_path}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
    else:
        print(f"[Phase 2]: Starting transformer training loop...")
        start_epoch = 0
        best_val = float("inf")

    wait = 0
    patience = TRAINING_SETTINGS.get("patience", 5)
    train_losses, val_losses = [], []

    def run_epoch(loader, train_flag=False):
        model.train() if train_flag else model.eval()
        total_loss = 0.0
        with torch.set_grad_enabled(train_flag):
            for batch in tqdm(loader, desc="Training" if train_flag else "Validation", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs, loss = model(
                    token_ids=batch["token_ids"],
                    time_deltas=batch["time_deltas"],
                    context_vec=batch["context_vec"],
                    targets=batch["targets"]
                )

                if train_flag:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(start_epoch, TRAINING_SETTINGS.get("phase2_n_epochs")):
        tr_loss = run_epoch(train_dl, train_flag=True)
        vl_loss = run_epoch(val_dl, train_flag=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[Training Transformer]: Epoch {epoch:02d} | Train={tr_loss:.4f} | Val={vl_loss:.4f}")
        scheduler.step(vl_loss)

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val": best_val,
        }, outdir / "ckpt_last.pt")

        # Save best
        if vl_loss < best_val - 1e-3:
            best_val = vl_loss
            torch.save(model.state_dict(), TRANSFORMER_CHECKPOINT)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("[Phase 2]: Early stopping triggered!")
                break
    
    # --- plot loss curves
    plot_losses(train_losses, val_losses)


def run_two_phase_training():
    train_dl, val_dl, scaler = prepare_data()

    embedder = EMREmbedding(
        vocab_size=MODEL_CONFIG['vocab_size'],
        ctx_dim=MODEL_CONFIG['ctx_dim'],
        time2vec_dim=MODEL_CONFIG.get('time2vec_dim'),
        embed_dim=MODEL_CONFIG['embed_dim']
    )

    # Phase 1: will resume from ckpt internally if exists
    embedder, _, _ = phase_one(train_dl, val_dl, embedder, resume=True, scaler=scaler)

    # Phase 2: continues with the best embedder
    phase_two(train_dl, val_dl, embedder, resume=True)



# if __name__ == "__main__":
#     run_two_phase_training()
