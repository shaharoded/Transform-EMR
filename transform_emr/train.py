"""
pre-train.py
=====================
A two-phase transformer training process:
Phase‑1 : call embedding.train()  ------------>  pretrained_embedder.pt
Phase‑2 : GPT( pretrained_embedder, fine-tuned during training )  ->  best.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr
from transform_emr.embedder import EMREmbedding, train_embedder
from transform_emr.transformer import GPT
from transform_emr.utils import *
from transform_emr.config.model_config import *
from transform_emr.config.dataset_config import TRAIN_TEMPORAL_DATA_FILE, TRAIN_CTX_DATA_FILE

def summarize_patient_data_split(train_ds, val_ds, train_ids, val_ids, tokenizer):
    """
    Prints summary statistics about your train/val split:
    - Patient counts
    - Record counts
    - Context shapes
    - Event count per patient (min/max/avg)
    - Token coverage (raw, concept, value, position)
    """

    print("✅ Data Split Summary")
    print(f"  - Train patients: {len(train_ids)}")
    print(f"  - Val patients:   {len(val_ids)}")

    print(f"  - Train records:  {len(train_ds.tokens_df):,}")
    print(f"  - Val records:    {len(val_ds.tokens_df):,}")

    # Per-patient record count stats
    train_counts = train_ds.tokens_df.groupby('PatientID').size()
    val_counts = val_ds.tokens_df.groupby('PatientID').size()

    print(f"\n📊 Train patient records:")
    print(f"  - Min:     {train_counts.min()}")
    print(f"  - Max:     {train_counts.max()}")
    print(f"  - Mean:    {train_counts.mean():.1f}")
    print(f"  - Median:  {train_counts.median()}")

    print(f"\n📊 Val patient records:")
    print(f"  - Min:     {val_counts.min()}")
    print(f"  - Max:     {val_counts.max()}")
    print(f"  - Mean:    {val_counts.mean():.1f}")
    print(f"  - Median:  {val_counts.median()}")

    # Token vocab sizes (from tokenizer)
    print(f"\n🧠 Vocabulary sizes:")
    print(f"  - Raw concepts:     {len(tokenizer.rawconcept2id):,}")
    print(f"  - Concepts:         {len(tokenizer.concept2id):,}")
    print(f"  - Concept+Value:    {len(tokenizer.value2id):,}")
    print(f"  - Full Tokens:      {len(tokenizer.token2id):,}")


def prepare_data():
    print(f"[Pre-processing]: Reading dataset...")
    temporal_df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)

    print(f"[Pre-processing]: Building tokenizer...")
    processor = DataProcessor(temporal_df, ctx_df, scaler=None)
    temporal_df, ctx_df = processor.run()

    tokenizer = EMRTokenizer.from_processed_df(temporal_df)
    tokenizer.save()

    print(f"[Pre-processing]: Building dataset...")
    pids = temporal_df["PatientID"].unique()
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)

    train_df, val_df = temporal_df[temporal_df.PatientID.isin(train_ids)].copy(), temporal_df[temporal_df.PatientID.isin(val_ids)].copy()
    train_ctx, val_ctx = ctx_df.loc[ctx_df.index.isin(train_ids)], ctx_df.loc[ctx_df.index.isin(val_ids)]

    train_ds = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
    val_ds   = EMRDataset(val_df, val_ctx, tokenizer=tokenizer)
    
    summarize_patient_data_split(train_ds, val_ds, train_ids, val_ids, tokenizer)   

    MODEL_CONFIG['ctx_dim'] = train_ds.context_df.shape[1] # Dinamically updating shape

    train_dl = DataLoader(train_ds, batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=True, collate_fn=collate_emr)
    val_dl   = DataLoader(val_ds,  batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=False, collate_fn=collate_emr)
    return train_dl, val_dl, tokenizer

def phase_one(train_dl, val_dl, embedder, resume=True):
    return train_embedder(
        embedder=embedder,
        train_loader=train_dl,
        val_loader=val_dl,
        resume=resume
    )

def phase_two(train_dl, val_dl, embedder, tune_embedder=True, resume=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not tune_embedder:
        for p in embedder.parameters(): p.requires_grad = False
        embedder.eval()
    else:
        embedder.train()

    model = GPT(MODEL_CONFIG, embedder=embedder).to(device)

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
                logits = model(
                    raw_concept_ids=batch["raw_concept_ids"],
                    concept_ids=batch["concept_ids"],
                    value_ids=batch["value_ids"],
                    position_ids=batch["position_ids"],
                    delta_ts=batch["delta_ts"],
                    abs_ts=batch["abs_ts"],
                    context_vec=batch["context_vec"]
                )

                # Multi-hot targets
                multi_hot = get_multi_hot_targets(
                    position_ids=batch["targets"],
                    padding_idx=model.embedder.padding_idx,
                    vocab_size=logits.size(-1),
                    k=TRAINING_SETTINGS["k_window"]
                )

                # Main loss: BCE with logits
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=model.embedder.tokenizer.token_weights.to(logits.device))
                loss = loss_fn(logits[:, 1:], multi_hot) # [B, T, V] vs. [B, T, V]

                # Get predicted token IDs
                pred_ids = logits[:, :-1].argmax(dim=-1)              # [B, T]
                target_ids = batch["targets"][:, 1:]                  # [B, T]

                # Load penalties
                penalty = 0.0
                penalty += penalty_meal_order(pred_ids, model.embedder.tokenizer.id2token)
                penalty += penalty_hallucinated_intervals(pred_ids, target_ids, model.embedder.tokenizer.id2token)
                penalty += penalty_false_positives(
                    predictions=torch.sigmoid(logits[:, :-1]),
                    targets=multi_hot[:, :-1],
                    token_weights=model.embedder.tokenizer.token_weights,
                    important_token_ids=model.embedder.tokenizer.important_token_ids
                )

                # Combine with weighted penalty
                loss = loss + TRAINING_SETTINGS.get("penalty_weight", 1.0) * penalty

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
            "model_state": model.state_dict(),
            "model_config": MODEL_CONFIG,
            "epoch": epoch,
            "optim_state": optimizer.state_dict(),
            "best_val": best_val,
        }, TRANSFORMER_CHECKPOINT)

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
    train_dl, val_dl, tokenizer = prepare_data()

    # Initiate empty
    embedder = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=MODEL_CONFIG.get("ctx_dim"),
        time2vec_dim=MODEL_CONFIG.get("time2vec_dim"),
        embed_dim=MODEL_CONFIG.get("embed_dim")
    )

    # Phase 1: will resume from ckpt internally if exists
    embedder, _, _ = phase_one(train_dl, val_dl, embedder=embedder, resume=True)

    # Phase 2: continues with the best embedder
    phase_two(train_dl, val_dl, embedder, tune_embedder=True, resume=True)



if __name__ == "__main__":
    train_dl, val_dl, tokenizer = prepare_data()

    # Initiate empty
    embedder = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=MODEL_CONFIG.get("ctx_dim"),
        time2vec_dim=MODEL_CONFIG.get("time2vec_dim"),
        embed_dim=MODEL_CONFIG.get("embed_dim")
    )

    # Phase 1: will resume from ckpt internally if exists
    embedder, _, _ = phase_one(train_dl, val_dl, embedder=embedder, resume=True)

    # # Phase 2: continues with the best embedder
    # phase_two(train_dl, val_dl, embedder, tune_embedder=True, resume=True)
