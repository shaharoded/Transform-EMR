import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import math
import joblib
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.model_config import *


class Time2Vec(nn.Module):
    """
    Time2Vec layer for encoding continuous time intervals (deltas) into fixed-size vectors.

    This layer captures both linear trends and periodic patterns in the time between events.

    Output dimensions:
        - 1 dimension for linear time progression (trend)
        - k-1 dimensions for periodic representation using learnable sine functions

    Args:
        out_dim (int): Output dimensionality of the time embedding (must be >= 2)

    Input:
        t (Tensor): FloatTensor of shape [B, T] or [B*T], representing time deltas in days (as float representing partial days too)

    Output:
        Tensor of shape [B, T, out_dim] or [B*T, out_dim], where out_dim = 1 + k
    """

    def __init__(self, out_dim):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >= 2")
        self.linear = nn.Linear(1, 1, bias=True)          # ω0 t + b0 -> Linear component
        self.freq   = nn.Linear(1, out_dim - 1, bias=True) # ω_i t + b_i -> Sinusiodal component

    def forward(self, t):
        """
        t: [B, T] or [B*T] float tensor of time deltas
        Returns: [B, T, out_dim] or [B*T, out_dim]
        """
        t = t.unsqueeze(-1)                     # [B, T, 1]
        linear_out = self.linear(t)             # [B, T, 1]
        periodic_out = torch.sin(self.freq(t))  # [B, T, k-1]
        return torch.cat([linear_out, periodic_out], dim=-1)


class EMREmbedding(nn.Module):
    """
    Embedding layer for electronic medical record (EMR) sequences.

    This module creates time-aware, context-enhanced event representations suitable
    for Transformer-based models. It replaces traditional token and positional embeddings
    by explicitly decomposing each event into structured components:

      - Concept ID (e.g., "GLUCOSE")
      - Concept + Value ID (e.g., "GLUCOSE_Low")
      - Concept + Value + Position ID (e.g., "GLUCOSE_Low_START")
      - Relative time delta since the previous event (Δt)
      - Absolute time since admission (t_abs)
      - Patient-level context vector (e.g., age, sex, diagnosis group)

    These components are embedded, concatenated, and projected into a shared
    fixed-size embedding space. A special [CTX] token is prepended to each sequence
    to incorporate patient-level context at the start of modeling.

    All embeddings are regularized with dropout and normalized with LayerNorm.

    Args:
        concept_vocab_size (int): Number of unique clinical concepts (e.g., "GLUCOSE", "MEAL").
        value_vocab_size (int): Number of unique concept-value combinations (e.g., "GLUCOSE_High").
        position_vocab_size (int): Number of unique concept-value-position combinations (e.g., "GLUCOSE_High_START").
        ctx_dim (int): Dimensionality of the patient context vector.
        time2vec_dim (int): Output dimension of each Time2Vec component (must be ≥ 2).
        embed_dim (int): Final embedding size for each event.
        dropout (float): Dropout rate applied to the combined embeddings.

    Inputs:
        concept_ids (LongTensor): [B, T] — concept-level token IDs
        value_ids (LongTensor): [B, T] — concept+value token IDs
        position_ids (LongTensor): [B, T] — concept+value+position token IDs
        delta_ts (FloatTensor): [B, T] — relative time (Δt) since previous event (in days)
        abs_ts (FloatTensor): [B, T] — absolute time since admission (in days)
        patient_contexts (FloatTensor): [B, ctx_dim] — per-patient non-temporal features

    Output:
        embeddings (FloatTensor): [B, T+1, embed_dim] — event embeddings, prepended with [CTX]
        (optionally) attention_mask (BoolTensor): [B, T+1] — True for real tokens, False for [PAD]
    """

    def __init__(self, concept_vocab_size, value_vocab_size, position_vocab_size, ctx_dim, 
                 time2vec_dim=8, embed_dim=128, dropout=0.1):
        super().__init__()

        # --- for compatibility -------------------------------------------------
        self.padding_idx = 0 # Hard coded. Should never change.
        self.scaler = None # Place holder. Will be saved during training.
        self.output_dim = embed_dim  # keep public attr for compatibility
        self.vocab_size = position_vocab_size  # keep public attr for compatibility

        # --- Token-level embeddings ---
        self.concept_embed = nn.Embedding(concept_vocab_size, embed_dim) # Embed for "GLUCOSE"
        self.value_embed = nn.Embedding(value_vocab_size, embed_dim) # Embed for "GLUCOSE_High"
        self.position_embed = nn.Embedding(position_vocab_size, embed_dim) # Embed for "GLUCOSE_High_Start" -> the full vocab size

        # --- Time embeddings ---
        self.time2vec_rel = Time2Vec(time2vec_dim)
        self.time2vec_abs = Time2Vec(time2vec_dim)

        # --- Time projection ---
        time_cat_dim = 2 * time2vec_dim
        self.time_proj = nn.Linear(time_cat_dim, embed_dim, bias=False)

        # --- patient‑context slot ----------------------------------------
        self.ctx_token  = nn.Parameter(torch.randn(embed_dim)) # learnable [CTX] token
        self.context_proj = nn.Linear(ctx_dim, embed_dim, bias=False)

        # --- Final projection ---
        concat_dim = 4 * embed_dim  # concept + value + pos + time
        self.final_proj = nn.Linear(concat_dim, embed_dim)

        # --- regularisation ----------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)

        self.output_dim = embed_dim

        # --- Decoder tied to token embeddings ----------------------------
        self.decoder = nn.Linear(embed_dim, position_vocab_size, bias=False)
        self.decoder.weight = self.position_embed.weight  # weight tying


    def forward(self, concept_ids, value_ids, position_ids,
                delta_ts, abs_ts, patient_contexts, return_mask=False):
        """
        Build time-aware event embeddings for a full sequence.

        Args:
            concept_ids (LongTensor): [B, T]
            value_ids (LongTensor):   [B, T]
            position_ids (LongTensor):[B, T]
            delta_ts (FloatTensor):   [B, T]
            abs_ts (FloatTensor):     [B, T]
            patient_contexts (FloatTensor): [B, ctx_dim]
            return_mask (bool): Whether to return an attention mask

        Returns:
            embeddings:   [B, T+1, D] — [CTX] prepended
            attention_mask (optional): [B, T+1] (True for real tokens)
        """
        # --- Token lookups ---
        c_emb = self.concept_embed(concept_ids)     # [B, T, D]
        v_emb = self.value_embed(value_ids)
        p_emb = self.position_embed(position_ids)

        # --- Time encoding ---
        t_rel = self.time2vec_rel(delta_ts)         # [B, T, k]
        t_abs = self.time2vec_abs(abs_ts)
        t_cat = torch.cat([t_rel, t_abs], dim=-1)   # [B, T, 2k]
        t_emb = self.time_proj(t_cat)               # [B, T, D]

        # --- Combine all token-wise pieces ---
        combined = torch.cat([c_emb, v_emb, p_emb, t_emb], dim=-1)  # [B, T, 4D]
        ev_vec = self.final_proj(combined)                          # [B, T, D]
        ev_vec = self.dropout(ev_vec) / self.scale                 # [B, 1, D]

        # --- [CTX] slot ---
        ctx_vec = self.ctx_token + self.context_proj(patient_contexts)  # [B, D]
        ctx_vec = ctx_vec.unsqueeze(1)                                  # [B, 1, D]

        seq = torch.cat([ctx_vec, ev_vec], dim=1)                       # [B, T+1, D]
        seq = self.layernorm(seq)

        if return_mask:
            pad_mask = (position_ids != self.padding_idx)
            pad_mask = torch.cat([torch.ones_like(pad_mask[:, :1]), pad_mask], dim=1)
            return seq, pad_mask

        return seq
    
    def forward_with_decoder(self, concept_ids, value_ids, position_ids,
                            delta_ts, abs_ts, patient_contexts):
        """
        Runs full forward pass + decoding (for training phase 1).

        Returns:
            logits: [B, T, vocab_size] — scores for next-token prediction
        """
        seq = self.forward(
            concept_ids, value_ids, position_ids,
            delta_ts, abs_ts, patient_contexts,
            return_mask=False
        )  # [B, T+1, D]

        return self.decoder(seq[:, :-1, :])  # Predict next token at each step


def train_embedder(embedder, train_loader, val_loader, resume=True, scaler=None):
    """
    Trains an EMREmbedding model (with internal decoder) on EMR data.

    Args:
        embedder (EMREmbedding): A fully initialized embedder with decoder.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        resume (bool): Whether to resume training from ckpt_last.pt if exists.
        scaler (sklearn.preprocessing.StandardScaler): Used to scale the ctx vector. Passed here to be saved in a checkpoint.

    Returns:
        Tuple: (trained embedder, train_losses, val_losses)
    """
    # ----- Device and init setup -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = embedder.to(device)
    embedder.scaler = scaler # Save the scaler in the init.

    # ----- Checkpoint -----
    ckpt_path = Path(EMBEDDER_CHECKPOINT).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last_path = ckpt_path.parent / "ckpt_last.pt"

    # ----- Loss and optimizer -----
    loss_fn = nn.CrossEntropyLoss(ignore_index=embedder.padding_idx)
    optimizer = torch.optim.AdamW(embedder.parameters(), 
                                  lr=TRAINING_SETTINGS.get('phase1_learning_rate'))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    # ----- Optionally resume -----
    train_losses, val_losses = [], []
    best_val, bad_epochs = float("inf"), 0
    start_epoch = 1

    if resume and ckpt_last_path.exists():
        print(f"[Phase 1]: Resuming from checkpoint: {ckpt_last_path}...")
        ckpt = torch.load(ckpt_last_path, map_location=device)
        embedder.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val = ckpt.get("best_val", float("inf"))
    else:
        print(f"[Phase 1]: Starting embedder training loop...")

    # ----- Epoch loop -----
    def run_epoch(loader, train_flag=False):
        embedder.train() if train_flag else embedder.eval()
        total_loss = 0.0

        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            time_deltas = batch["time_deltas"].to(device)
            context_vec = batch["context_vec"].to(device)

            if train_flag:
                optimizer.zero_grad()

            logits = embedder.forward_with_decoder(token_ids, time_deltas, context_vec)  # [B, T, V]
            loss = loss_fn(logits.reshape(-1, embedder.vocab_size), token_ids.reshape(-1))

            if train_flag:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ----- Training loop -----
    for epoch in range(start_epoch, TRAINING_SETTINGS.get('phase1_n_epochs') + 1):
        tr_loss = run_epoch(train_loader, train_flag=True)
        vl_loss = run_epoch(val_loader, train_flag=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[Training Embedder]: Epoch {epoch:03d} | Train: {tr_loss:.4f} | Val: {vl_loss:.4f}")
        scheduler.step(vl_loss)

        # Save checkpoint (last)
        torch.save({
            "epoch": epoch,
            "model_state": embedder.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
        }, ckpt_last_path)

        # Save best
        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            torch.save(embedder.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= TRAINING_SETTINGS.get("patience"):
                print("[Phase 1]: Early stopping triggered!")
                break
    # Also save scaler from dataset for use on test set without initiatin a new one
    joblib.dump(embedder.scaler, os.path.join(ckpt_path.parent, "scaler.pkl"))
    
    return embedder, train_losses, val_losses