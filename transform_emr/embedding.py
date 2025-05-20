import os
import torch
import torch.nn as nn
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
      - Raw Concept ID (e.g., "GLUCOSE")
      - Concept ID (e.g., "GLUCOSE_STATE")
      - Concept + Value ID (e.g., "GLUCOSE_STATE_Low")
      - Concept + Value + Position ID (e.g., "GLUCOSE_STATE_Low_START")
      - Relative time delta since the previous event (Δt)
      - Absolute time since admission (t_abs)
      - Patient-level context vector (e.g., age, sex, diagnosis group)

    These components are embedded, concatenated, and projected into a shared
    fixed-size embedding space. A special [CTX] token is prepended to each sequence
    to incorporate patient-level context at the start of modeling.

    All embeddings are regularized with dropout and normalized with LayerNorm.

    Args:
        concept_vocab_size (int): Number of unique clinical concepts (e.g., "GLUCOSE", "MEAL").
        concept_vocab_size (int): Number of unique clinical concepts (e.g., "GLUCOSE_STATE").
        value_vocab_size (int): Number of unique concept-value combinations (e.g., "GLUCOSE_STATE_High").
        position_vocab_size (int): Number of unique concept-value-position combinations (e.g., "GLUCOSE_STATE_High_START").
        ctx_dim (int): Dimensionality of the patient context vector.
        time2vec_dim (int): Output dimension of each Time2Vec component (must be ≥ 2).
        embed_dim (int): Final embedding size for each event.
        dropout (float): Dropout rate applied to the combined embeddings.

    Inputs:
        raw_ids (LongTensor): [B, T] — raw-concept-level token IDs
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

    def __init__(self, raw_concept_vocab_size, concept_vocab_size, value_vocab_size, position_vocab_size, ctx_dim, 
                 time2vec_dim=8, embed_dim=128, dropout=0.1):
        super().__init__()

        # --- for compatibility -------------------------------------------------
        self.padding_idx = 0 # Hard coded. Should never change.
        self.scaler = None # Place holder. Will be saved during training.
        self.output_dim = embed_dim  # keep public attr for compatibility
        self.vocab_size = position_vocab_size  # keep public attr for compatibility
        self.token2id = {} # keep public attr for compatibility

        # --- Token-level embeddings ---
        self.raw_concept_embed = nn.Embedding(raw_concept_vocab_size, embed_dim) # Embed for "GLUCOSE_MEASURE"
        self.concept_embed = nn.Embedding(concept_vocab_size, embed_dim) # Embed for "GLUCOSE_MEASURE_STATE"
        self.value_embed = nn.Embedding(value_vocab_size, embed_dim) # Embed for "GLUCOSE_MEASURE_STATE_High"
        self.position_embed = nn.Embedding(position_vocab_size, embed_dim) # Embed for "GLUCOSE_MEASURE_STATE_High_Start" -> the full vocab size

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
        concat_dim = 5 * embed_dim  # concept + value + pos + time
        self.final_proj = nn.Linear(concat_dim, embed_dim)

        # --- regularisation ----------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)

        self.output_dim = embed_dim

        # --- Decoder tied to token embeddings ----------------------------
        self.decoder = nn.Linear(embed_dim, position_vocab_size, bias=False)
        self.decoder.weight = self.position_embed.weight  # weight tying


    def forward(self, raw_concept_ids, concept_ids, value_ids, position_ids,
                delta_ts, abs_ts, patient_contexts, return_mask=False):
        """
        Build time-aware event embeddings for a full sequence.

        Args:
            raw_concept_ids (LongTensor): [B, T]
            concept_ids (LongTensor):     [B, T]
            value_ids (LongTensor):       [B, T]
            position_ids (LongTensor):    [B, T]
            delta_ts (FloatTensor):       [B, T]
            abs_ts (FloatTensor):         [B, T]
            patient_contexts (FloatTensor): [B, ctx_dim]
            return_mask (bool): Whether to return an attention mask

        Returns:
            embeddings:   [B, T+1, D] — [CTX] prepended
            attention_mask (optional): [B, T+1] (True for real tokens)
        """
        # --- Token lookups ---
        r_emb = self.raw_concept_embed(raw_concept_ids)     # [B, T, D]
        c_emb = self.concept_embed(concept_ids)     # [B, T, D]
        v_emb = self.value_embed(value_ids)
        p_emb = self.position_embed(position_ids)

        # --- Time encoding ---
        t_rel = self.time2vec_rel(delta_ts)         # [B, T, k]
        t_abs = self.time2vec_abs(abs_ts)
        t_cat = torch.cat([t_rel, t_abs], dim=-1)   # [B, T, 2k]
        t_emb = self.time_proj(t_cat)               # [B, T, D]

        # --- Combine all token-wise pieces ---
        combined = torch.cat([r_emb, c_emb, v_emb, p_emb, t_emb], dim=-1)  # [B, T, 5D]
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
    
    def forward_with_decoder(self, raw_concept_ids, concept_ids, value_ids, position_ids,
                            delta_ts, abs_ts, patient_contexts):
        """
        Runs full forward pass + decoding (for training phase 1).

        Returns:
            logits: [B, T, vocab_size] — scores for next-token prediction
        """
        seq = self.forward(
            raw_concept_ids, concept_ids, value_ids, position_ids,
            delta_ts, abs_ts, patient_contexts,
            return_mask=False
        )  # [B, T+1, D]

        return self.decoder(seq[:, :-1, :])  # Predict next token at each step


def train_embedder(embedder, train_loader, val_loader, resume=True, scaler=None, token_weights=None, token2id={}, window_k=5):
    """
    Trains an EMREmbedding model using weighted k-step prediction loss, to allow for a softer loss penalty.
    IDEA: The exact order of the token is not really important, only the existance of important tokens and patterns.

    Args:
        embedder (EMREmbedding): The embedding model with decoder.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        resume (bool): Resume from last checkpoint if available.
        scaler (StandardScaler): Context scaler to persist in checkpoint.
        token_weights (Tensor): Per-token loss weights.
        token2id (Dict): Mapping from token to it's ID, for inference from the embedder alone.
        window_k (int): Number of next tokens to predict jointly.

    Returns:
        Tuple: (trained model, train_losses, val_losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder.to(device)
    embedder.scaler = scaler
    embedder.token2id = token2id

    ckpt_path = Path(EMBEDDER_CHECKPOINT).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    # ----- Loss & Optimizer -----
    if token_weights is not None:
        token_weights = token_weights.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=token_weights)
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=TRAINING_SETTINGS["phase1_learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    # ----- Resume logic -----
    train_losses, val_losses = [], []
    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0

    if resume and ckpt_last.exists():
        print(f"[Phase 1] Resuming from checkpoint: {ckpt_last}")
        ckpt = torch.load(ckpt_last, map_location=device)
        embedder.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1

    # ----- Epoch function -----
    def run_epoch(loader, train=False):
        embedder.train() if train else embedder.eval()
        total_loss = 0

        for batch in loader:
            raw_concept_ids = batch["raw_concept_ids"].to(device)
            concept_ids = batch["concept_ids"].to(device)
            value_ids = batch["value_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            delta_ts = batch["delta_ts"].to(device)
            abs_ts = batch["abs_ts"].to(device)
            context_vec = batch["context_vec"].to(device)

            if train:
                optimizer.zero_grad()

            logits = embedder.forward_with_decoder(
                raw_concept_ids=raw_concept_ids,
                concept_ids=concept_ids,
                value_ids=value_ids,
                position_ids=position_ids,
                delta_ts=delta_ts,
                abs_ts=abs_ts,
                patient_contexts=context_vec
            )  # [B, T, V]

            B, T, V = logits.shape
            multi_hot_targets = get_multi_hot_targets(position_ids, vocab_size=V, k=window_k)
            loss = loss_fn(logits, multi_hot_targets)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
    
    def get_multi_hot_targets(position_ids, vocab_size, k):
        """
        For each timestep t, mark all tokens in [t+1, t+k] in a multi-hot vector.
        """
        B, T = position_ids.shape
        targets = torch.zeros((B, T, vocab_size), dtype=torch.float32, device=position_ids.device)

        for step in range(1, k + 1):
            if T - step <= 0:
                continue
            for b in range(B):
                for t in range(T - step):
                    token = position_ids[b, t + step].item()
                    if token != embedder.padding_idx:
                        targets[b, t, token] = 1.0
        return targets

    # ----- Training loop -----
    for epoch in range(start_epoch, TRAINING_SETTINGS["phase1_n_epochs"] + 1):
        tr_loss = run_epoch(train_loader, train=True)
        vl_loss = run_epoch(val_loader, train=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[Phase 1] Epoch {epoch:03d} | Train: {tr_loss:.4f} | Val: {vl_loss:.4f}")
        scheduler.step(vl_loss)

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": embedder.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
        }, ckpt_last)

        # Save best model
        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            torch.save(embedder.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= TRAINING_SETTINGS["patience"]:
                print("[Phase 1] Early stopping triggered.")
                break

    # Save scaler for downstream use
    joblib.dump(embedder.scaler, os.path.join(ckpt_path.parent, "scaler.pkl"))
    return embedder, train_losses, val_losses