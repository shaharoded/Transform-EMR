import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ───────── local code ─────────────────────────────────────────────────── #
from config.dataset_config import *
from config.model_config import *
from dataset import EMRDataset, collate_emr


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
    Replaces token‑emb + positional‑emb in a Transformer.

    This layer combines:
      - token embeddings (e.g., "GLUCOSE_MEASURE_Low_START")
      - time delta embeddings (via Time2Vec)
      - √D scaling, dropout & LayerNorm for stability
      - a learnable [CTX] token for patient-level context (age, gender, etc.)

    Each event is represented by the sum of its token and time embeddings.
    The final sequence is prepended with a context-aware [CTX] embedding.

    Args:
        token_vocab_size (int): Number of unique concept-value tokens
        ctx_dim (int): Dimension of patient context vector
        time2vec_dim (int): Output dimension of the Time2Vec embedding
        embed_dim (int): Final embedding size for each token/event

    Input:
        token_ids (LongTensor): [B, T] token IDs
        time_deltas (FloatTensor): [B, T] time since the previous event (in days)
        patient_contexts (FloatTensor): [B, ctx_dim] patient-level context features

    Output:
        embeddings (FloatTensor): [B, T+1, embed_dim] — with a [CTX] token prepended
    """

    def __init__(self, token_vocab_size, ctx_dim, time2vec_dim=8, embed_dim=128, padding_idx=0, dropout=0.1):
        super().__init__()

        # --- token & time -------------------------------------------------
        self.padding_idx = padding_idx
        self.token_embed = nn.Embedding(token_vocab_size, embed_dim, padding_idx=padding_idx)
        self.time2vec = Time2Vec(time2vec_dim)
        self.time_proj  = nn.Linear(time2vec_dim, embed_dim, bias=False)

        # --- patient‑context slot ----------------------------------------
        self.ctx_token  = nn.Parameter(torch.randn(embed_dim)) # learnable [CTX] token
        self.context_proj = nn.Linear(ctx_dim, embed_dim, bias=False)

        self.output_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, token_vocab_size)

        # --- regularisation ----------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)

        self.output_dim = embed_dim  # keep public attr for compatibility

        # --- Decoder tied to token embeddings ----------------------------
        self.decoder = nn.Linear(embed_dim, token_vocab_size, bias=False)
        self.decoder.weight = self.token_embed.weight  # weight tying

    def forward(self, token_ids, time_deltas, patient_contexts, return_mask):
        """
        process a full patient’s event stream in one shot.

        Args:
          token_ids (torch.Tensor):      [B, T] LongTensor
          time_deltas (torch.Tensor):    [B, T]  (Δt)  FloatTensor (time since previous event)
          patient_contexts (torch.Tensor): [B, ctx_dim] FloatTensor
          return_mask (bool): 

        Returns:
            embeddings:   [B, T+1, D] FloatTensor with [CTX] prepended
            attention_mask (optional) : BoolTensor [B, T+1]
            True for real tokens, False for PAD
        """
        # ---- look‑ups ---------------------------------------------------
        tok_vec  = self.token_embed(token_ids)            # [B, T, D]
        time_vec = self.time_proj(self.time2vec(time_deltas))  # [B, T, D]

        ev_vec = (tok_vec + time_vec) / self.scale        # [B, T, D]
        ev_vec = self.dropout(ev_vec)

        # ---- [CTX] slot -------------------------------------------------
        ctx_vec = self.ctx_token + self.context_proj(patient_contexts)  # [B, D]
        ctx_vec = ctx_vec.unsqueeze(1)                    # [B, 1, D]

        seq = torch.cat([ctx_vec, ev_vec], dim=1)         # [B, T+1, D]
        seq = self.layernorm(seq)

        if return_mask:
            pad_mask = (token_ids != self.token_embed.padding_idx)
            pad_mask = torch.cat(
                [torch.ones_like(pad_mask[:, :1]), pad_mask], dim=1
            )  # prepend 1 for [CTX]
            return seq, pad_mask

        return seq
    
    def forward_with_decoder(self, token_ids, time_deltas, patient_contexts):
        """
        Runs the full forward pass with decoding (for solo training)

        Args:
            token_ids: [B, T]
            time_deltas: [B, T]
            patient_contexts: [B, ctx_dim]

        Returns:
            logits: [B, T, vocab_size] — predicted next-token scores
        """
        seq = self.forward(token_ids, time_deltas, patient_contexts, return_mask=False)  # [B, T+1, D]
        return self.decoder(seq[:, :-1, :])  # remove [CTX] for prediction


def train_embedder(embedder, train_loader, val_loader, vocab_size, lr=1e-4,
          n_epochs=150, patience=5, pad_token_id=0, device=None):
    """
    Trains an EMREmbedding model (with internal decoder) on EMR data.

    Args:
        embedder (EMREmbedding): A fully initialized embedder with decoder.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        vocab_size (int): Size of token vocabulary.
        lr (float): Learning rate.
        n_epochs (int): Max number of training epochs.
        patience (int): Early stopping patience.
        pad_token_id (int): Token to ignore in loss.
        device (str): 'cuda' or 'cpu'. Auto-detect if None.

    Returns:
        Tuple: (trained embedder, train_losses, val_losses)
    """
    # ----- Device setup -----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = embedder.to(device)

    # ----- Loss and optimizer -----
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val, bad_epochs = float("inf"), 0

    def run_epoch(loader, train_flag=False):
        embedder.train() if train_flag else embedder.eval()
        total_loss = 0.0

        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            time_deltas = batch["time_deltas"].to(device)
            context_vec = batch["context_vector"].to(device)

            if train_flag:
                optimizer.zero_grad()

            logits = embedder.forward_with_decoder(token_ids, time_deltas, context_vec)  # [B, T, V]
            loss = loss_fn(logits.reshape(-1, vocab_size), token_ids.reshape(-1))

            if train_flag:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ----- Training loop -----
    for epoch in range(1, n_epochs + 1):
        tr_loss = run_epoch(train_loader, train_flag=True)
        vl_loss = run_epoch(val_loader, train_flag=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[Training Embedder]: Epoch {epoch:03d} | Train: {tr_loss:.4f} | Val: {vl_loss:.4f}")

        if vl_loss + 1e-4 < best_val:
            best_val, bad_epochs = vl_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("[Training Embedder]: Early stopping in phase 1!")
                break

    return embedder, train_losses, val_losses

def plot_losses(train_losses, val_losses):
    """
    Plot train vs. validation loss to inspect training quality.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‑entropy loss")
    plt.title("Training vs. validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split 

    # Initiate on dataset
    temporal_df = pd.read_csv(TEMPORAL_DATA_FILE)
    ctx_df = pd.read_csv(CTX_DATA_FILE)

    # Generate random patient context data
    patient_ids = temporal_df['PatientID'].unique()  
    
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_df = temporal_df[temporal_df['PatientID'].isin(train_ids)].copy()
    val_df = temporal_df[temporal_df['PatientID'].isin(val_ids)].copy()

    train_context = ctx_df[ctx_df['PatientID'].isin(train_ids)].copy()
    val_context = ctx_df[ctx_df['PatientID'].isin(val_ids)].copy()

    train_dataset = EMRDataset(train_df, train_context, states=STATES)
    val_dataset = EMRDataset(val_df, val_context, states=STATES, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_emr)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_emr)

    # Hyperparameters
    ctx_dim       = len(train_dataset.context_df.columns)
    time2vec_dim  = MODEL_CONFIG.get("time2vec_dim", 16)
    embed_dim     = MODEL_CONFIG.get("embed_dim", 256)
    vocab_size    = len(train_dataset.token2id)

    N_EPOCHS      = 50
    PATIENCE      = 5
    PAD_TOKEN_ID  = 0
    LR            = 1e-2

    # Initialize embedder with decoder
    embedding_model = EMREmbedding(
        token_vocab_size=vocab_size,
        ctx_dim=ctx_dim,
        time2vec_dim=time2vec_dim,
        embed_dim=embed_dim,
        padding_idx=PAD_TOKEN_ID
    )

    # Train
    embedding_model, train_losses, val_losses = train_embedder(
        embedding_model,
        train_loader,
        val_loader,
        vocab_size=vocab_size,
        lr=LR,
        n_epochs=N_EPOCHS,
        patience=PATIENCE,
        pad_token_id=PAD_TOKEN_ID
    )

    # Visualize loss curves
    plot_losses(train_losses, val_losses)