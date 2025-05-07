import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Local Code
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


def train(train_loader, val_loader, vocab_size, ctx_dim=2, time2vec_dim=8, embed_dim=128, lr=1e-4,
        n_epochs=150, patience=5, pad_token_id=0, device=None):
    """
    Trains an EMREmbedding model with a decoder on sequence data using a
    specified training and validation loader.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        vocab_size (int): Vocabulary size for output projection.
        ctx_dim (int, optional): Context vector dimension. Defaults to 2.
        time2vec_dim (int, optional): Time2Vec embedding dimension. Defaults to 8.
        embed_dim (int, optional): Final embedding size. Defaults to 128.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        n_epochs (int, optional): Maximum number of epochs. Defaults to 150.
        patience (int, optional): Early stopping patience. Defaults to 5.
        pad_token_id (int, optional): Token ID used for padding, to be ignored in loss. Defaults to 0.
        device (str, optional): Device for training ('cuda' or 'cpu'). Auto-detects if None.

    Returns:
        Tuple: (trained model, decoder, training losses, validation losses)
    """
    # ---------- device selection ---------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---------- model & tied decoder -----------------------------
    embedder = EMREmbedding( token_vocab_size=vocab_size, ctx_dim=ctx_dim,
        time2vec_dim=time2vec_dim, embed_dim=embed_dim, padding_idx=pad_token_id).to(device)

    decoder = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
    decoder.weight = embedder.token_embed.weight  # weight tying

    loss_fn   = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val, bad_epochs = float("inf"), 0

    # ---------- helper -------------------------------------------
    def run_epoch(loader, model, train_flag=False):
        if train_flag:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        for batch in loader:
            token_ids     = batch["token_ids"].to(device)      # [B,T]
            time_deltas   = batch["time_deltas"].to(device)    # [B,T]
            context_vec   = batch["context_vector"].to(device) # [B,C]

            # forward
            if train_flag:
                optimizer.zero_grad()

            emb = model(token_ids, time_deltas, context_vec, return_mask=False)
            # emb shape = [B, T+1, D]  ( [CTX] + events )

            logits = decoder(emb[:, :-1, :])          # predict next token
            target = token_ids                        # next token = original ids
            loss   = loss_fn(logits.reshape(-1, vocab_size),
                             target.reshape(-1))

            if train_flag:
                loss.backward()
                clip_grad_norm_(list(model.parameters()) + list(decoder.parameters()), 1.0)
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ---------- main loop ----------------------------------------
    for epoch in range(1, n_epochs + 1):
        tr_loss = run_epoch(train_loader, embedder, train_flag=True)
        vl_loss = run_epoch(val_loader,   embedder, train_flag=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[Training Embedder]: Epoch {epoch}: train={tr_loss:.4f} | val={vl_loss:.4f}")

        # early‑stopping
        if vl_loss + 1e-4 < best_val:           # tiny margin to avoid float jitter
            best_val, bad_epochs = vl_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    return embedder, decoder, train_losses, val_losses

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

    # Initiate on random data
    df = pd.read_csv(DATA_FILE)

    # Generate random patient context data
    patient_ids = df['PatientID'].unique()
    np.random.seed(42)

    patient_context_df = pd.DataFrame({
        'PatientID': patient_ids,
        'age': np.random.randint(18, 66, size=len(patient_ids)),
        'gender': np.random.choice([0, 1], size=len(patient_ids))  # 0 = male, 1 = female
    })    
    
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_df = df[df['PatientID'].isin(train_ids)].copy()
    val_df = df[df['PatientID'].isin(val_ids)].copy()

    train_context = patient_context_df[patient_context_df['PatientID'].isin(train_ids)].copy()
    val_context = patient_context_df[patient_context_df['PatientID'].isin(val_ids)].copy()

    train_dataset = EMRDataset(train_df, train_context, states=STATES, context_columns=['age', 'gender'])
    val_dataset = EMRDataset(val_df, val_context, states=STATES, context_columns=['age', 'gender'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_emr)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_emr)

    # Model Params
    time2vec_dim = MODEL_CONFIG.get('time2vec_dim')
    embed_dim = MODEL_CONFIG.get('embed_dim')
    ctx_dim=2

    # Training Params
    N_EPOCHS = 50
    PATIENCE = 5    # Epochs
    PAD_TOKEN_ID = 0
    LR = 1e-2
    
    # Initiate and Train
    embedding_model = EMREmbedding(
        token_vocab_size=len(train_dataset.token2id), ctx_dim=ctx_dim, time2vec_dim=time2vec_dim, embed_dim=embed_dim)
    
    embedding_model, decoder, train_losses, val_losses = train(train_loader, val_loader, len(train_dataset.token2id), 
                                                               ctx_dim=ctx_dim, time2vec_dim=time2vec_dim, embed_dim=embed_dim, lr=LR,
                                                               n_epochs=N_EPOCHS, patience=PATIENCE, pad_token_id=PAD_TOKEN_ID)
    plot_losses(train_losses, val_losses)