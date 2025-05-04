import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Local Code
from config.dataset_config import *
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
        self.linear = nn.Linear(1, 1)  # Linear trend
        self.freq = nn.Linear(1, out_dim - 1)  # Periodic part

    def forward(self, t):
        """
        t: [B, T] or [B*T] float tensor of time deltas
        Returns: [B, T, out_dim] or [B*T, out_dim]
        """
        t = t.unsqueeze(-1)  # [B, T, 1]
        linear_out = self.linear(t)
        periodic_out = torch.sin(self.freq(t))
        return torch.cat([linear_out, periodic_out], dim=-1)


class EMREmbedding(nn.Module):
    """
    Embedding layer for electronic medical record (EMR) sequences.

    This layer combines:
      - token embeddings (e.g., "GLUCOSE_MEASURE_Low_START")
      - time delta embeddings (via Time2Vec)
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

    def __init__(self, token_vocab_size, ctx_dim, time2vec_dim=8,
                 embed_dim=128):
        super().__init__()
        self.token_embed = nn.Embedding(token_vocab_size, embed_dim)
        self.time2vec = Time2Vec(time2vec_dim)
        self.ctx_token = nn.Parameter(
                torch.randn(embed_dim))  # learnable [CTX] token
        self.context_proj = nn.Linear(ctx_dim, embed_dim)

        self.output_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, token_vocab_size)

    def forward(self, token_ids, time_deltas, patient_contexts):
        """
        process a full patient’s event stream in one shot.

        Args:
          token_ids:      [B, T] LongTensor
          time_deltas:    [B, T] FloatTensor (time since previous event)
          patient_contexts: [B, ctx_dim] FloatTensor

        Returns:
            embeddings:   [B, T+1, D] FloatTensor with [CTX] prepended
        """
        tok_embeds = self.token_embed(token_ids)  # [B, T, D]
        time_embeds = self.time2vec(time_deltas)  # [B, T, T2V]
        time_embeds = F.pad(time_embeds, (
            0, tok_embeds.shape[-1] - time_embeds.shape[-1]))  # Match D

        event_embeds = tok_embeds + time_embeds  # [B, T, D] -> concating both parts of event: time + token

        ctx_embed = self.ctx_token + self.context_proj(
                patient_contexts)  # [B, D]
        ctx_embed = ctx_embed.unsqueeze(1)  # [B, 1, D]

        return torch.cat([ctx_embed, event_embeds], dim=1)  # [B, T+1, D]


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

    def train_one_epoch(train_loader, model, decoder, loss_fn, vocab_size, device, lr):
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): Training data loader.
            model (nn.Module): Embedding model.
            decoder (nn.Module): Decoder model.
            loss_fn (Loss): Loss function.
            vocab_size (int): Vocabulary size.
            device (str): Device to run on.
            lr (float): Learning rate.

        Returns:
            float: Average training loss for the epoch.
        """
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(decoder.parameters()), lr=lr)
        size = len(train_loader.dataset)
        model.train()
        decoder.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            token_ids = batch['token_ids'].to(device)
            time_deltas = batch['time_deltas'].to(device)
            context_vector = batch['context_vector'].to(device)

            # Forward pass
            embeddings = model(token_ids, time_deltas, context_vector)
            logits = decoder(embeddings[:, :-1, :])
            target = token_ids

            loss = loss_fn(logits.reshape(-1, vocab_size), target.reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_one_epoch(val_loader, model, decoder, loss_fn, vocab_size, device):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): Validation data loader.
            model (nn.Module): Embedding model.
            decoder (nn.Module): Decoder model.
            loss_fn (Loss): Loss function.
            vocab_size (int): Vocabulary size.
            device (str): Device to run on.

        Returns:
            float: Average validation loss.
        """
        model.eval()
        decoder.eval()
        val_total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch['token_ids'].to(device)
                time_deltas = batch['time_deltas'].to(device)
                context_vector = batch['context_vector'].to(device)

                embeddings = model(token_ids, time_deltas, context_vector)
                logits = decoder(embeddings[:, :-1, :])
                target = token_ids

                loss = loss_fn(logits.reshape(-1, vocab_size), target.reshape(-1))
                val_total_loss += loss.item()

        return val_total_loss / len(val_loader)

    # Device selection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model and decoder
    embedding_model = EMREmbedding(token_vocab_size=vocab_size, ctx_dim=ctx_dim, time2vec_dim=time2vec_dim,
                                   embed_dim=embed_dim).to(device)

    decoder = nn.Linear(embed_dim, vocab_size).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(train_loader=train_loader, model=embedding_model, decoder=decoder, loss_fn=loss_fn,
                                     vocab_size=vocab_size, device=device, lr=lr)

        val_loss = validate_one_epoch(val_loader=val_loader, model=embedding_model, decoder=decoder, loss_fn=loss_fn, 
                                      vocab_size=vocab_size, device=device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Embedder Training Status]: Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[Embedder Training Status]: No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"[Embedder Training Status]: Early stopping triggered after {epoch + 1} epochs.")
            break

    return embedding_model, decoder, train_losses, val_losses


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

    train_dataset = EMRDataset(train_df, train_context, numeric_concepts=NUMERIC_CONCEPTS, context_columns=['age', 'gender'])
    val_dataset = EMRDataset(val_df, val_context, numeric_concepts=NUMERIC_CONCEPTS, context_columns=['age', 'gender'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_emr)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_emr)

    embedding_model = EMREmbedding(
        token_vocab_size=len(train_dataset.token2id), ctx_dim=2, time2vec_dim=8, embed_dim=128)
    
    embedding_model, decoder, train_losses, val_losses = train(train_loader, val_loader, len(train_dataset.token2id), 
                                                               ctx_dim=2, time2vec_dim=8, embed_dim=128, lr=1e-4,
                                                               n_epochs=150, patience=5, pad_token_id=0, device=None)