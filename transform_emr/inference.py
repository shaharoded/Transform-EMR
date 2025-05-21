import torch
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
import joblib
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *
from transform_emr.config.model_config import *
from transform_emr.transformer import GPT
from transform_emr.embedder import EMREmbedding
from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset


def load_test_data(max_input_days=5):
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)

    # Load scaler and tokenizer
    scaler = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")
    tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")

    # Run processing
    processor = DataProcessor(df, ctx_df, scaler=scaler, max_input_days=max_input_days)
    df, ctx_df = processor.run()

    # Create dataset
    dataset = EMRDataset(df, ctx_df, tokenizer=tokenizer)
    return dataset


def load_embedder(checkpoint_path=EMBEDDER_CHECKPOINT):
    """Load the trained embedder with full config, weights, and scaler from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    tokenizer = EMRTokenizer.load()

    embedder = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=ckpt["model_config"]["ctx_dim"],
        time2vec_dim=ckpt["model_config"]["time2vec_dim"],
        embed_dim=ckpt["model_config"]["embed_dim"],
        dropout=ckpt["model_config"].get("dropout", 0.1),
    )

    embedder.load_state_dict(ckpt["model_state"])
    embedder.eval()

    return embedder



def load_transformer():
    """Load full transformer model including fine-tuned embedder state."""
    embedder = load_embedder()  # Includes all internal state (scaler, token2id, etc.)

    ckpt = torch.load(TRANSFORMER_CHECKPOINT, map_location=torch.device("cpu"))
    model_config = ckpt.get("model_config", MODEL_CONFIG)
    
    model = GPT(model_config, embedder, use_checkpoint=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def get_token_embedding(embedder, token: str) -> torch.Tensor:
    """
    Returns the embedding vector of a specific token from a trained embedder.

    Args:
        embedder (EMREmbedding): A trained EMREmbedding model.
        token (str): The string token to lookup.

    Returns:
        torch.Tensor: Embedding vector of shape [embed_dim].
    """
    if token not in embedder.token2id:
        raise ValueError(f"Token '{token}' not found in vocabulary.")
    
    token_id = embedder.tokenizer.token2id[token]
    embedding = embedder.token_embed.weight[token_id].detach()
    return embedding

def infer_event_stream(model, dataset, max_len=500):
    """
    Generates a stream of events for each patient in the dataset.
    The output includes the input sequence and all generated tokens,
    marking input vs. generated, outcomes, and terminal tokens.

    Args:
        model: Trained GPT model (transform_emr).
        dataset: EMRDataset object (only context and patient_groups needed).
        max_len: Max number of tokens to generate per patient (excluding input).

    Returns:
        pd.DataFrame with columns: PatientID, Step, Token, IsInput, IsOutcome, IsTerminal
    
    NOTE: Before activation you need to assign the scaler from checkpoints/phase1 (training) as as input of EMRDataset
          So that you'll scale the context features the same way they were scaled during training.
    """
    token2id = model.embedder.tokenizer.token2id
    id2token = model.embedder.tokenizer.id2token
    pad_token = token2id["[PAD]"]
    ctx_token = token2id["[CTX]"]

    outcome_ids = {token2id[o] for o in OUTCOMES if o in token2id}
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}

    rows = []
    device = next(model.parameters()).device

    for pid in tqdm(dataset.patient_ids, desc="Generating"):
        context_vec = torch.tensor(dataset.context_df.loc[pid].values, dtype=torch.float32).unsqueeze(0).to(device)  # [1, ctx_dim]

        # Start with [CTX] token
        token_seq = torch.tensor([[ctx_token]], dtype=torch.long, device=device)  # [1, 1]
        time_seq = torch.zeros((1, 1), dtype=torch.float32, device=device)

        rows.append({
            "PatientID": pid,
            "Step": 0,
            "Token": "[CTX]",
            "IsInput": 1,
            "IsOutcome": 0,
            "IsTerminal": 0
        })

        # Add patient's input sequence (excluding [CTX])
        input_tokens = dataset.patient_groups[pid]['PositionID'].tolist()
        for step, tok_id in enumerate(input_tokens):
            token_seq = torch.cat([token_seq, torch.tensor([[tok_id]], device=device)], dim=1)
            time_seq = torch.cat([time_seq, torch.zeros((1, 1), device=device)], dim=1)
            rows.append({
                "PatientID": pid,
                "Step": step + 1,
                "Token": id2token.get(tok_id, f"<UNK_{tok_id}>"),
                "IsInput": 1,
                "IsOutcome": int(tok_id in outcome_ids),
                "IsTerminal": int(tok_id in terminal_ids)
            })
            if tok_id in terminal_ids:
                break

        # Begin generation
        steps = 0
        while steps < max_len:
            with torch.no_grad():
                logits, _ = model(token_seq, time_seq, context_vec)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1).item()

            is_outcome = next_token in outcome_ids
            is_terminal = next_token in terminal_ids

            rows.append({
                "PatientID": pid,
                "Step": len(token_seq[0]),
                "Token": id2token.get(next_token, f"<UNK_{next_token}>") if next_token != pad_token else "[PAD]",
                "IsInput": 0,
                "IsOutcome": int(is_outcome),
                "IsTerminal": int(is_terminal)
            })

            if is_terminal:
                break

            token_seq = torch.cat([token_seq, torch.tensor([[next_token]], device=device)], dim=1)
            time_seq = torch.cat([time_seq, torch.zeros((1, 1), device=device)], dim=1)
            steps += 1

    return pd.DataFrame(rows)


def infer_event_stream(model, dataset, max_len=500):
    """
    Generates a stream of events for each patient in the dataset.
    Args:
        model: Trained GPT model.
        dataset: EMRDataset object (must contain all token components and context).
        max_len: Number of new tokens to generate.
    Returns:
        DataFrame with PatientID, Step, Token, IsInput, IsOutcome, IsTerminal
    """
    tokenizer = model.embedder.tokenizer
    token2id = tokenizer.token2id
    id2token = tokenizer.id2token
    ctx_token = token2id["[CTX]"]
    pad_token = token2id["[PAD]"]
    outcome_ids = {token2id[o] for o in OUTCOMES if o in token2id}
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}
    
    device = next(model.parameters()).device
    rows = []

    def decode_token_components(token):
        parts = token.split("_")
        raw = parts[0]
        concept = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
        value = "_".join(parts[:-1]) if parts[-1] in ("START", "END") else "_".join(parts)
        return (
            tokenizer.rawconcept2id.get(raw, tokenizer.mask_token_id),
            tokenizer.concept2id.get(concept, tokenizer.mask_token_id),
            tokenizer.value2id.get(value, tokenizer.mask_token_id)
        )

    for pid in tqdm(dataset.patient_ids, desc="Generating"):
        df = dataset.patient_groups[pid]
        ctx_vec = torch.tensor(dataset.context_df.loc[pid].values, dtype=torch.float32).unsqueeze(0).to(device)

        # Prepare input tensors
        raw_ids     = torch.tensor([df["RawConceptID"].tolist()], dtype=torch.long, device=device)
        concept_ids = torch.tensor([df["ConceptID"].tolist()], dtype=torch.long, device=device)
        value_ids   = torch.tensor([df["ValueID"].tolist()], dtype=torch.long, device=device)
        pos_ids     = torch.tensor([df["PositionID"].tolist()], dtype=torch.long, device=device)
        delta_ts    = torch.tensor([df["TimeDelta"].tolist()], dtype=torch.float32, device=device)
        abs_ts      = torch.tensor([df["TimePoint"].tolist()], dtype=torch.float32, device=device)

        seq_len = pos_ids.size(1)

        # Log all inputs
        rows.append({
            "PatientID": pid, "Step": 0, "Token": "[CTX]",
            "IsInput": 1, "IsOutcome": 0, "IsTerminal": 0
        })
        for i in range(seq_len):
            tok_id = pos_ids[0, i].item()
            rows.append({
                "PatientID": pid,
                "Step": i + 1,
                "Token": id2token.get(tok_id, f"<UNK_{tok_id}>"),
                "IsInput": 1,
                "IsOutcome": int(tok_id in outcome_ids),
                "IsTerminal": int(tok_id in terminal_ids)
            })
            if tok_id in terminal_ids:
                break

        # Begin autoregressive generation
        steps = 0
        while steps < max_len:
            with torch.no_grad():
                logits = model(
                    raw_concept_ids=raw_ids,
                    concept_ids=concept_ids,
                    value_ids=value_ids,
                    position_ids=pos_ids,
                    delta_ts=delta_ts,
                    abs_ts=abs_ts,
                    context_vec=ctx_vec
                )
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            tok_str = id2token.get(next_token_id, f"<UNK_{next_token_id}>")
            is_outcome = next_token_id in outcome_ids
            is_terminal = next_token_id in terminal_ids

            rows.append({
                "PatientID": pid,
                "Step": raw_ids.shape[1] + 1,
                "Token": tok_str,
                "IsInput": 0,
                "IsOutcome": int(is_outcome),
                "IsTerminal": int(is_terminal)
            })

            if is_terminal:
                break

            # Get generated tokens components for future decode
            raw_id, concept_id, value_id = decode_token_components(tok_str)

            raw_ids     = torch.cat([raw_ids, torch.tensor([[raw_id]], device=device)], dim=1)
            concept_ids = torch.cat([concept_ids, torch.tensor([[concept_id]], device=device)], dim=1)
            value_ids   = torch.cat([value_ids, torch.tensor([[value_id]], device=device)], dim=1)
            pos_ids     = torch.cat([pos_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            delta_ts    = torch.cat([delta_ts, torch.tensor([[1.0]], device=device)], dim=1) # Incrementing time by 1 hour
            abs_ts      = torch.cat([abs_ts, abs_ts[:, -1:] + 1.0], dim=1) # Incrementing time by 1 hour

            steps += 1

    return pd.DataFrame(rows)
