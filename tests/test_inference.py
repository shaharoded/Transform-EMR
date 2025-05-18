
import torch
import pandas as pd
import joblib
from pathlib import Path
from transform_emr.dataset import EMRDataset
from transform_emr.inference import load_transformer, load_embedder, infer_event_stream, get_token_embedding
from transform_emr.config.model_config import MODEL_CONFIG, EMBEDDER_CHECKPOINT
from transform_emr.config.dataset_config import TEST_TEMPORAL_DATA_FILE, TEST_CTX_DATA_FILE, OUTCOMES, TERMINAL_OUTCOMES

import pytest

@pytest.mark.order(4)
def test_embedder_token_embedding():
    # Load trained embedder
    embedder = load_embedder(MODEL_CONFIG)

    # Load scaler
    scaler_path = Path(EMBEDDER_CHECKPOINT).resolve().parent / "scaler.pkl"
    scaler = joblib.load(scaler_path)

    # Load test dataset to access vocabulary
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)
    dataset = EMRDataset(df, ctx_df, scaler=scaler)

    # Inject token2id into embedder (if not already present)
    embedder.token2id = dataset.token2id

    # Pick an example token from the vocab
    sample_token = next(iter(embedder.token2id))  # Get any token, e.g., "GLUCOSE_MEASURE_Low_START"

    # Run embedding lookup
    vec = get_token_embedding(embedder, sample_token)

    # Assertions
    assert isinstance(vec, torch.Tensor)
    assert vec.shape[0] == embedder.token_embed.embedding_dim
    assert torch.isfinite(vec).all()

@pytest.mark.order(5)
def test_inference_on_test_data():
    # Load test data
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE, low_memory=False)
    
    # Load scaler from the checkpoint directory
    scaler_path = Path(EMBEDDER_CHECKPOINT).resolve().parent / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    
    # Create dataset using the same scaler
    dataset = EMRDataset(df, ctx_df, scaler=scaler)

    # Load model
    model = load_transformer(MODEL_CONFIG)
    model.eval()

    # Run inference
    result_df = infer_event_stream(model, dataset)

    # Validate output
    assert isinstance(result_df, pd.DataFrame)
    assert set(["PatientID", "Step", "Token", "IsInput", "IsOutcome", "IsTerminal"]).issubset(result_df.columns)
    assert len(result_df) > 0
