
import torch
import pandas as pd
from transform_emr.dataset import EMRTokenizer
from transform_emr.inference import load_test_data, load_transformer, load_embedder, infer_event_stream, get_token_embedding

import pytest

@pytest.mark.order(4)
def test_embedder_token_embedding():
    tokenizer = EMRTokenizer.load()
    embedder = load_embedder()

    sample_token = next(iter(tokenizer.token2id))
    vec = get_token_embedding(embedder, sample_token)

    assert isinstance(vec, torch.Tensor)
    assert vec.shape[0] == embedder.output_dim
    assert torch.isfinite(vec).all()


@pytest.mark.order(5)
def test_inference_on_test_data():
    # Load test data    
    dataset = load_test_data(max_input_days=5)

    # Load model
    model = load_transformer()
    model.eval()

    # Run inference
    result_df = infer_event_stream(model, dataset)

    # Validate output
    assert isinstance(result_df, pd.DataFrame)
    assert set(["PatientID", "Step", "Token", "IsInput", "IsOutcome", "IsTerminal"]).issubset(result_df.columns)
    assert len(result_df) > 0
