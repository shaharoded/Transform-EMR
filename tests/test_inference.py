
import torch
import pandas as pd
import pytest

from transform_emr.dataset import EMRTokenizer
from transform_emr.inference import load_test_data, infer_event_stream, get_token_embedding
from transform_emr.embedder import EMREmbedding
from transform_emr.transformer import GPT
from transform_emr.config.model_config import EMBEDDER_CHECKPOINT, TRANSFORMER_CHECKPOINT


@pytest.mark.order(4)
def test_embedder_token_embedding():
    tokenizer = EMRTokenizer.load()
    embedder, _, _, _, _ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)

    sample_token = next(iter(tokenizer.token2id))
    vec = get_token_embedding(embedder, sample_token)

    assert isinstance(vec, torch.Tensor)
    assert vec.shape[0] == embedder.output_dim
    assert torch.isfinite(vec).all()


@pytest.mark.order(5)
def test_inference_on_test_data():
    dataset = load_test_data(max_input_days=5)

    tokenizer = EMRTokenizer.load()
    embedder, _, _, _, _ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)
    model, _, _, _, _ = GPT.load(TRANSFORMER_CHECKPOINT, embedder=embedder)

    model.eval()
    result_df = infer_event_stream(model, dataset)

    assert isinstance(result_df, pd.DataFrame)
    assert set(["PatientID", "Step", "Token", "IsInput", "IsOutcome", "IsTerminal"]).issubset(result_df.columns)
    assert len(result_df) > 0
