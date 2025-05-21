import torch
from transform_emr.embedder import EMREmbedding
from transform_emr.dataset import EMRTokenizer
from transform_emr.config.model_config import MODEL_CONFIG

import pytest

@pytest.mark.order(2)
def test_embedder_initialization():
    tokenizer = EMRTokenizer.load()
    model = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=MODEL_CONFIG.get("ctx_dim"),
        time2vec_dim=MODEL_CONFIG.get("time2vec_dim"),
        embed_dim=MODEL_CONFIG.get("embed_dim")
    )
    assert isinstance(model, torch.nn.Module)
    assert model.output_dim == MODEL_CONFIG["embed_dim"]