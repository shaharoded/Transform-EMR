import torch
from transform_emr.embedding import EMREmbedding
from transform_emr.config.model_config import MODEL_CONFIG

import pytest

@pytest.mark.order(2)
def test_embedder_initialization():
    model = EMREmbedding(
        vocab_size=MODEL_CONFIG["vocab_size"],
        ctx_dim=MODEL_CONFIG["ctx_dim"],
        time2vec_dim=MODEL_CONFIG.get("time2vec_dim", 8),
        embed_dim=MODEL_CONFIG["embed_dim"]
    )
    assert isinstance(model, torch.nn.Module)
    assert model.output_dim == MODEL_CONFIG["embed_dim"]