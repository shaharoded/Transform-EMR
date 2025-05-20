import torch
from transform_emr.embedding import EMREmbedding
from transform_emr.config.model_config import MODEL_CONFIG

import pytest

@pytest.mark.order(2)
def test_embedder_initialization():
    model = EMREmbedding(
        raw_concept_vocab_size=MODEL_CONFIG.get("raw_concept_vocab_size"),
        concept_vocab_size=MODEL_CONFIG.get("concept_vocab_size"),
        value_vocab_size=MODEL_CONFIG.get("value_vocab_size"),
        position_vocab_size=MODEL_CONFIG.get("vocab_size"),
        ctx_dim=MODEL_CONFIG.get("ctx_dim"),
        time2vec_dim=MODEL_CONFIG.get("time2vec_dim"),
        embed_dim=MODEL_CONFIG.get("embed_dim")
    )
    assert isinstance(model, torch.nn.Module)
    assert model.output_dim == MODEL_CONFIG["embed_dim"]