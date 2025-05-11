import pandas as pd
from transform_emr.dataset import EMRDataset
from transform_emr.config.dataset_config import TRAIN_TEMPORAL_DATA_FILE, TRAIN_CTX_DATA_FILE
from transform_emr.config.model_config import MODEL_CONFIG

import pytest

@pytest.mark.order(1)
def test_dataset_initialization():
    df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE)
    ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)
    ds = EMRDataset(df, ctx_df)
    MODEL_CONFIG['vocab_size'] = len(set(ds.token2id.keys())) # Dinamically updating vocab size
    MODEL_CONFIG['ctx_dim'] = ds.context_df.shape[1] # Dinamically updating shape

    
    assert len(ds) > 0
    assert "TokenID" in ds.tokens_df.columns
    assert ds.context_df.shape[1] > 0