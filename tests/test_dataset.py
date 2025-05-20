import pandas as pd
from transform_emr.dataset import EMRDataset
from transform_emr.config.dataset_config import TRAIN_TEMPORAL_DATA_FILE, TRAIN_CTX_DATA_FILE
from transform_emr.config.model_config import MODEL_CONFIG

import pytest

@pytest.mark.order(1)
def test_dataset_initialization():
    df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE)
    ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)

    # Convert datetime columns
    df['StartDateTime'] = pd.to_datetime(df['StartTime'], utc=True, errors='raise').dt.tz_convert(None)
    df['EndDateTime'] = pd.to_datetime(df['EndTime'], utc=True, errors='raise').dt.tz_convert(None)
    df.drop(columns=["StartTime", "EndTime"], inplace=True)

    # Initialize dataset
    ds = EMRDataset(df, ctx_df)

    # Update model config to match dataset vocab
    MODEL_CONFIG['raw_concept_vocab_size'] = len(ds.rawconcept2id)
    MODEL_CONFIG['concept_vocab_size'] = len(ds.concept2id)
    MODEL_CONFIG['value_vocab_size'] = len(ds.value2id)
    MODEL_CONFIG['vocab_size'] = len(ds.token2id)
    MODEL_CONFIG['ctx_dim'] = ds.context_df.shape[1]

    assert len(ds) > 0
    assert "ConceptID" in ds.tokens_df.columns
    assert "ValueID" in ds.tokens_df.columns
    assert "PositionID" in ds.tokens_df.columns
    assert ds.context_df.shape[1] > 0