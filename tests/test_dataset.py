from transform_emr.inference import load_test_data
import pytest


@pytest.mark.order(1)
def test_dataset_initialization():
    dataset = load_test_data(max_input_days=5)

    assert len(dataset) > 0
    assert "ConceptID" in dataset.tokens_df.columns
    assert "ValueID" in dataset.tokens_df.columns
    assert "PositionID" in dataset.tokens_df.columns
    assert dataset.context_df.shape[1] > 0