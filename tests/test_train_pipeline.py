from transform_emr.train import run_two_phase_training

import pytest

@pytest.mark.order(3)
def test_full_training_loop_runs():
    # This assumes data is present in the location specified in dataset_config
    run_two_phase_training()  # Will resume from checkpoint or start fresh