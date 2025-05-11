import os

# Go two levels up from this config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data file paths (relative to project root)
TRAIN_TEMPORAL_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'train', 'synthetic_diabetes_temporal_data.csv')
TRAIN_CTX_DATA_FILE      = os.path.join(PROJECT_ROOT, 'data', 'train', 'synthetic_diabetes_context_data.csv')
TEST_TEMPORAL_DATA_FILE  = os.path.join(PROJECT_ROOT, 'data', 'test', 'synthetic_diabetes_temporal_data.csv')
TEST_CTX_DATA_FILE       = os.path.join(PROJECT_ROOT, 'data', 'test', 'synthetic_diabetes_context_data.csv')

# Define the prediction targets and <eot> tokens to terminate the inference
OUTCOMES = [
    "RELEASE_EVENT",
    "DEATH_EVENT",
    "KETOACIDOSIS_EVENT",
    "KIDNEY_DISORDER_EVENT",
    "COMA_EVENT",
    "EYE_DISORDER_EVENT",
    "NERVOUS_SYSTEM_DISORDER_EVENT",
    "VASCULAR_DISORDER_EVENT",
    "OTHER_COMPLICATION_EVENT",
    "DEMENTIA_EVENT",
    "CARDIOVASCULAR_DISORDER_EVENT",
    "ULCER_EVENT",
    "INFECTION_EVENT",
    "MUSCULOSKELETAL_COMPLICATION_EVENT",
    "NEUROVASCULAR_COMPLICATION_EVENT"
]

TERMINAL_OUTCOMES = ["RELEASE_EVENT", "DEATH_EVENT"]