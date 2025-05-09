"""
Possible issues to solve:
    - Ensure synthetic datasets match the real current states and trends, so they can be used for tests
    - Ensure scaler is checkpointed and passed normally
    - Ensure model can infer properly withi this current flow
    - Ensure synthetic datasets match the real current states and trends, so they can be used for tests
    - Update README.md with full activation flow and instructions

Future work: block size limit the context the model is aware of. Papers like BEHRT tried to handle these aspects.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # Go up to project root

EMBEDDER_CHECKPOINT = str(ROOT_DIR / "checkpoints/phase1/best_embedder.pt")
TRANSFORMER_CHECKPOINT = str(ROOT_DIR / "checkpoints/phase2/best.pt")

MODEL_CONFIG = {
      "vocab_size": 0, # A place holder to fill after creating the dataset. Adjust value post-training before deploying.
      "ctx_dim": 0, # A place holder to fill after creating the dataset. Adjust value post-training before deploying.
      "time2vec_dim": 16,
      "embed_dim": 256,
      "block_size": 256,  # //e.g. sequence length, number of tokens processed concurrently
      "n_head": 8,
      "n_layer": 4,
      "dropout": 0.1,
      "bias": True,
      "compile": True # Allows JIT compile for the model - Better memory and speed.
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 500,
    "phase2_n_epochs": 50,
    "patience": 5,
    "phase1_learning_rate": 1e-2,
    "phase2_learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "batch_size": 4, # Number of patients processed concurrently
}