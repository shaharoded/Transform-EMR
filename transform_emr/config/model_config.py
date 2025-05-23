import os

# Get project root (2 levels up from config/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
EMBEDDER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'best_embedder.pt')
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'best_model.pt')

MODEL_CONFIG = {
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
    "phase1_n_epochs": 100,
    "phase2_n_epochs": 80,
    "warmup_epochs": 10,
    "patience": 5,
    "phase1_learning_rate": 5e-4,
    "phase2_learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "batch_size": 8, # Number of patients processed concurrently
    "k_window": 5, # For soft targets per token on BCE loss, number of next tokens to predict jointly.
    "penalty_weight": 1.0, # Weight for special penalties given on next token loss function. Currently as calculated.
    "delta_t_weight": 1.0, # Weight loss on the delta_t prediction, which is combined with regular loss. Currently as calculated.
}