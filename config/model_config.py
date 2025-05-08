"""
Possible issues to solve:
 - adjust block_size, n_layer, n_head to data size and shape

Future work: block size limit the context the model is aware of. Papers like BEHRT tried to handle these aspects.
"""

MODEL_CONFIG = {
      "vocab_size": 0, # A place holder to fill after creating the dataset
      "ctx_dim": 0, # A place holder to fill after creating the dataset
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
    "weight_decay": 1e-2,
    "batch_size": 4, # Number of patients processed concurrently
}