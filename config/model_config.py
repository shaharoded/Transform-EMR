"""
Possible issues to solve:
 - embed_dim needs to match with Embedder and Transformer
 - vocab_size needs to be automatically extracted from Dataset and nto pre-defined
 - adjust block_size, n_layer, n_head to data size and shape
"""

MODEL_CONFIG = {
      "time2vec_dim": 16,
      "embed_dim": 256,
      "block_size": 256,  # //e.g. sequence length, number of tokens processed concurrently
      "n_head": 8,
      "n_layer": 4,
      "dropout": 0.1
    }