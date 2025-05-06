"""
Possible issues to solve:
 - embed_dim needs to match with Embedder and Transformer
 - vocab_size needs to be automatically extracted from Dataset and nto pre-defined
 - adjust block_size, n_layer, n_head to data size and shape
"""

MODEL_CONFIG = {
      "block_size": 128,
      "vocab_size": 50257,
      "n_layer": 6,
      "n_head": 6,
      "embed_dim": 384,
      "dropout": 0.1,
      "bias": True
    }