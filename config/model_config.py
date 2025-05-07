"""
Possible issues to solve:
 - embed_dim needs to match with Embedder and Transformer
 - vocab_size needs to be automatically extracted from Dataset
 - adjust block_size, n_layer, n_head to data size and shape

Future work: block size limit the context the model is aware of. Papers like BEHRT tried to handle these aspects.
"""

MODEL_CONFIG = {
      "vocab_size": 0, # A place holder to fill after creating the dataset
      "time2vec_dim": 16,
      "embed_dim": 256,
      "block_size": 256,  # //e.g. sequence length, number of tokens processed concurrently
      "n_head": 8,
      "n_layer": 4,
      "dropout": 0.1,
      "bias": True,
      "compile": True # Allows JIT compile for the model - Better memory and speed.
    }