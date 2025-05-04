import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Local Code
from dataset import EMRDataset
from embedding import EMREmbedding
from config.model_config import MODEL_CONFIG


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ Causal Self-Attention mechanism used in the GPT model """

    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"], bias=config["bias"])
        # Output projection
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"], bias=config["bias"])
        # Dropouts
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.dropout = config["dropout"]

        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config["block_size"], config["block_size"])).view(1, 1, config["block_size"], config["block_size"]))

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality (n_embd)
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"], bias=config["bias"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"], bias=config["bias"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ Transformer block, including a self-attention layer and an MLP """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Model """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config["vocab_size"], config["n_embd"]), # Token embedding layer, text -> embedding vector
            "wpe": nn.Embedding(config["block_size"], config["n_embd"]), # Positional embedding layer, position -> embedding vector
            "drop": nn.Dropout(config["dropout"]),
            "h": nn.ModuleList([Block(config) for _ in range(config["n_layer"])]), # Transformer blocks
            "ln_f": LayerNorm(config["n_embd"], bias=config["bias"]), # Final layer normalization
        })
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layer"]))

        # Report the number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Collect parameters that require gradients
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer["wpe"].weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.
        If `targets` is provided, also calculate the loss.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config["block_size"], f"Cannot forward sequence of length {t}, block size is only {self.config['block_size']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # Shape (t)

        # Forward the GPT model
        tok_emb = self.transformer["wte"](idx)  # Token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer["wpe"](pos)  # Position embeddings of shape (t, n_embd)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        if targets is not None:
            # Calculate the loss if targets are provided
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Only forward the lm_head on the last position during inference
            logits = self.lm_head(x[:, [-1], :])  # Using list [-1] to preserve the time dimension
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate a sequence of tokens given a starting sequence `idx`.
        
        Args:
            idx (torch.Tensor): The starting sequence of token IDs. This can be a prefix, such as a question in
                                a QA setup, which the model will continue from. The shape should be (1, sequence_length).
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The temperature value for controlling the randomness of the generation. 
                                Lower values (e.g., 0.7) make the output more deterministic, while higher values 
                                (e.g., 1.2) make it more random.
            top_k (int, optional): The number of highest probability tokens to keep for sampling. This helps 
                                focus the generation on the most likely tokens and can reduce output randomness.
        
        Returns:
            torch.Tensor: The complete sequence of token IDs, including both the prefix and the generated tokens.
        """
        eot_token_id = TOKENIZER.eot_token  # The token ID for <|endoftext|>
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config["block_size"] else idx[:, -self.config["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Scale by temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Check if the generated token is the end-of-text token
            if idx_next.item() == eot_token_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
