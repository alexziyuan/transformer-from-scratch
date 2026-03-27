import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ops import (
    matmul,
    layer_norm,
    multi_head_attention,
    feed_forward,
    causal_mask,
)

class TransformerBlock:
    """
    One GPT-2 transformer block:

        x -> LayerNorm -> MultiHeadAttention -> residual
          -> LayerNorm -> FFN                -> residual
    
    Pre-norm architecture (LayerNorm before the sublayer, not after),
    which differs from the original "Attention Is All You Need".
    GPT-2 uses pre-norm throughout.
    """

    def __init__(self, weights: dict, layer_idx: int, n_head: int) -> None:
        """
        Extract this layer's weights from the full weight dict.

        Args:
            weights:   full weight dict from loader.load_weights()
            layer_idx: which block (0-indexed)
            n_head:    number of attention heads
        """
        self.n_head = n_head
        prefix = f"h.{layer_idx}"

        # Layer norm 1 (before attention)
        self.ln1_w = weights[f"{prefix}.ln_1.weight"]
        self.ln1_b = weights[f"{prefix}.ln_1.bias"]

        # Fused QKV projection + output projection
        self.c_attn_w = weights[f"{prefix}.attn.c_attn.weight"]
        self.c_attn_b = weights[f"{prefix}.attn.c_attn.bias"]
        self.c_proj_w = weights[f"{prefix}.attn.c_proj.weight"]
        self.c_proj_b = weights[f"{prefix}.attn.c_proj.bias"]

        # Layer norm 2 (before FFN)
        self.ln2_w = weights[f"{prefix}.ln_2.weight"]
        self.ln2_b = weights[f"{prefix}.ln_2.bias"]

        # Feed-forward network
        self.fc_w   = weights[f"{prefix}.mlp.c_fc.weight"]
        self.fc_b   = weights[f"{prefix}.mlp.c_fc.bias"]
        self.proj_w = weights[f"{prefix}.mlp.c_proj.weight"]
        self.proj_b = weights[f"{prefix}.mlp.c_proj.bias"]

    def forward(self,
                x: np.ndarray,
                mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x:    [seq_len, d_model]
            mask: [seq_len, seq_len] causal mask
        Returns:
            x:           [seq_len, d_model]
            attn_weights:[n_head, seq_len, seq_len]
        """
        # Pre-norm attention with residual connection
        x_norm = layer_norm(x, self.ln1_w, self.ln1_b)
        attn_out, attn_weights = multi_head_attention(
            x_norm,
            self.c_attn_w,
            self.c_attn_b,
            self.c_proj_w,
            self.c_proj_b,
            self.n_head,
            mask,
        )
        x = x + attn_out   # residual

        # Pre-norm FFN with residual connection
        x_norm = layer_norm(x, self.ln2_w, self.ln2_b)
        ff_out = feed_forward(
            x_norm,
            self.fc_w,
            self.fc_b,
            self.proj_w,
            self.proj_b,
        )
        x = x + ff_out     # residual

        return x, attn_weights


class GPT2:
    """
    Full GPT-2 model: embedding -> N transformer blocks -> LM head.

    The LM head (token prediction) shares weights with the token embedding
    matrix (weight tying) - this is a deliberate GPT-2 design choice that
    saves parameters and often improves performance.    
    """

    def __init__(self, weights: dict, config: dict) -> None:
        """
        Args:
            weights: from loader.load_weights()
            config:  from loader.get_config()
        """
        self.config = config
        n_head  = config["n_head"]
        n_layer = config["n_layer"]

        # Token and positional embeddings
        self.wte = weights["wte.weight"]    # [n_vocab, d_model]
        self.wpe = weights["wpe.weight"]    # [n_ctx,   d_model]

        # Transformer blocks
        self.blocks = [
            TransformerBlock(weights, i, n_head)
            for i in range(n_layer)
        ]
        # Final layer norm
        self.ln_f_w = weights["ln_f.weight"]
        self.ln_f_b = weights["ln_f.bias"]

        # LM head - weight-tied to wte (no separate weight)
        # Logits = hidden_state @ wte.T

    def forward(
        self,
        token_ids: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Args:
            token_ids: [seq_len] integer token indices

        Returns:
            logits:      [seq_len, n_vocab]
                         logits[i] predicts token at position i+1
            all_weights: list of length n_layer, each [n_head, seq_len, seq_len]
                         attention weights per layer — the visualization hook
        """
        T = len(token_ids)

        # Embeddings: token + positional
        positions = np.arange(T)
        x = self.wte[token_ids] + self.wpe[positions]    # [T, d_model]
        
        # Causal mask - same mask used by all blocks
        mask = causal_mask(T)

        # Run through all transformer blocks, collecting attention weights
        all_weights = []
        for block in self.blocks:
            x, attn_weights = block.forward(x, mask)
            all_weights.append(attn_weights)    # [n_head, T, T]

        # Final layer norm
        x = layer_norm(x, self.ln_f_w, self.ln_f_b)    # [T, d_model]

        # LM head (weight-tied with token embedding)
        logits = matmul(x, self.wte.T)                  # [T, n_vocab]

        return logits, all_weights