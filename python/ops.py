# python/ops.py
import numpy as np


# ─────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Standard matrix multiplication. Wraps np.matmul so has a clear
    single replacement target for C++ port.

    Args:
        A: [..., M, K]
        B: [..., K, N]
    Returns:
        [..., M, N]
    """
    return np.matmul(A, B)

# ─────────────────────────────────────────────
#  Nonlinearities
# ─────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over the last axis.

    Subtracting the row maximum before exponentiation prevents overflow.
    This mathematically equivalent to the naive form but safe for large logits.
    Will do this exact implementation in C++.

    Args:
        x: [..., N]
    Returns:
        [..., N], same shape, each row sums to 1.0
    """
    # Shift by row max for numerical stability
    x_shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x_shifted)
    return e / e.sum(axis=-1, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit - GPT-2's activation function in the 
    Feed-Forward Network.

    Uses the tanh approximation from the original BERT/GPT papers:
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    This approximation is what GPT-2's weights were trained with.
    Do not substitute ReLu - the weights will not match.

    Args:
        x: any shape
    Returns:
        same shape
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def relu(x: np.ndarray) -> np.ndarray:
    """Standard ReLu. Included for completeness; not used in GPT-2."""
    return np.maximum(0.0, x)


# ─────────────────────────────────────────────
#  Normalization
# ─────────────────────────────────────────────

def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Layer Normalization (Ba et al. 2016).

    Normalizes across the last dimension (the feature/embedding dimension),
    then applies learned affine parameters gamma (scale) and beta (shift).

    Unlike BatchNorm, LayerNorm is computed independently per token,
    making it sequence-length agnostic — critical for autoregressive inference.

    Args:
        x:     [seq_len, d_model]
        gamma: [d_model]  — learned scale (called 'weight' in HuggingFace)
        beta:  [d_model]  — learned shift (called 'bias' in HuggingFace)
        eps:   small constant for numerical stability
    Returns:
        [seq_len, d_model]
    """
    mean = x.mean(axis=-1, keepdims=True)       # [seq_len, 1]
    var  = x.var(axis=-1, keepdims=True)        # [seq_len, 1]
    x_hat = (x - mean) / np.sqrt(var + eps)    # [seq_len, d_model]
    return gamma * x_hat + beta
    

# ─────────────────────────────────────────────
#  Attention
# ─────────────────────────────────────────────

def causal_mask(seq_len: int) -> np.ndarray:
    """
    Upper-triangular mask for autoregressive (causal) attention.

    Positions in the upper triangle (future tokens) are set to -inf,
    so they become zero after softmax. Lower triangle (past + current)
    is zero, leaving scores unchanged.

    Args:
        seq_len: sequence length T
    Returns:
        [T, T] mask with 0.0 on/below diagonal, -inf above
    """
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    mask[np.triu_indices(seq_len, k=1)] = float("-inf")
    return mask


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product atention (Vaswani et al. 2017).

    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    The scaling by sqrt(d_k) prevents the dot products from growing
    large in magnitude, which would push softmax into regions of
    near-zero gradient.

    Args:
        Q:    [seq_len, d_k]
        K:    [seq_len, d_k]
        V:    [seq_len, d_v]
        mask: [seq_len, seq_len] optional additive mask (use causal_mask())
    Returns:
        out:     [seq_len, d_v]   — attended values
        weights: [seq_len, seq_len] — attention probabilities (for visualization)
    """
    d_k = Q.shape[-1]
    scores = matmul(Q, K.T) / np.sqrt(d_k)    # [seq_len, seq_len]

    if mask is not None:
        scores = scores + mask                  # additive mask: -inf zeros out

    weights = softmax(scores)                  # [seq_len, seq_len]
    out = matmul(weights, V)                   # [seq_len, d_v]
    return out, weights


def multi_head_attention(
    x: np.ndarray,
    c_attn_w: np.ndarray,
    c_attn_b: np.ndarray,
    c_proj_w: np.ndarray,
    c_proj_b: np.ndarray,
    n_head: int,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-head self-attention as used in GPT-2.

    GPT-2 uses a single [d_model, 3*d_model] weight matrix (c_attn) to
    project the input into Q, K, V in one matmul - more cache friendly
    than three separate projections.

    After attention, a second linear layer (c_proj) miaxes the heads
    back into d_model space.

    Args:
        x:        [seq_len, d_model]
        c_attn_w: [d_model, 3 * d_model]  — fused QKV projection weight
        c_attn_b: [3 * d_model]           — fused QKV projection bias
        c_proj_w: [d_model, d_model]      — output projection weight
        c_proj_b: [d_model]               — output projection bias
        n_head:   number of attention heads
        mask:     [seq_len, seq_len]
    Returns:
        out:           [seq_len, d_model]
        all_weights:   [n_head, seq_len, seq_len]  — per-head attention weights
    """
    T, d_model = x.shape
    d_head = d_model // n_head

    # Fused QKV projection: one matmul instead of three
    qkv = matmul(x, c_attn_w) + c_attn_b          # [T, 3 * d_model]
    Q, K, V = np.split(qkv, 3, axis=-1)            # each [T, d_model]

    # Split into heads: [T, d_model] -> [T, n_head, d_head] -> [n_head, T, d_head]
    def split_heads(t):
        return t.reshape(T, n_head, d_head).transpose(1, 0, 2)

    Q = split_heads(Q)   # [n_head, T, d_head]
    K = split_heads(K)
    V = split_heads(V)

    # Run attention independently per head
    head_outputs = []
    all_weights  = []

    for h in range(n_head):
        out_h, w_h = scaled_dot_product_attention(Q[h], K[h], V[h], mask)
        head_outputs.append(out_h)     # [T, d_head]
        all_weights.append(w_h)        # [T, T]

    # Concatenate heads: [n_head, T, d_head] -> [T, d_model]
    out = np.concatenate(head_outputs, axis=-1)     # [T, d_model]
    all_weights = np.stack(all_weights, axis=0)     # [n_head, T, T]

    # Output projection
    out = matmul(out, c_proj_w) + c_proj_b          # [T, d_model]

    return out, all_weights


# ─────────────────────────────────────────────
#  Feed-Forward Network
# ─────────────────────────────────────────────

def feed_forward(
    x: np.ndarray,
    fc_w: np.ndarray,
    fc_b: np.ndarray,
    proj_w: np.ndarray,
    proj_b: np.ndarray,
) -> np.ndarray:
    """
    Position-wise Feed-Forward Network (FFN).

    Two linear layers with GELU in between. The inner dimension expands
    to 4 * d_model (3072 for GPT-2 small) then contracts back to d_model.

    This is applied identically and independently to eah token position
    - hence "position-wise."


    Args:
        x:      [seq_len, d_model]
        fc_w:   [d_model, 4 * d_model]   — expand weight
        fc_b:   [4 * d_model]
        proj_w: [4 * d_model, d_model]   — contract weight
        proj_b: [d_model]
    Returns:
        [seq_len, d_model]
    """
    h = gelu(matmul(x, fc_w) + fc_b)    # [seq_len, 4 * d_model]
    return matmul(h, proj_w) + proj_b   # [seq_len, d_model]