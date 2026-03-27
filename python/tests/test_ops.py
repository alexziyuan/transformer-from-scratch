import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ops import softmax, gelu, layer_norm, scaled_dot_product_attention, multi_head_attention

# Tolerance for float32 comparisons
ATOL = 1e-5
RTOL =1e-5

def assert_close(a: np.ndarray, b: np.ndarray, name: str) -> None:
    if not np.allclose(a, b, atol=ATOL, rtol=RTOL):
        max_diff = np.abs(a - b).max()
        raise AssertionError(f"FAIL [{name}] max_diff={max_diff:.2e}")
    print(f"  PASS [{name}]")


def test_softmax():
    print("softmax")
    rng = np.random.default_rng(0)

    # Standard case
    x = rng.standard_normal((4, 16)).astype(np.float32)
    ref = F.softmax(torch.tensor(x), dim=-1).numpy()
    assert_close(softmax(x), ref, "random_input")

    # Large values - numerical stability check
    x_large = x * 100.0
    ref_large = F.softmax(torch.tensor(x_large), dim=-1).numpy()
    assert_close(softmax(x_large), ref_large, "large values (stability)")

    # Rows sum to 1
    out = softmax(x)
    row_sums = out.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"rows don't sum to 1: {row_sums}"
    print("  PASS [rows sum to 1.0]")


def test_gelu():
    print("gelu")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    ref = F.gelu(torch.tensor(x), approximate="tanh").numpy()
    assert_close(gelu(x), ref, "random input")

    # Known values
    assert_close(gelu(np.array([0.0])), np.array([0.0]), "gelu(0) = 0")
    # gelu is approximately linear for large positive x
    val = gelu(np.array([10.0]))[0]
    assert abs(val - 10.0) < 0.01, f"gelu(10) should be ~10, got {val}"
    print("  PASS [known values]")


def test_layer_norm():
    print("layer_norm")
    rng = np.random.default_rng(2)
    T, D = 6, 64

    x     = rng.standard_normal((T, D)).astype(np.float32)
    gamma = rng.standard_normal((D,)).astype(np.float32)
    beta  = rng.standard_normal((D,)).astype(np.float32)

    # PyTorch reference
    ln = torch.nn.LayerNorm(D, elementwise_affine=True, eps=1e-5)
    ln.weight.data = torch.tensor(gamma)
    ln.bias.data   = torch.tensor(beta)
    ref = ln(torch.tensor(x)).detach().numpy()

    assert_close(layer_norm(x, gamma, beta), ref, "random input")

    # With identity gamma/beta, output should have mean~0 and std~1 per row
    ones  = np.ones(D, dtype=np.float32)
    zeros = np.zeros(D, dtype=np.float32)
    out   = layer_norm(x, ones, zeros)
    row_means = out.mean(axis=-1)
    row_stds  = out.std(axis=-1)
    assert np.allclose(row_means, 0.0, atol=1e-5), f"means not 0: {row_means}"
    assert np.allclose(row_stds,  1.0, atol=1e-3), f"stds not 0: {row_stds}"
    print("  PASS [identity gamma/beta: mean≈0, std≈1]")

def test_scaled_dot_product_attention():
    print("scaled_dot_product_attention")
    rng = np.random.default_rng(3)
    T, d_k, d_v = 5, 16, 16
    
    Q = rng.standard_normal((T, d_k)).astype(np.float32)
    K = rng.standard_normal((T, d_k)).astype(np.float32)
    V = rng.standard_normal((T, d_v)).astype(np.float32)

    out, weights = scaled_dot_product_attention(Q, K, V)

    # Shape checks
    assert out.shape     == (T, d_v), f"outshape {out.shape}"
    assert weights.shape == (T, T),   f"weights shape {weights.shape}"
    print("  PASS [output shapes]")

    # Weights should be a valid probability distribution
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5), "weights don't sum to 1"
    print("  PASS [attention weights sum to 1]")

    #With causal mask: upper triangle of weights should be ~0
    from ops import causal_mask
    mask = causal_mask(T)
    _, masked_weights = scaled_dot_product_attention(Q, K, V, mask)
    upper = masked_weights[np.triu_indices(T, k=1)]
    assert np.allclose(upper, 0.0, atol=1e-6), f"causal mask leak: max={upper.max():.2e}"
    print("  PASS [causal mask: upper triangle ≈ 0]")

def test_multi_head_attention():
    print("multi_head_attention")
    rng = np.random.default_rng(4)
    T, d_model, n_head = 5, 64, 4
    
    x        = rng.standard_normal((T, d_model)).astype(np.float32)
    c_attn_w = rng.standard_normal((d_model, 3 * d_model)).astype(np.float32)
    c_attn_b = rng.standard_normal((3 * d_model,)).astype(np.float32)
    c_proj_w = rng.standard_normal((d_model, d_model)).astype(np.float32)
    c_proj_b = rng.standard_normal((d_model,)).astype(np.float32)

    from ops import causal_mask
    mask = causal_mask(T)
    out, weights = multi_head_attention(
        x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, n_head, mask
    )

    assert out.shape     == (T, d_model),       f"out shape {out.shape}"
    assert weights.shape == (n_head, T, T), f"weights shape {weights.shape}"
    print("  PASS [output shapes]")

    # Per-head attention weights sum to 1 along key dimension
    sums = weights.sum(axis=-1)  # [n_head, T]
    assert np.allclose(sums, 1.0, atol=1e-5), "per-head weights don't sum to 1"
    print("  PASS [per-head weights sum to 1]")


if __name__ == "__main__":
    print("=" * 50)
    print("ops.py unit tests")
    print("=" * 50)
    test_softmax()
    test_gelu()
    test_layer_norm()
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    print("=" * 50)
    print("All tests passed.")