# python/tests/test_model.py
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loader import load_weights, get_config
from model import GPT2
from tokenizer.encode import encode

ATOL = 1e-3

def test_output_shapes():
    """Forward pass output shapes are correct."""
    print("output shapes")
    weights = load_weights()
    config  = get_config(weights)
    model   = GPT2(weights, config)

    prompt    = "Hello"
    token_ids = encode(prompt)
    T         = len(token_ids)

    logits, all_weights = model.forward(token_ids)

    assert logits.shape == (T, config["n_vocab"]), \
        f"logits shape {logits.shape}"
    print(f"  PASS [logits: {logits.shape}]")

    assert len(all_weights) == config["n_layer"], \
        f"expected {config['n_layer']} layers, got {len(all_weights)}"
    print(f"  PASS [n_layer attention weight tensors]")

    for i, w in enumerate(all_weights):
        expected = (config["n_head"], T, T)
        assert w.shape == expected, \
            f"layer {i} weights shape {w.shape}, expected {expected}"
    print(f"  PASS [per-layer attention weight shapes: {all_weights[0].shape}]")


def test_attention_weights_are_distributions():
    """Each head's attention weights should sum to 1 per query token."""
    print("attention weights are valid distributions")
    weights = load_weights()
    config  = get_config(weights)
    model   = GPT2(weights, config)

    prompt    = "The cat sat on the mat"
    token_ids = encode(prompt)
    _, all_weights = model.forward(token_ids)

    for layer_idx, layer_weights in enumerate(all_weights):
        row_sums = layer_weights.sum(axis=-1)   # [n_head, T]
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise AssertionError(
                f"Layer {layer_idx} attention weights don't sum to 1. "
                f"Max deviation: {np.abs(row_sums - 1.0).max():.2e}"
            )

    print(f"  PASS [all {len(all_weights)} layers, all heads]")


def test_logits_match_hf():
    """Logits match HuggingFace output within tolerance."""
    print("logits match HuggingFace")
    from validate import validate
    assert validate("The transformer architecture is", atol=ATOL), \
        "logits do not match HuggingFace"
    print(f"  PASS [atol={ATOL}]")


if __name__ == "__main__":
    print("=" * 50)
    print("model.py integration tests")
    print("=" * 50)
    test_output_shapes()
    test_attention_weights_are_distributions()
    test_logits_match_hf()
    print("=" * 50)
    print("All model tests passed.")