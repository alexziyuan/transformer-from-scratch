import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from loader import load_weights, get_config
from model import GPT2
from tokenizer.encode import encode


def get_hf_logits(prompt: str) -> np.ndarray:
    """Run prompt through HuggingFace GPT-2 and return logits."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    hf_model  = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = hf_model(**inputs)

    return outputs.logits[0].numpy()    # [seq_len, n_vocab]


def get_numpy_logits(prompt: str) -> np.ndarray:
    """Run prompt through our NumPy model and return logits."""
    weights = load_weights()
    config  = get_config(weights)
    model   = GPT2(weights, config)

    token_ids = encode(prompt)
    logits, _ = model.forward(token_ids)
    return logits    # [seq_len, n_vocab]


def validate(prompt: str, atol: float = 1e-3) -> bool:
    """
    Compare logits from HuggingFace and our NumPy implementation.

    We use atol=1e-3 (not 1e-5) because HuggingFace may use float16
    internally on some hardware, introducing small rounding differences.

    Args:
        prompt: test string
        atol:   absolute tolerance for np.allclose
    Returns:
        True if outputs match within tolerance
    """
    print(f"Prompt: '{prompt}'")

    hf_logits  = get_hf_logits(prompt)
    np_logits  = get_numpy_logits(prompt)

    print(f"  HuggingFace logits shape: {hf_logits.shape}")
    print(f"  NumPy model  logits shape: {np_logits.shape}")

    max_diff = np.abs(hf_logits - np_logits).max()
    mean_diff = np.abs(hf_logits - np_logits).mean()

    print(f"  max absolute difference:  {max_diff:.6f}")
    print(f"  mean absolute difference: {mean_diff:.6f}")

    # Check top-1 token agreement (more lenient, more meaningful)
    hf_top1 = hf_logits.argmax(axis=-1)
    np_top1 = np_logits.argmax(axis=-1)
    token_agreement = (hf_top1 == np_top1).mean() * 100
    print(f"  top-1 token agreement:    {token_agreement:.1f}%")

    passed = np.allclose(hf_logits, np_logits, atol=atol)
    print(f"  Result: {'PASS' if passed else 'FAIL'} (atol={atol})")
    return passed


if __name__ == "__main__":
    prompts = [
        "The",
        "Hello, my name is",
        "The transformer architecture is",
        "In 1945, the war",
    ]

    results = []
    print("=" * 60)
    print("Validation: NumPy model vs HuggingFace GPT-2")
    print("=" * 60)

    for p in prompts:
        print()
        results.append(validate(p))

    print()
    print("=" * 60)
    passed = sum(results)
    print(f"Passed {passed}/{len(results)} validation checks.")
    if passed == len(results):
        print("Phase 1 complete. Ready to port to C++.")
    else:
        print("Failures detected. Debug before proceeding to Phase 2.")