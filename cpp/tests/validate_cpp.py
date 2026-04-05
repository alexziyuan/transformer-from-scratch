# cpp/tests/validate_cpp.py
import subprocess
import json
import numpy as np
import sys
import os

sys.path.insert(0, "python")
from tokenizer.encode import encode
from loader import load_weights, get_config
from model import GPT2

BINARY    = "./cpp/build/transformer"
WEIGHTS   = "weights/bin"
ATOL      = 1e-2    # slightly looser than phase 1 — C++ uses float32 throughout

def get_cpp_logits(token_ids: list[int]) -> np.ndarray:
    """Call C++ binary with token IDs via stdin, parse JSON output."""
    ids_str = "\n".join(str(i) for i in token_ids)
    result  = subprocess.run(
        [BINARY, "--weights", WEIGHTS, "--json"],
        input=ids_str, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"C++ binary failed:\n{result.stderr}")

    data = json.loads(result.stdout)
    # top_k_logits is only top-k — for full comparison we need all logits
    # For now compare top-1 token agreement via top_k
    return data


def get_numpy_logits(token_ids: list[int]) -> np.ndarray:
    weights = load_weights(WEIGHTS)
    config  = get_config(weights)
    model   = GPT2(weights, config)
    ids_arr = np.array(token_ids, dtype=np.int32)
    logits, _ = model.forward(ids_arr)
    return logits


def validate(prompt: str) -> bool:
    print(f"\nPrompt: '{prompt}'")
    token_ids = encode(prompt).tolist()

    cpp_data   = get_cpp_logits(token_ids)
    np_logits  = get_numpy_logits(token_ids)

    # Compare top-1 next token from C++ vs NumPy last-position logits
    cpp_top1   = cpp_data["top_k_logits"][0]["token_id"]
    np_top1    = int(np_logits[-1].argmax())

    print(f"  C++ top-1 next token:   {cpp_top1}")
    print(f"  NumPy top-1 next token: {np_top1}")

    match = cpp_top1 == np_top1
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    return match


if __name__ == "__main__":
    prompts = [
        "The",
        "Hello, my name is",
        "The transformer architecture is",
        "In 1945, the war",
    ]

    print("=" * 55)
    print("Phase 2 Validation: C++ vs NumPy")
    print("=" * 55)

    results = [validate(p) for p in prompts]

    print("\n" + "=" * 55)
    print(f"Passed {sum(results)}/{len(results)}")
    if all(results):
        print("Phase complete. Ready for the next phase.")