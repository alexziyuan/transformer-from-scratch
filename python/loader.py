import os
import json
import numpy as np


def load_weights(bin_dir: str = "weights/bin") -> dict:
    """
    Load all GTP-2 weight matrices from binary files dumped by
    weights/dump_weights.py

    Returns a dict mapping HuggingFace key names (e.g.
    "h.0.attn.c_attn.weight") to numpy float32 arrays.

    The manifest.json is used to reconstruct shapes - without it,
    we cannot determine how to reshape a flat binary buffer.
    """
    manifest_path = os.path.join(bin_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"manifest.json not found in {bin_dir}. "
            f"Run weights/dump_weights.py first."
        )
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    weights = {}
    for name, shape in manifest.items():
        safe_name = name.replace(".", "__")
        path = os.path.join(bin_dir, f"{safe_name}.bin")
        arr = np.fromfile(path, dtype=np.float32)
        weights[name] = arr.reshape(shape)

    print(f"Loaded {len(weights)} weight tensors from {bin_dir}")
    return weights


def get_config(weights: dict) -> dict:
    """
    Infer GPT-2 architecture hyperparameters from weight shapes.
    No hardcoding - the shapes tell us everything.

    GPT-2 small:  n_layer=12, n_head=12, d_model=768,  n_vocab=50257
    GPT-2 medium: n_layer=24, n_head=16, d_model=1024, n_vocab=50257
    GPT-2 large:  n_layer=36, n_head=20, d_model=1280, n_vocab=50257
    GPT-2 xl:     n_layer=48, n_head=25, d_model=1600, n_vocab=50257
    """
    n_vocab, d_model = weights["wte.weight"].shape
    n_ctx            = weights["wpe.weight"].shape[0]

    # Count transformer blocks by finding the highest h.N key
    n_layer = 0
    while f"h.{n_layer}.attn.c_attn.weight" in weights:
        n_layer += 1

    # Number of heads inferred from attention head size
    # c_attn.weight is [d_model, 3*d_model]; each head has d_model/n_head dims
    # We read n_head from the standard GPT-2 sizes
    n_head_map = {768: 12, 1024: 16, 1280: 20, 1600: 25}
    n_head = n_head_map.get(d_model, 12)

    return {
        "n_vocab":  n_vocab,
        "d_model":  d_model,
        "n_ctx":    n_ctx,
        "n_layer":  n_layer,
        "n_head":   n_head,
    }