# weights/dump_weights.py
import os
import numpy as np
from transformers import GPT2Model

def dump_weights(model_name: str = "gpt2", out_dir: str = "weights/bin") -> None:
    """
    Load GPT-2 weights from HuggingFace and dump each tensor as a raw
    float32 binary file. Directory structure mirrors the HuggingFace
    state_dict key names (dots replaced with slashes).

    Args:
        model_name: HuggingFace model identifier. "gpt2" is the 117M param model.
        out_dir: Directory to write .bin files into.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2Model.from_pretrained(model_name)
    state_dict = model.state_dict()

    manifest = {} # name -> shape, for loader reference

    # Dump the weights of each layer
    for name, tensor in state_dict.items():
        arr = tensor.cpu().numpy().astype(np.float32) # ensure float32
        # Replace dots with double-underscore so filenames stay flat
        safe_name = name.replace('.', '__')
        path = os.path.join(out_dir, f"{safe_name}.bin")
        arr.tofile(path) # write raw binary
        manifest[name] = list(arr.shape)
        print(f"  {name:60s} {str(arr.shape)}")

    # Write manifest so the loader knows shapes without reloading the model
    import json
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDumped {len(manifest)} tensors to {out_dir}/")

if __name__ == "__main__":
    dump_weights()