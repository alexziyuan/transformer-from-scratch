import numpy as np
from model import GPT2
from loader import load_weights, get_config
from tokenizer.encode import encode, decode
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def load_model(bin_dir: str = "weights/bin") -> GPT2:
    """Load weights and instantiate the model."""
    weights = load_weights(bin_dir)
    config  = get_config(weights)
    print(f"Config: {config}")
    return GPT2(weights, config)


def greedy_sample(logits: np.ndarray) -> int:
    """Pick the token with the highest logit."""
    return int(np.argmax(logits))


def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample from the logit distribution scaled by temperature.

    temperature < 1.0  -> sharper distribution (more conservative)
    temperature = 1.0  -> unmodified distribution
    temperature > 1.0  -> flatter distribution (more random)

    Args:
        logits:      [n_vocab] raw logits for the next token
        temperature: scaling factor
    Returns:
        sampled token ID
    """
    if temperature == 0.0:
        return greedy_sample(logits)

    scaled = logits / temperature
    # Subtract max for numerical stability before softmax
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate(
    model: GPT2,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
) -> tuple[str, list[np.ndarray]]:
    """
    Autoregressive text generation.

    At each step:
      1. Run the full forward pass on the current token sequence
      2. Take the logits at the last position (the next-token prediction)
      3. Sample a token
      4. Append it to the sequence
      5. Repeat

    Args:
        model:          GPT2 instance
        prompt:         input string
        max_new_tokens: how many tokens to generate
        temperature:    sampling temperature (0 = greedy)
    Returns:
        generated_text: full string including prompt
        final_weights:  attention weights from the last forward pass
                        list of [n_head, seq_len, seq_len] per layer
    """
    token_ids = encode(prompt).tolist()
    final_weights = None

    for step in range(max_new_tokens):
        ids_arr = np.array(token_ids, dtype=np.int32)
        logits, weights = model.forward(ids_arr)    # [T, n_vocab], [n_layer, n_head, T, T]

        # Only the last position predicts the next token
        next_logits = logits[-1]                    # [n_vocab]
        next_token  = temperature_sample(next_logits, temperature)

        token_ids.append(next_token)
        final_weights = weights

        # Optional: stop at EOS token (50256 for GPT-2)
        if next_token == 50256:
            break

    return decode(np.array(token_ids)), final_weights


def get_attention_weights(
    model: GPT2,
    prompt: str,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Run a forward pass and return token strings + attention weights.
    This is the function the Flask visualization server will call.

    Args:
        model:  GPT2 instance
        prompt: input string
    Returns:
        tokens:  list of token strings (for axis labels in the heatmap)
        weights: list of length n_layer, each [n_head, seq_len, seq_len]
    """
    from tokenizer.encode import get_tokenizer
    enc = get_tokenizer()

    token_ids = encode(prompt)
    tokens    = [enc.decode([t]) for t in token_ids.tolist()]

    _, all_weights = model.forward(token_ids)

    return tokens, all_weights


if __name__ == "__main__":
    model = load_model()
    text, _ = generate(model, "The transformer architecture", max_new_tokens=30, temperature=0.8)
    print(text)