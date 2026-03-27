import numpy as np
import tiktoken


def get_tokenizer() -> tiktoken.Encoding:
    """Return GPT-2's BPE tokenizer."""
    return tiktoken.get_encoding("gpt2")


def encode(text: str) -> np.ndarray:
    """
    Encode a string to GPT-2 token IDs.

    Args:
        text: input string
    Returns:
        [seq_len] int32 array of token IDs
    """
    enc = get_tokenizer()
    return np.array(enc.encode(text), dtype=np.int32)


def decode(token_ids: np.ndarray) -> str:
    """
    Decode GPT-2 token IDs back to a string.

    Args:
        token_ids: [seq_len] integer array
    Returns:
        decoded string
    """
    enc = get_tokenizer()
    return enc.decode(token_ids.tolist())


def decode_single(token_id: int) -> str:
    """Decode a single token ID."""
    return decode(np.array([token_id]))