"""
data.py — Data pipeline for MiniLLM-MoE training.

Three dataset modes:
  ShakespeareDataset   — auto-downloaded ~1MB, for smoke tests
  BinaryTokenDataset   — memory-mapped .bin file, for custom text
  StreamingTextDataset — HuggingFace streaming, for large-scale training

All return (x, y) tensor pairs where y = x shifted left by 1 token.
"""

import os
import urllib.request
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _get_encoder():
    """Lazy-import tiktoken to avoid hard dependency at module import."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _tokenize_text(text: str) -> np.ndarray:
    """Tokenize a string to a uint16 numpy array using GPT-2 BPE."""
    enc = _get_encoder()
    return np.array(enc.encode_ordinary(text), dtype=np.uint16)


# ─── Datasets ─────────────────────────────────────────────────────────────────

class ShakespeareDataset(Dataset):
    """
    Tiny Shakespeare (~1MB) for smoke tests and quick iteration.

    Auto-downloads from Karpathy's char-rnn repo on first use,
    tokenizes with GPT-2 BPE, and caches as a .bin file.
    90% train / 10% validation split by token index.

    Returns (x, y) pairs of shape (context_len,) where y = x shifted by 1.
    """

    def __init__(self, context_len: int, split: str = "train"):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.context_len = context_len

        os.makedirs(_DATA_DIR, exist_ok=True)
        bin_path = os.path.join(_DATA_DIR, "shakespeare.bin")

        if not os.path.exists(bin_path):
            print("Downloading Shakespeare dataset...")
            txt_path = os.path.join(_DATA_DIR, "shakespeare.txt")
            urllib.request.urlretrieve(SHAKESPEARE_URL, txt_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = _tokenize_text(text)
            tokens.tofile(bin_path)
            print(f"  {len(tokens):,} tokens cached to {bin_path}")

        data      = np.memmap(bin_path, dtype=np.uint16, mode="r")
        split_idx = int(0.9 * len(data))
        self.data = data[:split_idx] if split == "train" else data[split_idx:]

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(
            self.data[idx : idx + self.context_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


class BinaryTokenDataset(Dataset):
    """
    Memory-mapped dataset for pre-tokenized uint16 .bin files.

    Uses np.memmap so the file is never loaded into RAM — safe for
    datasets larger than system memory. Created via prepare_text_file().
    """

    def __init__(self, bin_path: str, context_len: int):
        self.context_len = context_len
        self.data        = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(
            self.data[idx : idx + self.context_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


class StreamingTextDataset(IterableDataset):
    """
    HuggingFace streaming text dataset — never downloads the full corpus.

    Streams documents, tokenizes on the fly, buffers tokens into fixed-length
    chunks. An eot_token is appended between documents to mark boundaries.

    Supports any HF text dataset with a text column. Tested with:
      HuggingFaceFW/fineweb — high-quality web text
      allenai/dolma         — open web + books + code

    buffer_size: minimum number of tokens to accumulate before yielding,
    avoiding the overhead of creating many tiny tensors.
    """

    def __init__(
        self,
        dataset_name: str,
        context_len: int,
        split: str = "train",
        text_column: str = "text",
        buffer_size: int = 10_000,
    ):
        self.dataset_name = dataset_name
        self.context_len  = context_len
        self.split        = split
        self.text_column  = text_column
        self.buffer_size  = buffer_size

    def __iter__(self):
        from datasets import load_dataset

        enc     = _get_encoder()
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        token_buffer: list = []

        for doc in dataset:
            text = doc.get(self.text_column, "")
            if not text:
                continue
            token_buffer.extend(enc.encode_ordinary(text))
            token_buffer.append(enc.eot_token)              # document separator

            while len(token_buffer) >= self.buffer_size:
                chunk = token_buffer[: self.context_len + 1]
                token_buffer = token_buffer[self.context_len :]
                if len(chunk) < self.context_len + 1:
                    break
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    dataset_name: str,
    context_len: int,
    batch_size: int,
    num_workers: int = 2,
    bin_path: Optional[str] = None,
    hf_dataset_name: Optional[str] = None,
    hf_text_col: str = "text",
) -> Tuple[DataLoader, DataLoader]:
    """
    Single entry point that returns (train_loader, val_loader).

    For streaming datasets (fineweb, dolma, custom_hf, reasoning) the
    val_loader uses Shakespeare so perplexity is comparable across runs.

    Args:
        dataset_name:    one of [shakespeare, binary, fineweb, dolma,
                         custom_hf, gsm8k, metamath, openhermes]
        context_len:     matches model's context_len
        batch_size:      micro-batch size per forward pass
        num_workers:     DataLoader workers
        bin_path:        required only when dataset_name == 'binary'
        hf_dataset_name: HuggingFace dataset repo id (for custom_hf)
        hf_text_col:     text column name in the HF dataset (for custom_hf)
    """
    pin = True  # always pin memory for faster CPU→GPU transfer

    if dataset_name == "shakespeare":
        train_ds = ShakespeareDataset(context_len, split="train")
        val_ds   = ShakespeareDataset(context_len, split="val")
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin),
        )

    if dataset_name == "binary":
        if bin_path is None:
            raise ValueError("--bin_path is required for dataset_name='binary'")
        full_ds   = BinaryTokenDataset(bin_path, context_len)
        n_val     = max(1, int(0.1 * len(full_ds)))
        n_train   = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin),
        )

    if dataset_name in ("fineweb", "dolma"):
        hf_names = {
            "fineweb": "HuggingFaceFW/fineweb",
            "dolma":   "allenai/dolma",
        }
        train_ds = StreamingTextDataset(hf_names[dataset_name], context_len)
        val_ds   = ShakespeareDataset(context_len, split="val")  # consistent eval
        return (
            DataLoader(train_ds, batch_size=batch_size,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin),
        )

    if dataset_name == "custom_hf":
        if hf_dataset_name is None:
            raise ValueError(
                "--hf_dataset_name is required for dataset_name='custom_hf'"
            )
        train_ds = StreamingTextDataset(
            hf_dataset_name, context_len, text_column=hf_text_col
        )
        val_ds = ShakespeareDataset(context_len, split="val")
        return (
            DataLoader(train_ds, batch_size=batch_size,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin),
        )

    # Reasoning datasets (gsm8k, metamath, openhermes)
    _reasoning_datasets = ("gsm8k", "metamath", "openhermes")
    if dataset_name in _reasoning_datasets:
        from reasoning import ReasoningStreamDataset
        train_ds = ReasoningStreamDataset(dataset_name, context_len)
        val_ds   = ShakespeareDataset(context_len, split="val")
        return (
            DataLoader(train_ds, batch_size=batch_size,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin),
        )

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        "Choose from: shakespeare, binary, fineweb, dolma, custom_hf, "
        "gsm8k, metamath, openhermes"
    )


# ─── Utility ──────────────────────────────────────────────────────────────────

def prepare_text_file(input_path: str, output_path: str) -> None:
    """
    Tokenize any .txt file and save as a uint16 .bin for fast training.

    The output can be loaded with BinaryTokenDataset without ever holding
    the full corpus in RAM. Token ids are stored as uint16 (max id 50256
    fits in uint16, saving ~50% disk vs int32).

    Args:
        input_path:  path to .txt file
        output_path: destination .bin file
    """
    print(f"Tokenizing {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = _tokenize_text(text)
    tokens.tofile(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Tokens:    {len(tokens):,}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Saved to:  {output_path}")
