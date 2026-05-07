"""
reasoning.py — Chain-of-thought reasoning model utilities.

Provides dataset streaming, CoT formatting, and output parsing for
reasoning models trained with <think>...</think> patterns.

Architecture note: the underlying MiniLLM model is identical for
standard and reasoning modes. The difference is training data format
and (optionally) dataset choice. The model learns to generate
<think>...</think> blocks as part of its output vocabulary.
"""

import re
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import IterableDataset


# ── Special token strings ──────────────────────────────────────────────────────
# These are regular strings tokenized by GPT-2 BPE — no vocab extension needed.
THINK_START  = "<think>"
THINK_END    = "</think>"
ANSWER_START = "<answer>"
ANSWER_END   = "</answer>"


# ── Supported reasoning datasets ──────────────────────────────────────────────
REASONING_DATASETS = {
    "gsm8k": {
        "label":        "GSM8K — Grade school math word problems",
        "hf_name":      "openai/gsm8k",
        "hf_config":    "main",
        "split":        "train",
        "question_col": "question",
        "answer_col":   "answer",
        "desc": (
            "8,500 grade-school math problems requiring multi-step reasoning. "
            "Answers include step-by-step working followed by #### <final number>."
        ),
    },
    "metamath": {
        "label":        "MetaMathQA — Augmented math reasoning",
        "hf_name":      "meta-math/MetaMathQA",
        "hf_config":    None,
        "split":        "train",
        "question_col": "query",
        "answer_col":   "response",
        "desc": (
            "395K math QA pairs with full chain-of-thought responses. "
            "Good for training strong mathematical reasoning."
        ),
    },
    "openhermes": {
        "label":        "OpenHermes 2.5 — General instruction + reasoning",
        "hf_name":      "teknium/OpenHermes-2.5",
        "hf_config":    None,
        "split":        "train",
        "question_col": "instruction",
        "answer_col":   "output",
        "desc": (
            "1M+ diverse instruction-following examples covering reasoning, "
            "coding, math, and general knowledge."
        ),
    },
}


# ── CoT text formatting ────────────────────────────────────────────────────────

def format_cot_example(question: str, reasoning: str, answer: str) -> str:
    """
    Format a Q&A pair as a chain-of-thought training example.

    The model is trained to generate the full sequence including the
    <think> block so it learns the reasoning pattern.
    """
    return (
        f"Question: {question.strip()}\n"
        f"{THINK_START}\n"
        f"{reasoning.strip()}\n"
        f"{THINK_END}\n"
        f"{ANSWER_START}\n"
        f"{answer.strip()}\n"
        f"{ANSWER_END}"
    )


def _parse_gsm8k_answer(raw_answer: str) -> Tuple[str, str]:
    """
    Split GSM8K answer into (reasoning_steps, final_answer).
    GSM8K uses '####' as the delimiter between working and answer.
    """
    if "####" in raw_answer:
        parts = raw_answer.split("####", 1)
        return parts[0].strip(), parts[1].strip()
    return raw_answer.strip(), ""


def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from generated text.
    Keeps content in <answer>...</answer> tags (tags stripped too).
    Falls back to returning the full text if no answer tags found.
    """
    # Remove thinking blocks entirely
    text = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL
    ).strip()

    # Extract answer content if present
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    return text


# ── Streaming reasoning dataset ───────────────────────────────────────────────

class ReasoningStreamDataset(IterableDataset):
    """
    Streams a HuggingFace reasoning dataset and formats examples as CoT text.

    Tokenizes each formatted example using GPT-2 BPE and yields
    (x, y) token pairs for language modelling.

    The model is trained to predict the entire sequence, including the
    <think> and <answer> blocks — this teaches it the reasoning format.
    """

    def __init__(
        self,
        dataset_key: str,
        context_len: int,
        buffer_size: int = 5_000,
    ):
        if dataset_key not in REASONING_DATASETS:
            raise ValueError(
                f"Unknown reasoning dataset '{dataset_key}'. "
                f"Choose from: {list(REASONING_DATASETS)}"
            )
        self.meta        = REASONING_DATASETS[dataset_key]
        self.context_len = context_len
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        import tiktoken
        from datasets import load_dataset

        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token

        load_kwargs = dict(
            split=self.meta["split"],
            streaming=True,
            trust_remote_code=True,
        )
        if self.meta["hf_config"]:
            load_kwargs["name"] = self.meta["hf_config"]

        dataset = load_dataset(self.meta["hf_name"], **load_kwargs)

        q_col = self.meta["question_col"]
        a_col = self.meta["answer_col"]

        token_buffer: list = []

        for row in dataset:
            question = row.get(q_col, "")
            raw_ans  = row.get(a_col, "")
            if not question or not raw_ans:
                continue

            # Dataset-specific formatting
            if self.meta["hf_name"] == "openai/gsm8k":
                reasoning, final = _parse_gsm8k_answer(raw_ans)
                text = format_cot_example(question, reasoning, final)
            else:
                # For MetaMathQA / OpenHermes: the response IS the reasoning
                text = format_cot_example(question, raw_ans, "")

            token_buffer.extend(enc.encode_ordinary(text))
            token_buffer.append(eot)

            while len(token_buffer) >= self.buffer_size:
                chunk = token_buffer[: self.context_len + 1]
                token_buffer = token_buffer[self.context_len :]
                if len(chunk) < self.context_len + 1:
                    break
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]
