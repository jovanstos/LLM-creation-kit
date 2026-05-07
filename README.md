# LLM Creation Kit

Build your own large language model from scratch — from a 30M smoke test all the way to a 1.5B flagship model. You can also edit the code if you'd like to go further. Includes an interactive TUI wizard, reasoning model support, and Mixture-of-Experts architecture.

```
 ██╗     ██╗     ███╗   ███╗    ██╗  ██╗██╗████████╗
 ██║     ██║     ████╗ ████║    ██║ ██╔╝██║╚══██╔══╝
 ██║     ██║     ██╔████╔██║    █████╔╝ ██║   ██║
 ██║     ██║     ██║╚██╔╝██║    ██╔═██╗ ██║   ██║
 ███████╗███████╗██║ ╚═╝ ██║    ██║  ██╗██║   ██║
 ╚══════╝╚══════╝╚═╝     ╚═╝    ╚═╝  ╚═╝╚═╝   ╚═╝
              C R E A T I O N   K I T  v1.0
```

---

## What is this?

A self-contained Python toolkit for training decoder-only transformer LLMs on consumer hardware (tested on RTX 4070 12 GB). The architecture mirrors modern models like LLaMA-2/3 and Mixtral:

| Component           | Choice                            | Why                                |
| ------------------- | --------------------------------- | ---------------------------------- |
| Positional encoding | RoPE                              | Better length generalization       |
| Normalization       | RMSNorm (pre-norm)                | Faster, more stable than LayerNorm |
| Attention           | Grouped Query Attention (GQA)     | Smaller KV cache at inference      |
| FFN                 | Mixture of Experts (MoE) + SwiGLU | Same compute, higher capacity      |
| Tokenizer           | GPT-2 BPE (tiktoken)              | No training required               |
| Embeddings          | Tied input/output weights         | ~10% fewer parameters              |

A 1.5B MoE model activates only ~25% of its FFN parameters per token — the quality of a large model at the compute of a smaller one.

---

## Quick start

### 1. Install dependencies

```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Then everything else
pip install -r requirements.txt
```

### 2. Run the interactive wizard

```bash
python kit.py
```

The wizard walks you through every choice and launches training for you. That's it.

---

## The Wizard (`kit.py`)

`kit.py` is a step-by-step TUI that guides you through building your model:

```
Step 1  Model Type          Standard LM  or  Reasoning (chain-of-thought)
Step 2  Model Scale         30M → 1.5B, or define a custom architecture
Step 3  Dataset             Shakespeare, FineWeb, Dolma, any HF dataset, or your .txt file
Step 4  Hyperparameters     Smart defaults per preset, all adjustable
Step 5  Early Stopping      Halt training when val_loss plateaus
Step 6  Advanced Options    8-bit AdamW, torch.compile, W&B logging, GGUF export
Step 7  Context Length      Override the default for your preset
Step 8  Output              Checkpoint and log directories
        Review              Full configuration table before launch
        Save Config         Export to YAML for reproducibility
        Launch              Streams training output live
```

Resume from a saved config:

```bash
python kit.py --load my_llm_config.yaml
```

---

## Training directly (without the wizard)

```bash
# 30M smoke test — verify your setup (~10 min)
python train.py --preset 30m --dataset shakespeare --max_steps 500

# 70M on Shakespeare (~1 hour)
python train.py --preset 70m --dataset shakespeare --max_steps 5000 \
  --batch_size 12 --grad_accum 4 --lr 3e-4 --patience 3

# 125M on FineWeb (~8 hours)
python train.py --preset 125m --dataset fineweb --max_steps 50000 \
  --batch_size 8 --grad_accum 8 --use_8bit_adam --patience 5

# 1B on FineWeb (~1 week)
python train.py --preset 1b --dataset fineweb --max_steps 500000 \
  --batch_size 4 --grad_accum 16 --use_8bit_adam --patience 5

# Resume interrupted training
python train.py --preset 1b --dataset fineweb --max_steps 500000 \
  --batch_size 4 --grad_accum 16 --use_8bit_adam \
  --resume checkpoints/best_1b.pt
```

### Custom HuggingFace dataset

Stream any text dataset from HuggingFace — never downloads fully:

```bash
python train.py --preset 70m \
  --dataset custom_hf \
  --hf_dataset_name "HuggingFaceFW/fineweb-edu" \
  --hf_text_col "text" \
  --max_steps 10000
```

### Custom text file

```bash
# Tokenize your file first
python -c "from data import prepare_text_file; prepare_text_file('corpus.txt', 'data/corpus.bin')"

# Train on it
python train.py --preset 70m --dataset binary --bin_path data/corpus.bin --max_steps 5000
```

---

## Reasoning Models

Reasoning models output structured chain-of-thought before their final answer, similar to DeepSeek-R1 and o1-style models. The pattern used:

```
Question: What is 12 × 34?
<think>
I need to multiply 12 by 34.
12 × 30 = 360
12 × 4  = 48
360 + 48 = 408
</think>
<answer>
408
</answer>
```

### Train a reasoning model

```bash
# Quick test — Shakespeare with reasoning format
python train.py --preset 30m --dataset shakespeare --model_type reasoning --max_steps 500

# Math reasoning — GSM8K (8,500 grade-school problems)
python train.py --preset 70m --dataset gsm8k --model_type reasoning --max_steps 5000

# Larger math reasoning — MetaMathQA (395K augmented problems)
python train.py --preset 125m --dataset metamath --model_type reasoning \
  --max_steps 50000 --use_8bit_adam

# General reasoning — OpenHermes 2.5 (1M+ instruction/reasoning pairs)
python train.py --preset 350m --dataset openhermes --model_type reasoning \
  --max_steps 100000 --use_8bit_adam
```

> **Tip:** For best results, pre-train on FineWeb first (standard mode), then fine-tune on a reasoning dataset. This mirrors how production reasoning models are built.

### Generate with a reasoning model

```bash
# Hide the thinking process (show only the answer)
python generate.py \
  --checkpoint checkpoints/best_70m.pt \
  --prompt "Question: If a train travels 60 mph for 2.5 hours, how far does it go?" \
  --max_new_tokens 300

# Show the full thinking chain
python generate.py \
  --checkpoint checkpoints/best_70m.pt \
  --prompt "Question: If a train travels 60 mph for 2.5 hours, how far does it go?" \
  --max_new_tokens 300 \
  --show_thinking
```

---

## Model Scale Reference

| Preset | Params | VRAM   | Time (RTX 4070) | Context |
| ------ | ------ | ------ | --------------- | ------- |
| 30m    | 30M    | ~2 GB  | ~10 min         | 512     |
| 70m    | 70M    | ~3 GB  | ~1 hr           | 1024    |
| 125m   | 125M   | ~5 GB  | ~8 hrs          | 1024    |
| 350m   | 350M   | ~8 GB  | ~2 days         | 2048    |
| 1b     | 1B     | ~10 GB | ~1 week         | 2048    |
| 1.5b   | 1.5B   | ~12 GB | ~3 weeks        | 2048    |

For 1B+: enable `--use_8bit_adam` (75% less optimizer VRAM) and gradient checkpointing (automatic).

---

## Inference

```bash
# Single prompt
python generate.py --checkpoint checkpoints/best_30m.pt --prompt "Once upon a time"

# Multiple completions
python generate.py --checkpoint checkpoints/best_70m.pt --prompt "The future of AI" --n 3

# Interactive chat
python generate.py --checkpoint checkpoints/best_1b.pt --interactive
```

Interactive commands:

```
:temp 0.7      change sampling temperature
:topk 40       change top-k filter
:tokens 500    change max new tokens
:quit          exit
```

### Sampling parameters

| Flag               | Default | Effect                                      |
| ------------------ | ------- | ------------------------------------------- |
| `--temperature`    | 0.8     | < 1 = sharper, > 1 = more random            |
| `--top_k`          | 50      | Keep only top-K logits                      |
| `--top_p`          | 0.9     | Nucleus: keep smallest set summing to ≥ 0.9 |
| `--max_new_tokens` | 200     | Tokens to generate                          |

---

## Supported Datasets

| Name           | Flag          | Source                 | Size      | Best for          |
| -------------- | ------------- | ---------------------- | --------- | ----------------- |
| Shakespeare    | `shakespeare` | Auto-download          | ~1 MB     | Smoke tests       |
| FineWeb        | `fineweb`     | HuggingFaceFW/fineweb  | Streaming | General LM        |
| Dolma          | `dolma`       | allenai/dolma          | Streaming | Diverse LM        |
| Custom HF      | `custom_hf`   | Any HF repo            | Streaming | Any text domain   |
| Custom file    | `binary`      | Your .txt              | Any       | Private data      |
| GSM8K          | `gsm8k`       | openai/gsm8k           | 8.5K      | Math reasoning    |
| MetaMathQA     | `metamath`    | meta-math/MetaMathQA   | 395K      | Math reasoning    |
| OpenHermes 2.5 | `openhermes`  | teknium/OpenHermes-2.5 | 1M+       | General reasoning |

---

## GGUF Export (llama.cpp / Ollama)

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt safetensors

# Export and quantize
python convert_gguf.py \
  --checkpoint checkpoints/best_1b.pt \
  --output models/my-llm-q4.gguf \
  --quantize q4_k_m

# Run with Ollama
echo 'FROM ./my-llm-q4.gguf' > Modelfile
ollama create my-llm -f Modelfile
ollama run my-llm
```

| Quantization | Size ratio | Quality                      |
| ------------ | ---------- | ---------------------------- |
| `f16`        | 2× model   | Lossless                     |
| `q8_0`       | 1× model   | Near-lossless                |
| `q4_k_m`     | 0.5× model | Recommended (best trade-off) |
| `q4_0`       | 0.5× model | Slightly lower quality       |

---

## MoE Diagnostics

Monitor expert load balance during or after training:

```python
from model import MiniLLM, ModelConfig
from data import get_dataloaders
from moe_utils import get_expert_load, print_expert_load_table, get_router_confidence
import torch

device = torch.device("cuda")
ckpt   = torch.load("checkpoints/best_70m.pt", map_location=device, weights_only=False)
config = ModelConfig(**ckpt["model_config"])
model  = MiniLLM(config).to(device)
model.load_state_dict(ckpt["model_state"])

_, val_loader = get_dataloaders("shakespeare", config.context_len, batch_size=4)

load = get_expert_load(model, val_loader, device, n_batches=100)
print_expert_load_table(load)

confidence = get_router_confidence(model, val_loader, device)
print(f"Router confidence: {confidence:.3f}")
```

Healthy ranges:

- Expert usage: 15–35% per expert (for 8 experts, top-2)
- Router confidence: > 0.6 = good specialization; < 0.4 = still early training
- If any expert > 40%: increase `aux_loss_coeff` from 0.01 → 0.05

---

## Training Metrics Reference

| Metric                       | Healthy range                 | Warning                              |
| ---------------------------- | ----------------------------- | ------------------------------------ |
| Initial loss                 | ~10.8                         | Stuck at 10.8 → model not on GPU     |
| Final loss (Shakespeare 70M) | 3.0–3.5                       | Not decreasing → check data pipeline |
| Final loss (FineWeb 1B)      | 2.5–2.8                       | —                                    |
| Perplexity                   | < 30 = learning real language | —                                    |
| tok/s (1B on RTX 4070)       | 500–900                       | < 300 → data loading bottleneck      |
| grad_norm                    | 0.5–2.0                       | Spikes > 10 → instability            |
| aux_loss                     | 0.001–0.05                    | Large → reduce aux_loss_coeff        |

---

## File Structure

```
LLM-creation-kit/
├── kit.py           ← TUI wizard — start here
├── train.py         ← Training loop
├── generate.py      ← Text generation / interactive chat
├── model.py         ← Model architecture (RoPE, GQA, MoE, SwiGLU)
├── data.py          ← Dataset pipeline (Shakespeare, HF streaming, binary)
├── reasoning.py     ← Chain-of-thought dataset formatting + streaming
├── config.py        ← YAML config save/load
├── moe_utils.py     ← MoE expert load diagnostics
├── convert_gguf.py  ← Export to GGUF for llama.cpp / Ollama
├── requirements.txt
├── checkpoints/     ← (generated) .pt training checkpoints
├── data/            ← (generated) cached tokenized datasets
└── logs/            ← (generated) JSONL training logs
```

---

## Architecture Deep-Dive

### Why MoE (Mixture of Experts)?

A dense transformer uses 100% of FFN parameters per token. MoE uses only top-K of N experts per token, so:

- Same computation per token as K/N of the FFN
- Total capacity of N experts
- A 1.5B MoE model uses ~400M active params per token (top-2 of 8 experts)

### Key components

**RoPE** — Rotary Position Embeddings encode position by rotating Q and K vectors. Better length generalization than learned absolute positions.

**RMSNorm** — Root Mean Square normalization. No mean subtraction = faster, equally stable. Used pre-attention and pre-FFN (pre-norm architecture like LLaMA).

**GQA** — Grouped Query Attention. Fewer K/V heads than Q heads; each group of Q heads shares one K/V head. Reduces KV cache at inference with negligible quality loss.

**SwiGLU** — `output = down(silu(gate(x)) * up(x))`. Three linear layers per expert. Consistently outperforms ReLU/GELU at all scales.

**MoE Router** — A linear gate layer assigns each token to its top-K experts. An auxiliary load-balancing loss prevents expert collapse (all tokens routing to 1-2 experts).

### VRAM budget (1.5B, all optimizations)

| Component                      | VRAM        |
| ------------------------------ | ----------- |
| Model weights (bf16)           | 3.6 GB      |
| Activations (grad checkpoint)  | 2.5 GB      |
| Optimizer states (8-bit AdamW) | 1.8 GB      |
| KV cache + misc                | 1.5 GB      |
| **Total**                      | **~9.4 GB** |

Without optimizations: 24+ GB (out-of-memory on 12 GB VRAM).

---

## Configuration Reference

All `train.py` flags (also exposed in the wizard):

```
--preset          30m|70m|125m|350m|1b|1.5b
--dataset         shakespeare|binary|fineweb|dolma|custom_hf|gsm8k|metamath|openhermes
--model_type      standard|reasoning
--hf_dataset_name <HuggingFace repo id>   (for custom_hf)
--hf_text_col     <column name>           (for custom_hf, default: text)
--bin_path        <path to .bin file>     (for binary dataset)
--max_steps       int
--batch_size      int   (micro-batch per step)
--grad_accum      int   (effective_batch = batch_size × grad_accum)
--lr              float (peak learning rate)
--min_lr          float (cosine decay floor)
--warmup_steps    int
--weight_decay    float (default 0.1, 2D+ params only)
--grad_clip       float (gradient norm clip, 0 = off)
--context_len     int   (override preset default)
--eval_interval   int
--save_interval   int
--patience        int   (early stopping; 0 = disabled)
--use_8bit_adam        (bitsandbytes 8-bit AdamW)
--compile              (torch.compile ~20% speedup)
--resume          <checkpoint path>
--checkpoint_dir  checkpoints/
--log_dir         logs/
--sample_prompt   "Once upon a time"
```

---

## License

MIT — do whatever you want with it, I made this project just for fun.

---

## Acknowledgements

Architecture inspired by [LLaMA 2](https://arxiv.org/abs/2307.09288), [Mixtral](https://arxiv.org/abs/2401.04088), and Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Tokenizer from [tiktoken](https://github.com/openai/tiktoken). Reasoning format inspired by [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
