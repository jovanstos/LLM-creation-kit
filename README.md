# LLM Creation Kit w/MoE

A decoder-only transformer with Mixture of Experts (MoE), trained from scratch on consumer hardware.

**Target:** NVIDIA RTX 4070 (12 GB VRAM) · Intel i5-10600K · 16 GB RAM  
**Scale ladder:** 30M → 70M → 125M → 350M → 1B → 1.5B

---

## Architecture

```
Token Embedding (vocab_size × hidden_dim)
    │
    ▼  × n_layers
┌─────────────────────────────────┐
│  RMSNorm (pre-norm)             │
│  Grouped Query Attention + RoPE │
│  + Residual                     │
│                                 │
│  RMSNorm (pre-norm)             │
│  MoE FFN                        │
│    ┌──────────────────────┐     │
│    │ Router (linear gate) │     │
│    │  → top-K expert idx  │     │
│    └──────────────────────┘     │
│    Expert 0 (SwiGLU) ─┐         │
│    Expert 1 (SwiGLU)  ├─ sum    │
│    ...                 │        │
│    Expert N (SwiGLU) ─┘         │
│  + Residual                     │
└─────────────────────────────────┘
    │
    ▼
Final RMSNorm → LM Head (tied weights)
```

**Key design choices:**

| Component           | Choice                 | Why                                                |
| ------------------- | ---------------------- | -------------------------------------------------- |
| Positional encoding | RoPE                   | Better length generalisation than learned absolute |
| FFN activation      | SwiGLU                 | Consistently better than ReLU/GELU at all scales   |
| Normalisation       | RMSNorm pre-norm       | More stable training, faster (no mean subtraction) |
| Attention           | GQA (n_kv < n_q)       | Smaller KV cache at inference, same quality        |
| FFN type            | MoE (8 experts, top-2) | More params, same compute per token                |
| Tokenizer           | GPT-2 BPE (tiktoken)   | 50,257 vocab, fast, no training required           |
| Embedding           | Input = Output weights | Reduces params ~10%, standard practice             |

### Why MoE?

A standard transformer activates 100% of its FFN parameters per token.
MoE routes each token to only K of N experts, so the 1.5B model activates
~25% of FFN params per token — quality of a large model, compute of a small one.

---

## Scale Presets

| Preset | Total params | n_layers | hidden_dim | n_experts | Context |
| ------ | ------------ | -------- | ---------- | --------- | ------- |
| 30m    | ~33M         | 6        | 384        | 4         | 512     |
| 70m    | ~85M         | 8        | 512        | 8         | 1024    |
| 125m   | ~180M        | 12       | 768        | 8         | 1024    |
| 350m   | ~500M        | 24       | 1024       | 8         | 2048    |
| 1b     | ~1.3B        | 32       | 2048       | 8         | 2048    |
| 1.5b   | ~1.8B        | 32       | 2048       | 8         | 2048    |

---

## Installation

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Training — Scale Ladder

Always start with the 30M smoke test before investing time in larger runs.

> **Note:** `--patience 3` enables early stopping — training halts automatically if validation loss stops improving for 3 consecutive eval checks. Use it for Shakespeare runs. Leave it off for FineWeb (the dataset is too large to overfit).

### Stage 1 — 30M Smoke Test (~10 min)

Expected final loss: ~2.5–3.0 on Shakespeare (our hardware beats the original estimates).

```powershell
python train.py --preset 30m --dataset shakespeare --max_steps 500 --batch_size 16 --grad_accum 2 --lr 3e-4 --warmup_steps 50 --eval_interval 100 --log_interval 20 --patience 3
```

### Stage 2 — 70M First Real Model (~45 min–1.5 hrs)

```powershell
python train.py --preset 70m --dataset shakespeare --max_steps 5000 --batch_size 12 --grad_accum 4 --lr 3e-4 --warmup_steps 200 --patience 3
```

### Stage 3 — 125M FineWeb (~6–8 hours)

```powershell
python train.py --preset 125m --dataset fineweb --max_steps 50000 --batch_size 8 --grad_accum 8 --lr 3e-4 --warmup_steps 500 --use_8bit_adam --eval_interval 2000 --save_interval 5000 --patience 5
```

### Stage 4 — 350M Quality Milestone (~1–2 days)

```powershell
python train.py --preset 350m --dataset fineweb --max_steps 200000 --batch_size 6 --grad_accum 12 --lr 2e-4 --warmup_steps 1000 --use_8bit_adam --context_len 2048 --eval_interval 5000 --save_interval 10000 --patience 5
```

### Stage 5 — 1B Main Target (~1–2 weeks)

```powershell
python train.py --preset 1b --dataset fineweb --max_steps 500000 --batch_size 4 --grad_accum 16 --lr 2e-4 --min_lr 2e-5 --warmup_steps 2000 --use_8bit_adam --context_len 2048 --eval_interval 5000 --save_interval 5000 --patience 5
```

### Stage 6 — 1.5B Flagship (~2–3 weeks)

```powershell
python train.py --preset 1.5b --dataset fineweb --max_steps 750000 --batch_size 2 --grad_accum 24 --lr 1.5e-4 --min_lr 1.5e-5 --warmup_steps 3000 --use_8bit_adam --context_len 2048 --eval_interval 5000 --save_interval 5000 --patience 5
```

### Resuming a run

```powershell
python train.py --preset 1b --dataset fineweb --max_steps 500000 --batch_size 4 --grad_accum 16 --lr 2e-4 --min_lr 2e-5 --warmup_steps 2000 --use_8bit_adam --context_len 2048 --eval_interval 5000 --save_interval 5000 --resume checkpoints/best_1b.pt
```

---

## Inference

```powershell
python generate.py --checkpoint checkpoints/best_30m.pt --prompt "Once upon a time"
```

```powershell
python generate.py --checkpoint checkpoints/best_1b.pt --n 3 --prompt "The future of AI"
```

```powershell
python generate.py --checkpoint checkpoints/best_1b.pt --interactive
```

---

## MoE Diagnostics

After 10 000+ training steps, check expert load balance:

```python
from model import MiniLLM, ModelConfig
from data import get_dataloaders
from moe_utils import get_expert_load, print_expert_load_table, get_router_confidence
import torch

device = torch.device("cuda")
ckpt   = torch.load("checkpoints/best_350m.pt", map_location=device)
model  = MiniLLM(ModelConfig(**ckpt["model_config"])).to(device)
model.load_state_dict(ckpt["model_state"])

_, val_loader = get_dataloaders("shakespeare", context_len=1024, batch_size=4)

load = get_expert_load(model, val_loader, device, n_batches=100)
print_expert_load_table(load)
get_router_confidence(model, val_loader, device)
```

Healthy output: each expert used 15–35% of token slots.  
If any expert is < 5% or > 40%, increase `aux_loss_coeff` (default 0.01 → try 0.05).

---

## GGUF Export (Ollama / llama.cpp)

```powershell
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt safetensors
```

```powershell
python convert_gguf.py --checkpoint checkpoints/best_1.5b.pt --output models/minillm-1.5b-q4.gguf --quantize q4_k_m
```

```powershell
ollama create minillm-1.5b -f Modelfile
ollama run minillm-1.5b
```

**Quantized sizes:**

| Model | fp16    | int8    | int4    |
| ----- | ------- | ------- | ------- |
| 30M   | ~66 MB  | ~33 MB  | ~17 MB  |
| 70M   | ~170 MB | ~85 MB  | ~43 MB  |
| 125M  | ~360 MB | ~180 MB | ~90 MB  |
| 350M  | ~1.0 GB | ~500 MB | ~250 MB |
| 1B    | ~3.8 GB | ~1.9 GB | ~950 MB |
| 1.5B  | ~5.5 GB | ~2.8 GB | ~1.4 GB |

---

## VRAM Budget (1.5B, all optimizations enabled)

| Component                     | VRAM        |
| ----------------------------- | ----------- |
| Model weights (bf16)          | ~3.6 GB     |
| Activations (grad checkpoint) | ~2.5 GB     |
| Optimizer states (8-bit Adam) | ~1.8 GB     |
| KV cache + misc               | ~1.5 GB     |
| **Total**                     | **~9.4 GB** |

Without optimizations: ~24+ GB (OOM on 12 GB card).

---

## Training Metrics Reference

| Metric                       | Healthy range                 | Warning sign                     |
| ---------------------------- | ----------------------------- | -------------------------------- |
| Initial loss                 | ~10.8                         | Stuck at 10.8 → model not on GPU |
| Final loss (Shakespeare 70M) | 3.0–3.5                       | Not decreasing → check data      |
| Final loss (FineWeb 1B)      | 2.5–2.8                       | —                                |
| Perplexity                   | < 30 = learning real language | —                                |
| aux_loss                     | 0.001–0.05                    | Large → reduce aux_loss_coeff    |
| tok/s (1B on RTX 4070)       | 500–900                       | < 300 → data bottleneck          |
| grad_norm                    | 0.5–2.0                       | Spikes > 10 → instability        |

---

## File Structure

```
miniLLM/
├── model.py          # Full model: config, architecture, MoE, generation
├── train.py          # Training loop with all VRAM optimizations
├── data.py           # Data pipeline: Shakespeare, binary, HF streaming
├── generate.py       # Inference + interactive chat CLI
├── moe_utils.py      # Router analysis, expert load visualization
├── convert_gguf.py   # Export trained model to GGUF for llama.cpp/Ollama
├── requirements.txt
├── README.md
├── .gitignore
├── data/             # Tokenized datasets (gitignored)
├── checkpoints/      # Model checkpoints (gitignored)
└── logs/             # JSONL training logs (gitignored)
```

---

## Preparing Custom Data

```powershell
python -c "from data import prepare_text_file; prepare_text_file('my_corpus.txt', 'data/my_corpus.bin')"
```

```powershell
python train.py --preset 70m --dataset binary --bin_path data/my_corpus.bin
```
