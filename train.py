"""
train.py — Training loop for MiniLLM-MoE.

All VRAM optimizations enabled by default on CUDA:
  - bf16 mixed precision autocast
  - Gradient checkpointing (saves ~60% activation memory)
  - Gradient accumulation (simulates large effective batch)
  - 8-bit AdamW via bitsandbytes (saves ~75% optimizer VRAM)
  - Flash Attention (automatic via F.scaled_dot_product_attention)
  - Gradient norm clipping

Usage:
  python train.py --preset 30m --dataset shakespeare --max_steps 500
  python train.py --preset 1b  --dataset fineweb    --max_steps 500000 --use_8bit_adam --compile
"""

import os
import sys
import json
import math
import time
import argparse
import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn

from model import MiniLLM, ModelConfig
from data import get_dataloaders

import tiktoken
_enc = tiktoken.get_encoding("gpt2")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MiniLLM-MoE")
    p.add_argument("--preset",         default="30m",
                   choices=["30m", "70m", "125m", "350m", "1b", "1.5b"])
    p.add_argument("--dataset",        default="shakespeare",
                   choices=["shakespeare", "binary", "fineweb", "dolma",
                            "custom_hf", "gsm8k", "metamath", "openhermes"])
    p.add_argument("--bin_path",       default=None,
                   help="Path to .bin file (required for --dataset binary)")
    p.add_argument("--hf_dataset_name", default=None,
                   help="HuggingFace dataset repo id (for --dataset custom_hf)")
    p.add_argument("--hf_text_col",    default="text",
                   help="Text column name in the HF dataset (for --dataset custom_hf)")
    p.add_argument("--model_type",     default="standard",
                   choices=["standard", "reasoning"],
                   help="Model type: 'standard' for LM, 'reasoning' for CoT training")
    p.add_argument("--max_steps",      type=int,   default=5000)
    p.add_argument("--batch_size",     type=int,   default=8,
                   help="Micro-batch size per step")
    p.add_argument("--grad_accum",     type=int,   default=4,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum")
    p.add_argument("--lr",             type=float, default=3e-4,
                   help="Peak learning rate")
    p.add_argument("--min_lr",         type=float, default=3e-5,
                   help="Minimum LR at end of cosine decay")
    p.add_argument("--warmup_steps",   type=int,   default=200)
    p.add_argument("--weight_decay",   type=float, default=0.1,
                   help="AdamW weight decay applied to 2D+ params only")
    p.add_argument("--grad_clip",      type=float, default=1.0,
                   help="Gradient norm clip threshold (0 = disabled)")
    p.add_argument("--use_8bit_adam",  action="store_true",
                   help="Use bitsandbytes 8-bit AdamW (~75 pct less optimizer VRAM)")
    p.add_argument("--compile",        action="store_true",
                   help="torch.compile() the model (~20-30 pct speedup after warmup)")
    p.add_argument("--context_len",    type=int,   default=None,
                   help="Override context length from preset")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--log_interval",   type=int,   default=50)
    p.add_argument("--eval_interval",  type=int,   default=500)
    p.add_argument("--save_interval",  type=int,   default=1000)
    p.add_argument("--resume",         default=None,
                   help="Path to .pt checkpoint to resume training from")
    p.add_argument("--sample_prompt",  default="Once upon a time")
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--patience",       type=int,   default=0,
                   help="Stop after this many eval intervals with no val_loss improvement. "
                        "0 = disabled. Recommended: 3 for Shakespeare, 0 for FineWeb.")
    p.add_argument("--min_delta",      type=float, default=1e-3,
                   help="Minimum val_loss improvement to count as progress for early stopping.")
    return p.parse_args()


# ─── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine LR schedule with linear warmup.

    Phase 1 (0 → warmup_steps): linear ramp 0 → max_lr.
    Phase 2 (warmup_steps → max_steps): cosine decay max_lr → min_lr.
    Phase 3 (> max_steps): constant min_lr.
    """
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, autocast_ctx, max_batches: int = 20):
    """
    Mean validation loss and perplexity over up to max_batches batches.

    Runs in eval mode and inference_mode for maximum speed. Uses the same
    autocast context as training for dtype consistency.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches  += 1
    model.train()

    if n_batches == 0:
        return float("inf"), float("inf")
    val_loss = total_loss / n_batches
    ppl      = math.exp(min(val_loss, 20.0))  # cap to avoid inf
    return val_loss, ppl


@torch.no_grad()
def sample_text(
    model, prompt: str, device, max_new_tokens: int = 150,
    temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
) -> str:
    """Generate a text sample for qualitative evaluation during training."""
    model.eval()
    tokens    = _enc.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    out_ids   = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
    )
    model.train()
    return _enc.decode(out_ids[0].tolist())


# ─── Optimizer ────────────────────────────────────────────────────────────────

def build_optimizer(model, args):
    """
    AdamW with selective weight decay.

    Weight decay is applied only to 2D+ params (linear weight matrices).
    Biases, norm gains (1D) should not be decayed — they'd be incorrectly
    regularised towards zero.

    Falls back to standard AdamW if bitsandbytes is not installed.
    """
    decay    = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]

    param_groups = [
        {"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(param_groups, lr=args.lr, betas=(0.9, 0.95))
            print("Optimizer: 8-bit AdamW (bitsandbytes)")
            return opt
        except ImportError:
            print("WARNING: bitsandbytes not installed — falling back to standard AdamW")

    opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("Optimizer: AdamW")
    return opt


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, val_loss, config, args, path: str):
    """
    Save a full training checkpoint.

    Unwraps torch.compile wrapper (model._orig_mod) if present so the
    state_dict is always from the raw uncompiled model.
    """
    raw = getattr(model, "_orig_mod", model)
    ckpt = {
        "model_state":     raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "val_loss":        val_loss,
        "model_config":    config.__dict__,
        "args":            vars(args),
        "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
    }
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


# ─── Training loop ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Device setup ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        prop   = torch.cuda.get_device_properties(0)
        vram   = prop.total_memory / 1024 ** 3
        print(f"Device: {prop.name}  ({vram:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU  (training will be very slow)")

    # ── Model ─────────────────────────────────────────────────────────────────
    config = ModelConfig.from_preset(args.preset)
    if args.context_len is not None:
        config.context_len = args.context_len

    model = MiniLLM(config).to(device)
    model.gradient_checkpointing_enable()
    n_params = model.count_parameters()
    print(f"Model:  {args.preset} preset  ({n_params / 1e6:.1f}M params)")
    print(f"Config: {config}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, args)

    # ── torch.compile (optional) ──────────────────────────────────────────────
    if args.compile:
        if device.type == "cuda":
            print("Compiling model (first step will be slow) ...")
            model = torch.compile(model)
        else:
            print("WARNING: --compile ignored on non-CUDA device")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        dataset_name=args.dataset,
        context_len=config.context_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bin_path=args.bin_path,
        hf_dataset_name=args.hf_dataset_name,
        hf_text_col=args.hf_text_col,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step    = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device)
        raw  = getattr(model, "_orig_mod", model)
        raw.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step    = ckpt["step"]
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed: step={start_step}  best_val_loss={best_val_loss:.4f}")

    # ── Autocast ──────────────────────────────────────────────────────────────
    # bf16 is the sweet spot on Ampere/Ada: same stability as fp32, half the memory.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir,        exist_ok=True)
    log_path = os.path.join(
        args.log_dir, f"train_{args.preset}_{int(time.time())}.jsonl"
    )
    log_fh = open(log_path, "w")

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nTraining {args.preset} ({args.model_type}) for {args.max_steps} steps")
    print(f"Effective batch: {eff_batch} sequences × {config.context_len} tokens")
    if args.model_type == "reasoning":
        print("Mode: Reasoning (chain-of-thought) — <think>...</think> format")
    print(f"Log: {log_path}\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    model.train()
    train_iter       = iter(train_loader)
    tokens_seen      = 0
    step             = start_step
    train_start      = time.perf_counter()   # never resets — total elapsed clock
    t0               = time.perf_counter()   # resets each log interval
    no_improve_count = 0   # consecutive eval intervals without val_loss improvement

    while step < args.max_steps:
        # Update LR
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ─────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tokens_seen += x.numel()

            with autocast_ctx:
                _, loss = model(x, y)

            # Divide loss so gradients are an average over micro-steps, not a sum
            (loss / args.grad_accum).backward()
            step_loss += loss.item() / args.grad_accum

        # Clip and step
        grad_norm = 0.0
        if args.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            ).item()
        optimizer.step()
        step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_interval == 0:
            t1           = time.perf_counter()
            dt           = max(t1 - t0, 1e-6)       # seconds for this interval
            elapsed_total = t1 - train_start          # seconds since training began
            tps = (
                args.batch_size * args.grad_accum
                * config.context_len
                * args.log_interval
            ) / dt
            t0  = t1
            ppl = math.exp(min(step_loss, 20.0))

            # Format total elapsed as HH:MM:SS
            elapsed_h = int(elapsed_total // 3600)
            elapsed_m = int((elapsed_total % 3600) // 60)
            elapsed_s = int(elapsed_total % 60)
            elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"

            # Format interval duration as e.g. "12.3s"
            interval_str = f"{dt:.1f}s"

            row = {
                "step":          step,
                "loss":          round(step_loss, 4),
                "ppl":           round(ppl, 2),
                "lr":            round(lr, 8),
                "tok_per_sec":   int(tps),
                "tokens_seen":   tokens_seen,
                "grad_norm":     round(grad_norm, 4),
                "elapsed":       elapsed_str,
                "interval_secs": round(dt, 1),
            }
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()

            print(
                f"[{elapsed_str}] (+{interval_str}) "
                f"step {step:6d} | loss {step_loss:.4f} | ppl {ppl:7.1f} | "
                f"lr {lr:.2e} | tok/s {tps:,.0f} | grad_norm {grad_norm:.2f}"
            )

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % args.eval_interval == 0:
            val_loss, val_ppl = evaluate(
                model, val_loader, device, autocast_ctx
            )
            print(f"\n{'─' * 65}")
            print(
                f"Eval  step {step:6d} | val_loss {val_loss:.4f} | "
                f"val_ppl {val_ppl:.1f}"
            )
            print(f"Sample:\n{sample_text(model, args.sample_prompt, device)}")
            print(f"{'─' * 65}\n")

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss    = val_loss
                no_improve_count = 0
                save_checkpoint(
                    model, optimizer, step, val_loss, config, args,
                    os.path.join(args.checkpoint_dir, f"best_{args.preset}.pt"),
                )
            else:
                no_improve_count += 1
                print(
                    f"  No improvement for {no_improve_count} eval interval(s). "
                    f"Best val_loss: {best_val_loss:.4f}"
                )
                if args.patience > 0 and no_improve_count >= args.patience:
                    print(
                        f"\nEarly stopping: val_loss has not improved by >{args.min_delta} "
                        f"for {args.patience} consecutive evals. "
                        f"Best checkpoint: checkpoints/best_{args.preset}.pt"
                    )
                    break

        # ── Periodic save ─────────────────────────────────────────────────────
        if step % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, step, step_loss, config, args,
                os.path.join(args.checkpoint_dir, f"step_{step:07d}_{args.preset}.pt"),
            )

    log_fh.close()
    print(f"\nTraining complete — best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
