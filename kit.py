#!/usr/bin/env python3
"""
LLM Creation Kit — Interactive TUI wizard.

Walk through building your own LLM from scratch:
  1. Choose model type (standard or reasoning/CoT)
  2. Pick model scale (30M → 1.5B) or define a custom architecture
  3. Select training dataset (built-in, HuggingFace, or your own file)
  4. Tune hyperparameters with smart defaults per preset
  5. Toggle advanced options (8-bit AdamW, compile, GGUF export …)
  6. Review, save config, and launch training

Run:
  python kit.py
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Dependency check ──────────────────────────────────────────────────────────
_MISSING = []
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.padding import Padding
    from rich.columns import Columns
    from rich import box
except ImportError:
    _MISSING.append("rich")

try:
    import questionary
    from questionary import Style as QStyle
except ImportError:
    _MISSING.append("questionary")

if _MISSING:
    print(f"Missing dependencies: {', '.join(_MISSING)}")
    print("Install them with:  pip install rich questionary")
    sys.exit(1)

console = Console()

# ── Questionary theme ─────────────────────────────────────────────────────────
Q_STYLE = QStyle([
    ("qmark",       "fg:#a78bfa bold"),
    ("question",    "bold"),
    ("answer",      "fg:#7c3aed bold"),
    ("pointer",     "fg:#a78bfa bold"),
    ("highlighted", "fg:#a78bfa bold"),
    ("selected",    "fg:#6d28d9"),
    ("separator",   "fg:#4b5563"),
    ("instruction", "fg:#9ca3af italic"),
    ("text",        ""),
    ("disabled",    "fg:#6b7280 italic"),
])

# ── ASCII banner ──────────────────────────────────────────────────────────────
BANNER = """\
 ██╗     ██╗     ███╗   ███╗    ██╗  ██╗██╗████████╗
 ██║     ██║     ████╗ ████║    ██║ ██╔╝██║╚══██╔══╝
 ██║     ██║     ██╔████╔██║    █████╔╝ ██║   ██║
 ██║     ██║     ██║╚██╔╝██║    ██╔═██╗ ██║   ██║
 ███████╗███████╗██║ ╚═╝ ██║    ██║  ██╗██║   ██║
 ╚══════╝╚══════╝╚═╝     ╚═╝    ╚═╝  ╚═╝╚═╝   ╚═╝
              C R E A T I O N   K I T  v1.0"""

# ── Preset definitions ────────────────────────────────────────────────────────
PRESETS: Dict[str, Dict] = {
    "30m": {
        "label": "30M   Smoke test — ~10 min on RTX 4070",
        "arch":  "6 layers · 384 hidden · 4 experts · 512 ctx",
        "vram":  "~2 GB",
        "desc":  "Perfect for a first run to verify your setup works end-to-end.",
        "defaults": dict(
            max_steps=500, batch_size=16, grad_accum=2,
            lr=3e-4, min_lr=3e-5, warmup_steps=50,
            eval_interval=100, save_interval=250, patience=3,
            weight_decay=0.1, grad_clip=1.0,
        ),
    },
    "70m": {
        "label": "70M   First real model — ~1 hour",
        "arch":  "8 layers · 512 hidden · 8 experts · 1024 ctx",
        "vram":  "~3 GB",
        "desc":  "Produces coherent text on Shakespeare. Good starting point.",
        "defaults": dict(
            max_steps=5000, batch_size=12, grad_accum=4,
            lr=3e-4, min_lr=3e-5, warmup_steps=200,
            eval_interval=500, save_interval=1000, patience=3,
            weight_decay=0.1, grad_clip=1.0,
        ),
    },
    "125m": {
        "label": "125M  Quality jump — ~6–8 hours",
        "arch":  "12 layers · 768 hidden · 8 experts · 1024 ctx",
        "vram":  "~5 GB",
        "desc":  "Noticeable coherence improvement. Recommend FineWeb for this size.",
        "defaults": dict(
            max_steps=50_000, batch_size=8, grad_accum=8,
            lr=3e-4, min_lr=3e-5, warmup_steps=500,
            eval_interval=2000, save_interval=5000, patience=5,
            weight_decay=0.1, grad_clip=1.0,
        ),
    },
    "350m": {
        "label": "350M  Serious model — ~1–2 days",
        "arch":  "24 layers · 1024 hidden · 8 experts · 2048 ctx",
        "vram":  "~8 GB",
        "desc":  "Strong language model. Enable 8-bit AdamW to fit in 12 GB VRAM.",
        "defaults": dict(
            max_steps=200_000, batch_size=6, grad_accum=12,
            lr=2e-4, min_lr=2e-5, warmup_steps=1000,
            eval_interval=5000, save_interval=10_000, patience=5,
            weight_decay=0.1, grad_clip=1.0,
        ),
    },
    "1b": {
        "label": "1B    Production quality — ~1 week",
        "arch":  "32 layers · 2048 hidden · 8 experts · 2048 ctx",
        "vram":  "~10 GB (needs 8-bit AdamW)",
        "desc":  "Requires 8-bit AdamW + gradient checkpointing. FineWeb recommended.",
        "defaults": dict(
            max_steps=500_000, batch_size=4, grad_accum=16,
            lr=2e-4, min_lr=2e-5, warmup_steps=2000,
            eval_interval=5000, save_interval=5000, patience=5,
            weight_decay=0.1, grad_clip=1.0,
        ),
    },
    "1.5b": {
        "label": "1.5B  Flagship — ~2–3 weeks",
        "arch":  "32 layers · 2048 hidden · 8 experts · 2048 ctx",
        "vram":  "~12 GB (tight, all optimizations required)",
        "desc":  "Maximum scale for 12 GB VRAM. 8-bit AdamW is mandatory.",
        "defaults": dict(
            max_steps=750_000, batch_size=2, grad_accum=24,
            lr=1.5e-4, min_lr=1.5e-5, warmup_steps=3000,
            eval_interval=5000, save_interval=5000, patience=5,
            weight_decay=0.1, grad_clip=1.0,
        ),
    },
    "custom": {
        "label": "Custom  Define your own architecture",
        "arch":  "You choose every hyperparameter",
        "vram":  "Depends on config",
        "desc":  "Full control over layers, hidden dim, experts, context length, etc.",
        "defaults": dict(
            max_steps=5000, batch_size=8, grad_accum=4,
            lr=3e-4, min_lr=3e-5, warmup_steps=200,
            eval_interval=500, save_interval=1000, patience=3,
            weight_decay=0.1, grad_clip=1.0,
        ),
    },
}

# ── Dataset definitions ───────────────────────────────────────────────────────
STANDARD_DATASETS: Dict[str, Dict] = {
    "shakespeare": {
        "label": "Shakespeare   Built-in ~1 MB (best for testing)",
        "desc":  "Auto-downloaded. GPT-2 tokenized. 90/10 train/val split. No internet needed after first run.",
        "requires_hf": False,
    },
    "fineweb": {
        "label": "FineWeb       HuggingFace — high-quality web text",
        "desc":  "HuggingFaceFW/fineweb — curated CommonCrawl. Streaming, never downloads fully.",
        "requires_hf": True,
    },
    "dolma": {
        "label": "Dolma         HuggingFace — diverse corpus",
        "desc":  "allenai/dolma — web + books + code + Wikipedia. Streaming.",
        "requires_hf": True,
    },
    "custom_hf": {
        "label": "Custom HuggingFace   Any HF dataset by repo ID",
        "desc":  "Stream any text dataset from HuggingFace. You provide the repo name and text column.",
        "requires_hf": True,
    },
    "binary": {
        "label": "Custom Text File   Your own .txt file",
        "desc":  "Point to any .txt file. It will be tokenized with GPT-2 BPE and cached as .bin.",
        "requires_hf": False,
    },
}

REASONING_DATASETS: Dict[str, Dict] = {
    "gsm8k": {
        "label": "GSM8K         Grade-school math (8.5K problems)",
        "desc":  "openai/gsm8k — step-by-step solutions. Great for arithmetic reasoning.",
    },
    "metamath": {
        "label": "MetaMathQA    Augmented math reasoning (395K)",
        "desc":  "meta-math/MetaMathQA — strong chain-of-thought mathematical reasoning.",
    },
    "openhermes": {
        "label": "OpenHermes 2.5   General instruction + reasoning (1M+)",
        "desc":  "teknium/OpenHermes-2.5 — diverse CoT: math, code, science, general knowledge.",
    },
    "fineweb": {
        "label": "FineWeb       General pre-training text",
        "desc":  "Use FineWeb for reasoning model pre-training before CoT fine-tuning.",
    },
    "shakespeare": {
        "label": "Shakespeare   Quick test",
        "desc":  "Tiny corpus for verifying the reasoning pipeline works.",
    },
    "binary": {
        "label": "Custom Text File   Your own CoT-formatted .txt",
        "desc":  "Provide a .txt with <think>...</think> formatted examples for custom reasoning training.",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step_header(n: int, total: int, title: str) -> None:
    console.print()
    console.rule(
        f"[bold bright_white]Step {n}/{total}[/bold bright_white]  "
        f"[bold #a78bfa]{title}[/bold #a78bfa]",
        style="#4b5563",
    )
    console.print()


def _info(text: str) -> None:
    console.print(Panel(
        text,
        border_style="#4b5563",
        padding=(0, 1),
    ))
    console.print()


def _ask(question, *args, **kwargs):
    """Wrap questionary calls to always use the kit style."""
    kwargs.setdefault("style", Q_STYLE)
    return question(*args, **kwargs).ask()


def _text_with_default(prompt: str, default: Any, cast=str) -> Any:
    raw = _ask(questionary.text, prompt, default=str(default))
    if raw is None:
        raise KeyboardInterrupt
    try:
        return cast(raw)
    except (ValueError, TypeError):
        console.print(f"  [yellow]Invalid input, using default: {default}[/yellow]")
        return default


# ── Welcome ───────────────────────────────────────────────────────────────────

def _welcome() -> None:
    console.print()
    console.print(
        Panel(
            Text(BANNER, style="bold #a78bfa", justify="center"),
            border_style="#7c3aed",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            "[bold white]Build your own LLM from scratch.[/bold white]\n\n"
            "This wizard walks you through every choice:\n"
            "  • Model architecture  (30M → 1.5B parameters, or custom)\n"
            "  • Training dataset    (Shakespeare, FineWeb, Dolma, or your own)\n"
            "  • Hyperparameters     (smart defaults, fully adjustable)\n"
            "  • Advanced options    (8-bit AdamW, torch.compile, GGUF export …)\n"
            "  • Reasoning mode      (chain-of-thought  <think>…</think>  training)\n\n"
            "[dim]Press [bold]Ctrl+C[/bold] at any time to exit.[/dim]",
            border_style="#4b5563",
            padding=(1, 2),
        )
    )
    console.print()
    _ask(questionary.press_any_key_to_continue, "  Press Enter to begin …")


# ── Step 1: Model type ────────────────────────────────────────────────────────

def _ask_model_type() -> str:
    _step_header(1, 8, "Model Type")
    _info(
        "[bold]Standard Language Model[/bold]\n"
        "Classic autoregressive text generation. Train on any text corpus.\n"
        "Recommended for most use cases.\n\n"
        "[bold]Reasoning Model  [dim](chain-of-thought)[/dim][/bold]\n"
        "Trains with [cyan]<think>…</think>[/cyan] structured output. The model learns\n"
        "to reason step-by-step before giving an answer — similar to DeepSeek-R1.\n"
        "Works best when pre-trained on general text first, then fine-tuned\n"
        "on a reasoning dataset (e.g. GSM8K, MetaMathQA)."
    )

    choice = _ask(questionary.select, "What kind of model do you want to build?", choices=[
        questionary.Choice("Standard Language Model",          value="standard"),
        questionary.Choice("Reasoning Model  (chain-of-thought)", value="reasoning"),
    ])
    if choice is None:
        raise KeyboardInterrupt
    return choice


# ── Step 2: Model scale ───────────────────────────────────────────────────────

def _ask_model_scale() -> str:
    _step_header(2, 8, "Model Scale")

    # Build a rich table to show all presets
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold #a78bfa", pad_edge=False)
    tbl.add_column("Size",           style="bold white",   no_wrap=True)
    tbl.add_column("Architecture",   style="#d1d5db",      no_wrap=True)
    tbl.add_column("VRAM",           style="cyan",         no_wrap=True)
    tbl.add_column("Est. time",      style="#f59e0b",      no_wrap=True)

    time_est = {
        "30m": "~10 min", "70m": "~1 hr", "125m": "~8 hrs",
        "350m": "~2 days", "1b": "~1 week", "1.5b": "~3 weeks", "custom": "varies",
    }
    for key, p in PRESETS.items():
        tbl.add_row(
            key.upper(),
            p["arch"],
            p["vram"],
            time_est[key],
        )
    console.print(tbl)
    console.print()

    choices = [
        questionary.Choice(p["label"], value=key)
        for key, p in PRESETS.items()
    ]
    preset = _ask(questionary.select, "Choose a model scale:", choices=choices)
    if preset is None:
        raise KeyboardInterrupt

    if preset != "custom":
        console.print()
        console.print(f"  [dim]{PRESETS[preset]['desc']}[/dim]")

    return preset


# ── Step 2b: Custom architecture ──────────────────────────────────────────────

def _ask_custom_arch() -> Dict[str, Any]:
    _step_header(2, 8, "Custom Architecture")
    _info(
        "Define your own model architecture.\n"
        "These map directly to [cyan]ModelConfig[/cyan] fields in [bold]model.py[/bold]."
    )

    console.print("  [dim]Enter values or press Enter to accept the default.[/dim]\n")

    arch = {}
    arch["n_layers"]         = _text_with_default("Number of transformer layers",  12,   int)
    arch["n_heads"]          = _text_with_default("Number of attention heads (Q)", 12,   int)
    arch["n_kv_heads"]       = _text_with_default("Number of KV heads (GQA — must divide n_heads)", 12, int)
    arch["hidden_dim"]       = _text_with_default("Hidden / embedding dimension",  768,  int)
    arch["ffn_dim"]          = _text_with_default("Expert FFN inner dimension",    2048, int)
    arch["n_experts"]        = _text_with_default("Total number of MoE experts",   8,    int)
    arch["n_experts_active"] = _text_with_default("Active experts per token (top-K)", 2, int)
    arch["context_len"]      = _text_with_default("Context length (tokens)",       1024, int)
    arch["dropout"]          = _text_with_default("Dropout probability (0.0 for 1B+)", 0.1, float)

    return arch


# ── Step 3: Dataset ───────────────────────────────────────────────────────────

def _ask_dataset(model_type: str) -> Dict[str, Any]:
    _step_header(3, 8, "Dataset")

    dataset_map = REASONING_DATASETS if model_type == "reasoning" else STANDARD_DATASETS

    if model_type == "reasoning":
        _info(
            "[bold]Reasoning datasets[/bold] are formatted as:\n"
            "  [cyan]Question:[/cyan] <question>\n"
            "  [cyan]<think>[/cyan]\n  <step-by-step reasoning>\n"
            "  [cyan]</think>[/cyan]\n  [cyan]<answer>[/cyan] <final answer> [cyan]</answer>[/cyan]\n\n"
            "[dim]Tip: pre-train on FineWeb or Shakespeare first, "
            "then fine-tune on GSM8K / MetaMathQA.[/dim]"
        )

    choices = [
        questionary.Choice(v["label"], value=k)
        for k, v in dataset_map.items()
    ]
    dataset_name = _ask(questionary.select, "Choose a training dataset:", choices=choices)
    if dataset_name is None:
        raise KeyboardInterrupt

    console.print()
    console.print(f"  [dim]{dataset_map[dataset_name]['desc']}[/dim]")
    console.print()

    result: Dict[str, Any] = {"dataset": dataset_name}

    if dataset_name == "custom_hf":
        result["hf_dataset_name"] = _ask(
            questionary.text,
            "HuggingFace dataset repo ID (e.g. HuggingFaceFW/fineweb-edu):",
            validate=lambda v: bool(v.strip()) or "Repo ID cannot be empty",
        )
        result["hf_text_col"] = _ask(
            questionary.text,
            'Text column name in the dataset:',
            default="text",
        )

    elif dataset_name == "binary":
        path = _ask(
            questionary.path,
            "Path to your .txt file (will be tokenized and cached):",
            validate=lambda p: os.path.isfile(p) or "File not found",
        )
        # Tokenize inline if it's a .txt
        if path and path.endswith(".txt"):
            bin_out = Path(path).with_suffix(".bin")
            console.print(f"\n  Tokenizing [cyan]{path}[/cyan] → [cyan]{bin_out}[/cyan] …")
            try:
                from data import prepare_text_file
                prepare_text_file(path, str(bin_out))
                result["bin_path"] = str(bin_out)
            except Exception as e:
                console.print(f"  [red]Tokenization failed: {e}[/red]")
                result["bin_path"] = path
        else:
            result["bin_path"] = path

    return result


# ── Step 4: Hyperparameters ───────────────────────────────────────────────────

def _ask_hyperparams(preset: str) -> Dict[str, Any]:
    _step_header(4, 8, "Hyperparameters")

    defaults = PRESETS[preset]["defaults"]

    _info(
        "Smart defaults are set for the chosen preset.\n"
        "Press [bold]Enter[/bold] to accept a default or type a new value.\n\n"
        "[dim]Effective batch size = batch_size × grad_accum[/dim]"
    )

    hp: Dict[str, Any] = {}

    console.print("  [bold #a78bfa]Training steps[/bold #a78bfa]")
    hp["max_steps"]    = _text_with_default("  Max training steps", defaults["max_steps"], int)
    hp["warmup_steps"] = _text_with_default("  Warmup steps (linear LR ramp)", defaults["warmup_steps"], int)

    console.print()
    console.print("  [bold #a78bfa]Batch & memory[/bold #a78bfa]")
    hp["batch_size"] = _text_with_default("  Micro-batch size (per step)", defaults["batch_size"], int)
    hp["grad_accum"] = _text_with_default("  Gradient accumulation steps", defaults["grad_accum"], int)
    eff = hp["batch_size"] * hp["grad_accum"]
    console.print(f"  [dim]→ Effective batch size: {eff} sequences[/dim]")

    console.print()
    console.print("  [bold #a78bfa]Learning rate[/bold #a78bfa]")
    hp["lr"]     = _text_with_default("  Peak LR (cosine schedule)", defaults["lr"], float)
    hp["min_lr"] = _text_with_default("  Min LR (end of cosine decay)", defaults["min_lr"], float)

    console.print()
    console.print("  [bold #a78bfa]Regularisation[/bold #a78bfa]")
    hp["weight_decay"] = _text_with_default("  Weight decay (2D+ params only)", defaults["weight_decay"], float)
    hp["grad_clip"]    = _text_with_default("  Gradient norm clip (0 = off)", defaults["grad_clip"], float)

    console.print()
    console.print("  [bold #a78bfa]Evaluation & checkpointing[/bold #a78bfa]")
    hp["eval_interval"] = _text_with_default("  Eval every N steps", defaults["eval_interval"], int)
    hp["save_interval"] = _text_with_default("  Save checkpoint every N steps", defaults["save_interval"], int)

    console.print()
    console.print("  [bold #a78bfa]Sample prompt[/bold #a78bfa]")
    hp["sample_prompt"] = _ask(
        questionary.text,
        "  Prompt used for text samples during training:",
        default="Once upon a time",
    ) or "Once upon a time"

    return hp


# ── Step 5: Early stopping ────────────────────────────────────────────────────

def _ask_early_stopping(preset: str) -> Dict[str, Any]:
    defaults = PRESETS[preset]["defaults"]

    want = _ask(questionary.confirm,
                "Enable early stopping? (halt if val_loss stops improving)",
                default=defaults.get("patience", 0) > 0)
    if not want:
        return {"patience": 0}

    patience = _text_with_default(
        "Stop after how many eval intervals with no improvement?",
        defaults.get("patience", 3), int,
    )
    return {"patience": patience}


# ── Step 6: Advanced options ──────────────────────────────────────────────────

def _ask_advanced(preset: str) -> Dict[str, Any]:
    _step_header(6, 8, "Advanced Options")

    large_model = preset in ("350m", "1b", "1.5b")
    very_large  = preset in ("1b", "1.5b")

    _info(
        "[bold]Memory & speed optimizations[/bold]\n"
        "Gradient checkpointing is always enabled (saves ~60% activation memory).\n"
        "Select additional options below."
    )

    choices = [
        questionary.Choice(
            "8-bit AdamW  (~75% less optimizer VRAM, via bitsandbytes)"
            + (" ← recommended for this size" if very_large else ""),
            value="use_8bit_adam",
            checked=very_large,
        ),
        questionary.Choice(
            "torch.compile  (UNIX BASED ONLY ~20% speedup after first step, longer startup)",
            value="compile",
            checked=False,
        ),
        questionary.Choice(
            "Weights & Biases logging  (requires WANDB_API_KEY)",
            value="wandb",
            checked=False,
        ),
        questionary.Choice(
            "Export to GGUF after training  (requires llama.cpp cloned nearby)",
            value="export_gguf",
            checked=False,
        ),
    ]

    selected = _ask(questionary.checkbox, "Select options:", choices=choices) or []

    result: Dict[str, Any] = {
        "use_8bit_adam": "use_8bit_adam" in selected,
        "compile":       "compile"       in selected,
        "wandb":         "wandb"         in selected,
        "export_gguf":   "export_gguf"   in selected,
    }

    if result["export_gguf"]:
        console.print()
        quant_opts = [
            questionary.Choice("q4_k_m  — 4-bit (recommended, ~0.5× model size)", value="q4_k_m"),
            questionary.Choice("q8_0   — 8-bit (~1× model size, higher quality)",  value="q8_0"),
            questionary.Choice("f16    — Full precision (~2× model size)",          value="f16"),
        ]
        result["gguf_quantize"] = _ask(
            questionary.select, "GGUF quantization format:", choices=quant_opts
        )

    return result


# ── Step 7: Context length (optional override) ────────────────────────────────

def _ask_context_override(preset: str, custom_arch: Optional[Dict]) -> Optional[int]:
    _step_header(7, 8, "Context Length")

    if preset == "custom" and custom_arch:
        return None  # already set in custom arch

    preset_ctx = {
        "30m": 512, "70m": 1024, "125m": 1024,
        "350m": 2048, "1b": 2048, "1.5b": 2048,
    }
    default_ctx = preset_ctx.get(preset, 1024)

    console.print(f"  Default context length for [bold]{preset}[/bold]: [cyan]{default_ctx}[/cyan] tokens")
    console.print()

    want_override = _ask(questionary.confirm,
                         "Override the context length?", default=False)
    if not want_override:
        return None

    ctx = _text_with_default("Context length (tokens):", default_ctx, int)
    return ctx


# ── Step 8: Output / naming ───────────────────────────────────────────────────

def _ask_output(preset: str, model_type: str) -> Dict[str, Any]:
    _step_header(8, 8, "Output")

    suffix    = f"_{model_type}" if model_type == "reasoning" else ""
    ckpt_name = f"best_{preset}{suffix}"

    console.print("  Checkpoint directory and naming:")
    console.print()

    ckpt_dir = _ask(questionary.text, "  Checkpoint directory:", default="checkpoints") or "checkpoints"
    log_dir  = _ask(questionary.text, "  Log directory:",        default="logs")         or "logs"

    return {
        "checkpoint_dir": ckpt_dir,
        "log_dir":        log_dir,
        "ckpt_name":      ckpt_name,
    }


# ── Build final config ────────────────────────────────────────────────────────

def _build_config(
    model_type: str,
    preset: str,
    custom_arch: Optional[Dict],
    dataset_info: Dict,
    hyperparams: Dict,
    early_stop: Dict,
    advanced: Dict,
    context_override: Optional[int],
    output: Dict,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    cfg["model_type"] = model_type
    cfg["preset"]     = preset if preset != "custom" else "30m"  # base for custom

    # Dataset
    cfg.update(dataset_info)

    # Hyperparams
    cfg.update(hyperparams)

    # Early stopping
    cfg.update(early_stop)

    # Advanced
    cfg["use_8bit_adam"] = advanced.get("use_8bit_adam", False)
    cfg["compile"]       = advanced.get("compile", False)

    # Context override
    if context_override is not None:
        cfg["context_len"] = context_override
    elif preset == "custom" and custom_arch:
        cfg["context_len"] = custom_arch.get("context_len", 1024)

    # Output
    cfg["checkpoint_dir"] = output["checkpoint_dir"]
    cfg["log_dir"]        = output["log_dir"]

    # Custom arch info (stored for display, not directly passed to train.py)
    if preset == "custom" and custom_arch:
        cfg["_custom_arch"] = custom_arch

    # GGUF export info
    if advanced.get("export_gguf"):
        cfg["export_gguf"]   = True
        cfg["gguf_quantize"] = advanced.get("gguf_quantize", "q4_k_m")

    return cfg


# ── Review & confirm ──────────────────────────────────────────────────────────

def _review(cfg: Dict[str, Any], advanced: Dict) -> bool:
    console.print()
    console.rule("[bold #a78bfa]Configuration Review[/bold #a78bfa]", style="#4b5563")
    console.print()

    # Main table
    tbl = Table(
        box=box.ROUNDED, show_header=False,
        border_style="#4b5563", padding=(0, 1),
        min_width=60,
    )
    tbl.add_column("Key",   style="bold #a78bfa", no_wrap=True)
    tbl.add_column("Value", style="white")

    def _row(k: str, v: Any, style: str = "") -> None:
        tbl.add_row(k, f"[{style}]{v}[/{style}]" if style else str(v))

    _row("Model type",     cfg.get("model_type", "standard").upper(), "bold cyan")
    _row("Scale preset",   cfg.get("preset", "30m").upper())

    if "_custom_arch" in cfg:
        arch = cfg["_custom_arch"]
        _row("Architecture",
             f"{arch['n_layers']}L × {arch['hidden_dim']}d × {arch['n_experts']}exp")

    _row("Dataset",        cfg.get("dataset", "shakespeare"))
    if cfg.get("hf_dataset_name"):
        _row("  HF dataset", cfg["hf_dataset_name"])
    if cfg.get("hf_text_col") and cfg.get("hf_text_col") != "text":
        _row("  Text column", cfg["hf_text_col"])
    if cfg.get("bin_path"):
        _row("  Bin path", cfg["bin_path"])

    tbl.add_section()
    _row("Max steps",      f"{cfg.get('max_steps', 5000):,}")
    _row("Batch size",     cfg.get("batch_size", 8))
    _row("Grad accum",     cfg.get("grad_accum", 4))
    eff = cfg.get("batch_size", 8) * cfg.get("grad_accum", 4)
    _row("Effective batch", f"{eff} sequences", "dim")
    _row("Warmup steps",   f"{cfg.get('warmup_steps', 200):,}")
    _row("Peak LR",        cfg.get("lr", 3e-4))
    _row("Min LR",         cfg.get("min_lr", 3e-5))
    _row("Weight decay",   cfg.get("weight_decay", 0.1))
    _row("Grad clip",      cfg.get("grad_clip", 1.0))

    if cfg.get("context_len"):
        _row("Context len", cfg["context_len"])

    tbl.add_section()
    _row("Eval interval",  f"{cfg.get('eval_interval', 500):,} steps")
    _row("Save interval",  f"{cfg.get('save_interval', 1000):,} steps")
    patience = cfg.get("patience", 0)
    _row("Early stopping", f"patience={patience}" if patience > 0 else "disabled")

    tbl.add_section()
    _row("8-bit AdamW",    "✓ enabled" if cfg.get("use_8bit_adam") else "off", "green" if cfg.get("use_8bit_adam") else "dim")
    _row("torch.compile",  "✓ enabled" if cfg.get("compile")       else "off", "green" if cfg.get("compile")       else "dim")

    if cfg.get("export_gguf"):
        _row("GGUF export", f"✓ {cfg.get('gguf_quantize', 'q4_k_m')}", "green")

    tbl.add_section()
    _row("Checkpoint dir", cfg.get("checkpoint_dir", "checkpoints"))
    _row("Log dir",        cfg.get("log_dir", "logs"))
    _row("Sample prompt",  f'"{cfg.get("sample_prompt", "Once upon a time")}"', "dim")

    console.print(tbl)
    console.print()
    return True


# ── Build CLI command ─────────────────────────────────────────────────────────

def _build_command(cfg: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, "train.py"]

    def _add(flag: str, val: Any) -> None:
        cmd.extend([f"--{flag}", str(val)])

    _add("preset",   cfg["preset"])
    _add("dataset",  cfg["dataset"])
    _add("model_type", cfg.get("model_type", "standard"))

    if cfg.get("bin_path"):
        _add("bin_path", cfg["bin_path"])
    if cfg.get("hf_dataset_name"):
        _add("hf_dataset_name", cfg["hf_dataset_name"])
    if cfg.get("hf_text_col", "text") != "text":
        _add("hf_text_col", cfg["hf_text_col"])

    _add("max_steps",    int(cfg["max_steps"]))
    _add("batch_size",   int(cfg["batch_size"]))
    _add("grad_accum",   int(cfg["grad_accum"]))
    _add("lr",           cfg["lr"])
    _add("min_lr",       cfg["min_lr"])
    _add("warmup_steps", int(cfg["warmup_steps"]))
    _add("weight_decay", cfg["weight_decay"])
    _add("grad_clip",    cfg["grad_clip"])
    _add("eval_interval",  int(cfg["eval_interval"]))
    _add("save_interval",  int(cfg["save_interval"]))
    _add("checkpoint_dir", cfg["checkpoint_dir"])
    _add("log_dir",        cfg["log_dir"])
    _add("sample_prompt",  cfg.get("sample_prompt", "Once upon a time"))

    patience = cfg.get("patience", 0)
    if patience > 0:
        _add("patience", patience)

    if cfg.get("context_len"):
        _add("context_len", int(cfg["context_len"]))

    if cfg.get("use_8bit_adam"):
        cmd.append("--use_8bit_adam")
    if cfg.get("compile"):
        cmd.append("--compile")

    return cmd


# ── Save config ───────────────────────────────────────────────────────────────

def _maybe_save_config(cfg: Dict[str, Any]) -> None:
    want = _ask(questionary.confirm, "Save this configuration to a YAML file?", default=True)
    if not want:
        return
    filename = _ask(questionary.text, "Config filename:", default="my_llm_config.yaml") or "my_llm_config.yaml"
    try:
        from config import save_config
        # Remove internal keys before saving
        clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
        save_config(clean, filename)
        console.print(f"\n  [green]Config saved to [bold]{filename}[/bold][/green]")
        console.print(f"  [dim]Resume anytime: python kit.py --load {filename}[/dim]\n")
    except Exception as e:
        console.print(f"  [yellow]Could not save config: {e}[/yellow]\n")


# ── Launch training ───────────────────────────────────────────────────────────

def _launch(cmd: List[str], cfg: Dict[str, Any]) -> None:
    console.print()
    console.rule("[bold green]Launching Training[/bold green]", style="green")
    console.print()

    # Pretty-print the command
    cmd_str = " \\\n    ".join(cmd)
    console.print(Panel(
        f"[dim]{cmd_str}[/dim]",
        title="[bold]Command[/bold]",
        border_style="#4b5563",
        padding=(0, 1),
    ))
    console.print()

    confirmed = _ask(questionary.confirm, "Start training now?", default=True)
    if not confirmed:
        console.print("\n  [yellow]Training cancelled. Run the command above when ready.[/yellow]\n")
        return

    console.print(
        "\n  [bold green]Training started![/bold green]  "
        "Press [bold]Ctrl+C[/bold] to interrupt.\n"
    )
    console.rule(style="green")

    # Run as a subprocess, streaming output to the terminal
    try:
        proc = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    except KeyboardInterrupt:
        console.print("\n\n  [yellow]Training interrupted by user.[/yellow]")
        return

    if proc.returncode == 0:
        console.print()
        console.rule("[bold green]Training Complete[/bold green]", style="green")
        ckpt_dir  = cfg.get("checkpoint_dir", "checkpoints")
        preset    = cfg.get("preset", "30m")
        model_sfx = "_reasoning" if cfg.get("model_type") == "reasoning" else ""
        ckpt_path = f"{ckpt_dir}/best_{preset}{model_sfx}.pt"
        console.print(
            Panel(
                f"[bold white]Best checkpoint:[/bold white] [cyan]{ckpt_path}[/cyan]\n\n"
                f"[bold white]Generate text:[/bold white]\n"
                f"  [dim]python generate.py --checkpoint {ckpt_path} --interactive[/dim]\n\n"
                f"[bold white]Export to GGUF (llama.cpp / Ollama):[/bold white]\n"
                f"  [dim]python convert_gguf.py --checkpoint {ckpt_path} "
                f"--output models/my_model.gguf --quantize q4_k_m[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            f"\n  [red]Training exited with code {proc.returncode}.[/red]\n"
            "  Check the output above for errors."
        )


# ── Load from YAML ────────────────────────────────────────────────────────────

def _load_and_run(yaml_path: str) -> None:
    try:
        from config import load_config
        cfg = load_config(yaml_path)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        sys.exit(1)

    console.print(f"\n[green]Loaded config from [bold]{yaml_path}[/bold][/green]\n")
    _review(cfg, {})
    cmd = _build_command(cfg)
    _launch(cmd, cfg)


# ── Main wizard ───────────────────────────────────────────────────────────────

def main() -> None:
    # Handle --load flag for loading existing config
    if len(sys.argv) == 3 and sys.argv[1] == "--load":
        _load_and_run(sys.argv[2])
        return

    try:
        _welcome()

        # Step 1 — model type
        model_type = _ask_model_type()

        # Step 2 — scale / custom arch
        preset = _ask_model_scale()
        custom_arch: Optional[Dict] = None
        if preset == "custom":
            custom_arch = _ask_custom_arch()

        # Step 3 — dataset
        dataset_info = _ask_dataset(model_type)

        # Step 4 — hyperparameters
        hp = _ask_hyperparams(preset if preset != "custom" else "70m")

        # Step 5 — early stopping
        _step_header(5, 8, "Early Stopping")
        early_stop = _ask_early_stopping(preset if preset != "custom" else "70m")

        # Step 6 — advanced
        advanced = _ask_advanced(preset if preset != "custom" else "30m")

        # Step 7 — context length override
        ctx_override = _ask_context_override(preset, custom_arch)

        # Step 8 — output
        output = _ask_output(preset if preset != "custom" else "custom", model_type)

        # Build config
        cfg = _build_config(
            model_type, preset if preset != "custom" else "30m",
            custom_arch, dataset_info, hp, early_stop,
            advanced, ctx_override, output,
        )

        # Review
        _review(cfg, advanced)

        # Save config
        _maybe_save_config(cfg)

        # Build command and launch
        cmd = _build_command(cfg)
        _launch(cmd, cfg)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Wizard cancelled.[/yellow]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
