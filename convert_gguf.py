"""
convert_gguf.py — Export a MiniLLM checkpoint to GGUF format.

Enables int4/int8 quantised inference via llama.cpp and
one-command serving via Ollama. The 1.5B model compresses to
~1.4 GB at int4, and runs fully in RAM + VRAM via llama.cpp.

Prerequisites:
  pip install safetensors transformers
  git clone https://github.com/ggerganov/llama.cpp
  pip install -r llama.cpp/requirements.txt

Usage:
  python convert_gguf.py \\
    --checkpoint checkpoints/best_1.5b.pt \\
    --output     models/minillm-1.5b-q4.gguf \\
    --quantize   q4_k_m

  # Import into Ollama:
  echo 'FROM ./minillm-1.5b-q4.gguf' > Modelfile
  ollama create minillm-1.5b -f Modelfile
  ollama run minillm-1.5b
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Convert MiniLLM checkpoint to GGUF")
    p.add_argument("--checkpoint",    required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--output",        required=True,
                   help="Output .gguf file path")
    p.add_argument("--quantize",      default="q4_k_m",
                   choices=["f16", "q8_0", "q4_k_m", "q4_0", "q5_k_m"],
                   help="Quantisation format (q4_k_m recommended for quality/size balance)")
    p.add_argument("--llama_cpp_dir", default="llama.cpp",
                   help="Path to local llama.cpp checkout")
    return p.parse_args()


# ─── Weight remapping ─────────────────────────────────────────────────────────

_BLOCK_REMAP = {
    "norm1.weight":           "input_layernorm.weight",
    "norm2.weight":           "post_attention_layernorm.weight",
    "attn.q_proj.weight":     "self_attn.q_proj.weight",
    "attn.k_proj.weight":     "self_attn.k_proj.weight",
    "attn.v_proj.weight":     "self_attn.v_proj.weight",
    "attn.o_proj.weight":     "self_attn.o_proj.weight",
    "moe.router.gate.weight": "mlp.gate.weight",
}

_TOP_REMAP = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "norm_final.weight":   "model.norm.weight",
    "lm_head.weight":      "lm_head.weight",
}


def _remap_key(key: str):
    """
    Map MiniLLM weight names to HuggingFace Llama naming so that
    llama.cpp's convert_hf_to_gguf.py can parse them.

    MiniLLM                         → HF Llama
    embed_tokens.weight             → model.embed_tokens.weight
    blocks.N.norm1.weight           → model.layers.N.input_layernorm.weight
    blocks.N.attn.q_proj.weight     → model.layers.N.self_attn.q_proj.weight
    blocks.N.moe.experts.M.*        → model.layers.N.mlp.experts.M.*
    blocks.N.moe.router.gate.weight → model.layers.N.mlp.gate.weight
    norm_final.weight               → model.norm.weight
    lm_head.weight                  → lm_head.weight
    """
    if key in _TOP_REMAP:
        return _TOP_REMAP[key]

    if key.startswith("blocks."):
        parts     = key.split(".", 2)
        layer_idx = parts[1]
        rest      = parts[2]

        if rest in _BLOCK_REMAP:
            return f"model.layers.{layer_idx}.{_BLOCK_REMAP[rest]}"

        # Expert weights: moe.experts.M.xxx → mlp.experts.M.xxx
        if rest.startswith("moe.experts."):
            suffix = rest[len("moe."):]           # experts.M.xxx
            return f"model.layers.{layer_idx}.mlp.{suffix}"

    return None  # skip unknowns (e.g. freqs_cis buffer)


# ─── HF format export ─────────────────────────────────────────────────────────

def _build_hf_config(cfg: dict) -> dict:
    """Minimal HuggingFace Llama config.json for llama.cpp compatibility."""
    return {
        "architectures":           ["LlamaForCausalLM"],
        "model_type":              "llama",
        "hidden_size":             cfg["hidden_dim"],
        "intermediate_size":       cfg["ffn_dim"],
        "num_hidden_layers":       cfg["n_layers"],
        "num_attention_heads":     cfg["n_heads"],
        "num_key_value_heads":     cfg["n_kv_heads"],
        "vocab_size":              cfg["vocab_size"],
        "max_position_embeddings": cfg["context_len"],
        "rms_norm_eps":            cfg.get("norm_eps", 1e-5),
        "rope_theta":              10000.0,
        "hidden_act":              "silu",
        "torch_dtype":             "float16",
        "num_local_experts":       cfg.get("n_experts", 8),
        "num_experts_per_tok":     cfg.get("n_experts_active", 2),
    }


def save_as_hf_safetensors(
    model_state: dict, config_dict: dict, out_dir: str
) -> None:
    """
    Save model weights in HuggingFace safetensors format + config.json.

    llama.cpp's convert script expects exactly this layout:
      out_dir/model.safetensors
      out_dir/config.json
      out_dir/tokenizer_config.json
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        sys.exit("ERROR: safetensors not installed.  Run: pip install safetensors")

    remapped = {}
    skipped  = 0
    for key, tensor in model_state.items():
        new_key = _remap_key(key)
        if new_key is None:
            skipped += 1
            continue
        remapped[new_key] = tensor.float().contiguous()

    save_file(remapped, os.path.join(out_dir, "model.safetensors"))
    print(f"  Saved {len(remapped)} tensors  ({skipped} skipped/non-param buffers)")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(_build_hf_config(config_dict), f, indent=2)
    print("  Saved config.json")

    tokenizer_cfg = {
        "tokenizer_class": "GPT2Tokenizer",
        "model_max_length": config_dict["context_len"],
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_cfg, f, indent=2)
    print("  Saved tokenizer_config.json")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        print(f"ERROR: llama.cpp not found at '{args.llama_cpp_dir}'")
        print("\nInstall llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  pip install -r llama.cpp/requirements.txt")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt        = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state"]
    config_dict = ckpt["model_config"]
    print(f"  Config: {config_dict}")

    tmp_dir = tempfile.mkdtemp(prefix="minillm_hf_")
    print(f"  Temp dir: {tmp_dir}")

    try:
        save_as_hf_safetensors(model_state, config_dict, tmp_dir)

        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        cmd = [
            sys.executable, convert_script,
            tmp_dir,
            "--outfile", out_path,
            "--outtype", args.quantize,
        ]
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        size_gb = os.path.getsize(out_path) / 1024 ** 3
        print(f"\nGGUF written: {out_path}  ({size_gb:.2f} GB)")

        stem = Path(out_path).stem.lower().replace(".", "-").replace("_", "-")
        print(f"\nTo serve with Ollama:")
        print(f"  echo 'FROM {out_path}' > Modelfile")
        print(f"  ollama create {stem} -f Modelfile")
        print(f"  ollama run {stem}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
