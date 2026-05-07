"""
generate.py — Inference script for MiniLLM-MoE.

Loads a trained checkpoint and generates text.
Supports single-prompt and interactive chat modes.

Usage:
  python generate.py --checkpoint checkpoints/best_30m.pt --prompt "Hello"
  python generate.py --checkpoint checkpoints/best_1b.pt  --interactive
  python generate.py --checkpoint checkpoints/best_1b.pt  --n 3 --prompt "Once upon"

Interactive commands:
  :temp <float>    change sampling temperature
  :topk <int>      change top-k filter size
  :tokens <int>    change max new tokens
  :quit            exit
"""

import argparse
import sys

import torch
import tiktoken

from model import MiniLLM, ModelConfig

_enc = tiktoken.get_encoding("gpt2")


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> MiniLLM:
    """
    Reconstruct a MiniLLM from a checkpoint file.

    The ModelConfig is stored in the checkpoint as a plain dict, so no
    separate config file is needed — the checkpoint is self-contained.
    """
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["model_config"])
    model  = MiniLLM(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_completions(
    model: MiniLLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    n: int,
) -> list:
    """
    Generate n completions for prompt.

    Batches all n sequences in one forward call for efficiency.
    Returns a list of n decoded strings (each includes the prompt).
    """
    tokens    = _enc.encode(prompt)
    input_ids = torch.tensor([tokens] * n, dtype=torch.long, device=device)

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    return [_enc.decode(ids.tolist()) for ids in out_ids]


# ─── Interactive mode ─────────────────────────────────────────────────────────

def interactive_mode(
    model: MiniLLM,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    """
    Interactive chat loop.

    Reads prompts from stdin and prints completions until :quit.
    Allows runtime tuning of sampling parameters via colon-commands.
    """
    print("MiniLLM Interactive Mode")
    print("Commands: :temp <float>  :topk <int>  :tokens <int>  :quit")
    print("─" * 55)

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue

        if prompt.startswith(":"):
            parts = prompt.split()
            cmd   = parts[0]

            if cmd == ":quit":
                print("Goodbye.")
                break
            elif cmd == ":temp" and len(parts) == 2:
                temperature = float(parts[1])
                print(f"  temperature → {temperature}")
            elif cmd == ":topk" and len(parts) == 2:
                top_k = int(parts[1])
                print(f"  top_k → {top_k}")
            elif cmd == ":tokens" and len(parts) == 2:
                max_new_tokens = int(parts[1])
                print(f"  max_new_tokens → {max_new_tokens}")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        completions = generate_completions(
            model, prompt, device, max_new_tokens, temperature, top_k, top_p, n=1
        )
        print(f"\n{completions[0]}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MiniLLM text generation")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--prompt",         default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",         type=int,   default=50)
    p.add_argument("--top_p",         type=float, default=0.9)
    p.add_argument("--n",             type=int,   default=1,
                   help="Number of completions to generate")
    p.add_argument("--interactive",    action="store_true",
                   help="Enter interactive chat mode")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading: {args.checkpoint}")

    model  = load_model(args.checkpoint, device)
    n_params = model.count_parameters()
    print(f"Model loaded ({n_params / 1e6:.1f}M params)\n")

    if args.interactive:
        interactive_mode(
            model, device, args.max_new_tokens,
            args.temperature, args.top_k, args.top_p,
        )
    else:
        completions = generate_completions(
            model, args.prompt, device, args.max_new_tokens,
            args.temperature, args.top_k, args.top_p, args.n,
        )
        for i, text in enumerate(completions, 1):
            if args.n > 1:
                print(f"\n─── Completion {i}/{args.n} ───")
            print(text)


if __name__ == "__main__":
    main()
