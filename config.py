"""
config.py — Save, load, and convert training configurations.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def save_config(config: Dict[str, Any], path: str) -> None:
    if not _YAML_OK:
        raise ImportError("pyyaml is required to save configs: pip install pyyaml")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(path: str) -> Dict[str, Any]:
    if not _YAML_OK:
        raise ImportError("pyyaml is required to load configs: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def build_train_command(config: Dict[str, Any]) -> List[str]:
    """
    Convert a wizard config dict into a list of CLI arguments for train.py.
    Returns a list suitable for subprocess (no shell=True needed).
    """
    cmd = [sys.executable, "train.py"]

    # Simple string / int / float flags
    _str_flags = [
        "preset", "dataset", "bin_path", "hf_dataset_name", "hf_text_col",
        "model_type", "checkpoint_dir", "log_dir", "sample_prompt", "resume",
    ]
    _int_flags = [
        "max_steps", "batch_size", "grad_accum", "warmup_steps", "context_len",
        "eval_interval", "save_interval", "log_interval", "num_workers", "seed", "patience",
    ]
    _float_flags = ["lr", "min_lr", "weight_decay", "grad_clip", "min_delta"]
    _bool_flags  = ["use_8bit_adam", "compile"]

    for key in _str_flags:
        val = config.get(key)
        if val is not None:
            cmd += [f"--{key}", str(val)]

    for key in _int_flags:
        val = config.get(key)
        if val is not None:
            cmd += [f"--{key}", str(int(val))]

    for key in _float_flags:
        val = config.get(key)
        if val is not None:
            cmd += [f"--{key}", str(val)]

    for key in _bool_flags:
        if config.get(key):
            cmd.append(f"--{key}")

    return cmd
