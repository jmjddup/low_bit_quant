#!/usr/bin/env python3
"""
使用 lm_eval (lm-evaluation-harness) 对指定模型做精度验证。

依赖:
  pip install lm-eval
  (HuggingFace 后端还需: pip install accelerate)
模型路径默认: /root/fshare/models/Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = "/root/fshare/models/Qwen/Qwen3-0.6B"
DEFAULT_TASKS = "hellaswag,arc_easy,arc_challenge,winogrande"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lm_eval for model accuracy validation")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help=f"Comma-separated task names (default: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (task default if not set)",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="Batch size: int, 'auto', or 'auto:N' (default: auto)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit examples per task (int or float 0-1). For quick testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cuda:0, cpu, mps (default: cuda if available)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Allow trust_remote_code for HuggingFace model (default: True)",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable trust_remote_code",
    )
    args = parser.parse_args()

    args.output_path = args.output_path.resolve()
    args.output_path.mkdir(parents=True, exist_ok=True)

    # lm_eval run --model hf --model_args pretrained=MODEL_PATH,trust_remote_code=True ...
    model_args = f"pretrained={args.model}"
    if args.trust_remote_code and not args.no_trust_remote_code:
        model_args += ",trust_remote_code=True"

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        args.tasks,
        "--output_path",
        str(args.output_path),
        "--batch_size",
        args.batch_size,
    ]
    if args.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(args.num_fewshot)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.device is not None:
        cmd.extend(["--device", args.device])

    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=Path(__file__).resolve().parent).returncode


if __name__ == "__main__":
    sys.exit(main())
