#!/usr/bin/env python3
"""
离线版 lm_eval 精度评估脚本。

从本地 save_to_disk 格式的数据集加载，无需联网即可完成精度测试。
数据集目录默认: /root/workspace/low_bit_quant/eval/datasets/

依赖:
  pip install lm-eval vllm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = "/root/fshare/models/Qwen/Qwen3-0.6B"
DEFAULT_DATASET_DIR = SCRIPT_DIR / "datasets"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_offline"
DEFAULT_TASKS = "hellaswag,gsm8k,truthfulqa_mc2"

# (hub_dataset_path, subset_or_None) -> local folder name under DATASET_DIR
# Derived from download_datasets.py and lm_eval task YAML configs.
DATASET_MAPPING = {
    ("Rowan/hellaswag", None):                       "hellaswag",
    ("cais/mmlu", "all"):                            "mmlu",
    ("openai/gsm8k", "main"):                        "gsm8k",
    ("truthfulqa/truthful_qa", "multiple_choice"):   "truthfulqa_mc2",
}


def _install_offline_patch(dataset_dir: Path) -> None:
    """
    Monkey-patch datasets.load_dataset BEFORE lm_eval is imported,
    so that lm_eval's `from datasets import load_dataset` picks up
    our patched version that redirects known Hub paths to local
    save_to_disk directories.
    """
    import datasets as _ds
    from datasets import load_from_disk

    _orig = _ds.load_dataset

    def _offline_load(path, name=None, *args, **kwargs):
        key = (path, name)
        local_name = DATASET_MAPPING.get(key)
        if local_name is not None:
            local_path = dataset_dir / local_name
            if local_path.is_dir():
                print(f"[offline] {path} ({name}) -> {local_path}")
                data = load_from_disk(str(local_path))
                split = kwargs.get("split")
                if split is not None and isinstance(data, _ds.DatasetDict):
                    if split in data:
                        return data[split]
                return data
        return _orig(path, name, *args, **kwargs)

    _ds.load_dataset = _offline_load
    if hasattr(_ds, "load") and hasattr(_ds.load, "load_dataset"):
        _ds.load.load_dataset = _offline_load


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run lm_eval offline using locally saved datasets",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help=f"Path to model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--tasks", default=DEFAULT_TASKS,
        help=f"Comma-separated task names (default: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR,
        help=f"Local dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output_path", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        print(f"Error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    # Patch BEFORE importing lm_eval so its load_dataset references are replaced.
    _install_offline_patch(dataset_dir)

    import lm_eval  # noqa: E402 — must be after patch
    import lm_eval.evaluator  # noqa: E402

    trust = args.trust_remote_code and not args.no_trust_remote_code
    model_args = f"pretrained={args.model}"
    if trust:
        model_args += ",trust_remote_code=True"

    task_list = [t.strip() for t in args.tasks.split(",")]

    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Backend:     vllm")
    print(f"Tasks:       {task_list}")
    print(f"Dataset dir: {dataset_dir}")
    print("=" * 70)

    results = lm_eval.evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    # ---- Save results ----
    output_dir = args.output_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_file = output_dir / f"results_offline_{ts}.json"

    save_data = {
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
        "n-samples": results.get("n-samples", {}),
        "config": {
            "model": "vllm",
            "model_args": model_args,
            "tasks": task_list,
            "dataset_dir": str(dataset_dir),
            "offline": True,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    # ---- Print summary ----
    print(f"\nResults saved to: {out_file}")
    print("\n" + "=" * 80)
    print(f"{'Task':<30} {'acc':>10} {'acc_norm':>10}")
    print("-" * 80)
    for task, metrics in results.get("results", {}).items():
        acc = metrics.get("acc,none", "N/A")
        acc_norm = metrics.get("acc_norm,none", "N/A")
        if isinstance(acc, float):
            acc = f"{acc:.4f}"
        if isinstance(acc_norm, float):
            acc_norm = f"{acc_norm:.4f}"
        print(f"  {task:<28} {acc:>10} {acc_norm:>10}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
