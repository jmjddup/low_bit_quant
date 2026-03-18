#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Keep inference behavior aligned with offline_basic.py.
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
MODEL_PATH = "/root/fshare/models/Qwen/Qwen3-0.6B"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 16

SCRIPT_DIR = Path(__file__).resolve().parent
DUMP_ROOT = SCRIPT_DIR / "dump"


def _parse_layer_info(module_name: str) -> str:
    match = re.search(r"\.(\d+)\.mlp\.(up_proj|down_proj)$", module_name)
    if not match:
        safe_name = module_name.replace(".", "_")
        return f"unknown_{safe_name}"
    layer_idx, proj_type = match.groups()
    return f"layer_{int(layer_idx):03d}_{proj_type}"


def _register_dump_hooks(
    model: torch.nn.Module, run_dir: Path
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    call_counts: Dict[str, int] = defaultdict(int)
    records: List[dict] = []

    def make_hook(module_name: str):
        layer_tag = _parse_layer_info(module_name)

        def hook(module: torch.nn.Module, inputs):  # noqa: ANN001
            if not inputs:
                return
            first_input = inputs[0]
            if not isinstance(first_input, torch.Tensor):
                return

            call_idx = call_counts[module_name]
            call_counts[module_name] += 1

            tensor_cpu = first_input.detach().to("cpu")
            file_name = f"{layer_tag}__call_{call_idx:04d}.pt"
            file_path = run_dir / file_name
            torch.save(tensor_cpu, file_path)

            records.append(
                {
                    "module": module_name,
                    "layer_tag": layer_tag,
                    "call_idx": call_idx,
                    "shape": list(tensor_cpu.shape),
                    "dtype": str(tensor_cpu.dtype),
                    "path": str(file_path),
                }
            )

        return hook

    for name, module in model.named_modules():
        if name.endswith(".mlp.up_proj") or name.endswith(".mlp.down_proj"):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    # Save records lazily at script end.
    def save_records() -> None:
        index_path = run_dir / "dump_index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    # Attach helper for caller.
    setattr(model, "_dump_records_writer", save_records)
    return handles


def main() -> None:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DUMP_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Dump directory: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    handles = _register_dump_hooks(model, run_dir)

    inputs = tokenizer(
        PROMPTS,
        return_tensors="pt",
        padding=True,
    ).to(device)
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    print("\nGenerated Outputs:\n" + "-" * 60)
    for i, prompt in enumerate(PROMPTS):
        completion_ids = output_ids[i][int(prompt_lengths[i]) :]
        generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

    for h in handles:
        h.remove()

    writer = getattr(model, "_dump_records_writer", None)
    if callable(writer):
        writer()

    print(f"Dump finished. Check index file: {run_dir / 'dump_index.json'}")


if __name__ == "__main__":
    main()
