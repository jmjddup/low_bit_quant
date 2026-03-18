import torch
import math
import numpy as np
from safetensors.torch import save_file
from safetensors.torch import load_file

class QuantConfig:
    def __init__(self, qbit=8, gsize=-1, sym=True):
        self.qbit = qbit
        self.gsize = gsize
        self.sym = sym

def fake_quant_int(
    input: torch.Tensor,
    bit_width: int = 8,
    group_size: int = -1,
    sym: bool = True
):
    """
    Args:
        input:          (..., k)    torch.bfloat16
        bit_width:                  int
        group_size:                 int
        sym:                        bool
    Returns:
        output:         (..., k)    torch.bfloat16
    """
    assert bit_width in [2, 3, 4, 8], "Only int8, int4, int3, int2 are supported !"

    output = input.float()
    oshape = input.shape

    bit_range = 2 ** bit_width
    if sym:
        qmin, qmax = -bit_range // 2, bit_range // 2 - 1
    else:
        qmin, qmax = 0, bit_range - 1

    if group_size == -1:
        group_size = oshape[-1]
    if group_size <= 0:
        raise ValueError("group_size must be -1 or a positive integer")
    if oshape[-1] % group_size != 0:
        raise ValueError(
            f"last dim ({oshape[-1]}) must be divisible by group_size ({group_size})"
        )

    qshape = list(oshape[:-1]) + [oshape[-1] // group_size, group_size]
    output = output.view(qshape)
    i_max = output.amax(dim=-1, keepdim=True)
    i_min = output.amin(dim=-1, keepdim=True)
    if sym:
        scales = 2 * torch.maximum(-i_min, i_max) / (qmax - qmin)
        zeros = torch.zeros_like(i_min)
    else:
        scales = (i_max - i_min) / (qmax - qmin)
        zeros = -(i_min / scales).round()

    # quant
    qout = (output / scales + zeros).round().clamp(qmin, qmax)
    # dequant
    fake_out = qout.sub(zeros).mul(scales)

    return fake_out.view(oshape)

def fake_quant_mix_int_bycolumn(
    resorted_input: torch.Tensor,
    low_bit: int = 2,
    high_bit: int = 4,
    l2h_ratio: float = 0.5,
    group_size: int = -1,
    sym: bool = True
):
    """
    Args:
        resorted_input:          (..., k)    torch.bfloat16
        low_bit:                    int
        high_bit:                   int
        l2h_ratio:                  float
        group_size:                 int
        sym:                        bool
    Returns:
        resorted_output:         (..., k)    torch.bfloat16
    """
    assert low_bit in [2, 3, 4, 8], "Only int8, int4, int3, int2 are supported !"
    assert high_bit in [2, 3, 4, 8], "Only int8, int4, int3, int2 are supported !"
    assert low_bit < high_bit, "low_bit must be less than high_bit"
    assert l2h_ratio > 0, "l2h_ratio must be greater than 0"

    resorted_output = resorted_input.float()
    k = resorted_input.shape[-1]

    low_k = int(k * l2h_ratio)

    # 支持任意维度的输入（只在最后一维进行区分）
    low_index = [slice(None)] * (resorted_input.ndim - 1) + [slice(0, low_k)]
    high_index = [slice(None)] * (resorted_input.ndim - 1) + [slice(low_k, None)]
    low_output = fake_quant_int(
        resorted_input[tuple(low_index)], bit_width=low_bit, group_size=group_size, sym=sym
    )
    high_output = fake_quant_int(
        resorted_input[tuple(high_index)], bit_width=high_bit, group_size=group_size, sym=sym
    )
    resorted_output = torch.cat([low_output, high_output], dim=-1)

    return resorted_output

def fake_quant_mixed_bycolumn(
    activation: torch.Tensor,
    weight: torch.Tensor,
    low_bit: int = 2,
    high_bit: int = 4,
    l2h_ratio: float = 0.5,
    group_size: int = -1,
    sym: bool = True,
    select_method: str = "by_weight"
):
    """
    Args:
        activation:                 (..., k)    torch.bfloat16
        weight:                     (..., k)    torch.bfloat16
        low_bit:                    int
        high_bit:                   int
        l2h_ratio:                  float
        group_size:                 int
        sym:                        bool
        select_method:              str
    Returns:
        weight_q:                   (..., k)    torch.bfloat16
    """
    
    assert activation.shape[-1] == weight.shape[-1], "activation and weight must have the same number of columns"
    
    if select_method == "by_weight":
        max_ = weight.abs().amax(dim=0) # TODO: 支持任意维度
        Importance = max_
    elif select_method == "by_activation":
        max_ = activation.abs().amax(dim=0) # TODO: 支持任意维度
        Importance = max_
    elif select_method == "by_sqformat":
        act_mean = activation.abs().mean(dim=0)
        wgt_mean = weight.abs().mean(dim=0)
        Importance = act_mean + wgt_mean
    else:
        raise ValueError(f"Invalid select_method: {select_method}")
    
    # 按重要性排序
    # 保证 sorted_indices 形状与 weight 相同
    sorted_indices = torch.argsort(Importance, dim=-1)
    if sorted_indices.dim() < weight.dim():
        # 扩展为和 weight 形状一致
        expanded_sorted_indices = sorted_indices
        for _ in range(weight.dim() - sorted_indices.dim()):
            expanded_sorted_indices = expanded_sorted_indices.unsqueeze(0)
        expand_shape = list(weight.shape)
        expand_shape[-1] = -1  # The last dim matches
        expanded_sorted_indices = expanded_sorted_indices.expand(*weight.shape[:-1], weight.shape[-1])
        sorted_indices = expanded_sorted_indices

    sorted_weight = weight.gather(dim=-1, index=sorted_indices)
    sorted_weight_q = fake_quant_mix_int_bycolumn(sorted_weight, low_bit, high_bit, l2h_ratio, group_size, sym)
    reversed_indices = torch.argsort(sorted_indices, dim=-1)
    weight_q = sorted_weight_q.gather(dim=-1, index=reversed_indices)

    return weight_q

def fake_quant_mix_int_bymask(
    input: torch.Tensor,
    hmask: torch.Tensor,
    low_bit: int = 2,
    high_bit: int = 4,
    sym: bool = True
):
    """
    Args:
        input:          (..., k)    torch.bfloat16
        hmask:          (..., k)    torch.int8, 0: low_bit, 1: high_bit
        low_bit:                    int
        high_bit:                   int
        sym:                        bool
    Returns:
        output:         (..., k)    torch.bfloat16
    """

    high_in = input * hmask
    low_in = input * (1 - hmask)
    high_out = fake_quant_int(high_in, bit_width=high_bit, group_size=-1, sym=sym)
    low_out = fake_quant_int(low_in, bit_width=low_bit, group_size=-1, sym=sym)
    output = high_out + low_out
    return output

def fake_quant_mixed_bymask(
    activation: torch.Tensor,
    weight: torch.Tensor,
    low_bit: int = 2,
    high_bit: int = 4,
    l2h_ratio: float = 0.5,
    sym: bool = True
):
    """
    Args:
        activation:                 (..., k)    torch.bfloat16
        weight:                     (..., k)    torch.bfloat16
        low_bit:                    int
        high_bit:                   int
        l2h_ratio:                  float
        sym:                        bool
    Returns:
        weight_q:                   (..., k)    torch.bfloat16
    """

    assert activation.shape[-1] == weight.shape[-1], "activation and weight must have the same number of columns"

    select_indices = torch.topk(weight.abs(), k=int(weight.shape[-1] * l2h_ratio), dim=-1, largest=False)[1]
    hmask = torch.ones_like(weight, dtype=torch.int8)
    hmask.scatter_(-1, select_indices, 0)
    weight_q = fake_quant_mix_int_bymask(weight, hmask, low_bit, high_bit, sym)
    return weight_q

def test_fake_quant_mix_int_bycolumn():
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16)
    low_bit = 2
    high_bit = 4
    l2h_ratio = 0.5
    group_size = -1
    sym = True
    output = fake_quant_mixed_bycolumn(input_tensor, input_tensor, low_bit, high_bit, l2h_ratio, group_size, sym)
    mse = torch.mean((input_tensor - output) ** 2).item()
    print(f"MSE: {mse}")
    return mse

def test_fake_quant_int_single(qbit=8, gsize=-1, sym=True, input_tensor=None):
    if input_tensor is None:
        input_tensor = torch.randn(1024, 4096).to(torch.bfloat16)
    output = fake_quant_int(
        input_tensor, bit_width=qbit, group_size=gsize, sym=sym
    )

    mse = torch.mean((input_tensor - output) ** 2).item()
    mae = torch.mean(torch.abs(input_tensor - output)).item()
    max_abs_err = torch.max(torch.abs(input_tensor - output)).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        input_tensor.flatten(), output.flatten(), dim=0
    ).item()
    pearson_corr = torch.corrcoef(
        torch.stack([input_tensor.flatten(), output.flatten()])
    )[0, 1].item()

    return {
        "scheme": f"int{qbit}-g{gsize}-{'sym' if sym else 'asym'}",
        "qbit": qbit,
        "gsize": gsize,
        "sym": sym,
        "mse": mse,
        "mae": mae,
        "max_abs_err": max_abs_err,
        "cos_sim": cos_sim,
        "pearson_corr": pearson_corr,
    }

def test_fake_quant_int_multi():
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16)
    qbits = [8, 4, 3, 2]
    gsizes = [-1, 512, 128, 64, 16]
    syms = [True, False]
    configs = [(qbit, gsize, sym) for qbit in qbits for gsize in gsizes for sym in syms]
    results = [
        test_fake_quant_int_single(
            qbit=qbit, gsize=gsize, sym=sym, input_tensor=input_tensor
        )
        for qbit, gsize, sym in configs
    ]

    # 误差指标越小越好，因此按 MSE 升序展示。
    results = sorted(results, key=lambda x: x["mse"])
    best = results[0]

    print("=" * 124)
    print("Quantization Error Comparison (same input across all schemes)")
    print(
        f"{'Rank':<6}{'Scheme':<24}{'MSE':>14}{'MAE':>14}{'MaxAbsErr':>14}"
        f"{'CosSim':>14}{'Pearson':>14}"
    )
    print("-" * 124)

    for idx, item in enumerate(results, start=1):
        marker = "  <= best" if item["scheme"] == best["scheme"] else ""
        print(
            f"{idx:<6}{item['scheme']:<24}{item['mse']:>14.8f}{item['mae']:>14.8f}"
            f"{item['max_abs_err']:>14.8f}{item['cos_sim']:>14.8f}"
            f"{item['pearson_corr']:>14.8f}{marker}"
        )

    print("-" * 124)
    print(
        f"Best by MSE: {best['scheme']} | MSE={best['mse']:.8f}, "
        f"MAE={best['mae']:.8f}, MaxAbsErr={best['max_abs_err']:.8f}"
    )
    print("=" * 124)

def parse_model(model_path):
    model = load_file(model_path)

    for key, value in model.items():
        print(f"{key}: {value.shape}, {value.dtype}")
    return model

def main():
    # parse_model("/root/fshare/models/Qwen/Qwen3-0.6B/model.safetensors")
    # test_fake_quant_int_multi()
    test_fake_quant_mix_int_bycolumn()

if __name__ == "__main__":
    main()