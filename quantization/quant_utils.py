import torch
import math
import numpy as np
from safetensors.torch import save_file
from safetensors.torch import load_file

class QuantConfig:
    """
    统一的量化配置类，兼容 fake_quant_int、fake_quant_mixed_bycolumn、fake_quant_mixed_bymask 等相关参数。

    int 量化 (fake_quant_int):
        qbit: int 比特宽度
        gsize: int 分组大小
        sym: bool 对称/非对称

    mixed 量化 (fake_quant_mixed_bycolumn / fake_quant_mixed_bymask):
        low_bit: int 低比特宽度
        high_bit: int 高比特宽度
        l2h_ratio: float 低到高比特转换的比例
        mixed_mode: str 量化混合模式，支持 "by_column", "by_mask", None
        column_criterion: str 当 mixed_mode=="by_column" 时区分判别依据: "by_weight"、"by_activation"、"by_sqformat"...
    """
    def __init__(
        self,
        qbit: int = 8,
        gsize: int = -1,
        sym: bool = True,
        # mixed 相关参数
        low_bit: int = None,
        high_bit: int = None,
        l2h_ratio: float = None,
        mixed_mode: str = None,               # "by_column", "by_mask", None
        column_criterion: str = None          # "by_weight", "by_activation", "by_sqformat"... (仅 by_column 有效)
    ):
        # int 量化参数
        self.qbit = qbit
        self.gsize = gsize
        self.sym = sym

        # mixed 量化参数
        self.low_bit = low_bit
        self.high_bit = high_bit
        self.l2h_ratio = l2h_ratio
        self.mixed_mode = mixed_mode
        self.column_criterion = column_criterion

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

def fake_quant_mixed_bycolumn(
    weight: torch.Tensor,
    activation: torch.Tensor = None,
    low_bit: int = 2,
    high_bit: int = 4,
    l2h_ratio: float = 0.5,
    group_size: int = -1,
    sym: bool = True,
    select_method: str = "by_weight",
):
    """
    Args:
        weight:                     (..., k)    torch.bfloat16
        activation:                 (..., k)    torch.bfloat16 or None
        low_bit:                    int
        high_bit:                   int
        l2h_ratio:                  float
        group_size:                 int
        sym:                        bool
        select_method:              str, one of "by_weight", "by_activation", "by_sqformat"
    Returns:
        weight_q:                   (..., k)    torch.bfloat16
    """
    assert low_bit in [2, 3, 4, 8], "Only int8, int4, int3, int2 are supported !"
    assert high_bit in [2, 3, 4, 8], "Only int8, int4, int3, int2 are supported !"
    assert low_bit < high_bit, "low_bit must be less than high_bit"
    assert l2h_ratio > 0, "l2h_ratio must be greater than 0"
    k = weight.shape[-1]
    assert activation is None or activation.shape[-1] == k, "activation and weight must have the same number of columns"

    # Determine importance by method
    if select_method == "by_weight":
        Importance = weight.abs().amax(dim=0)
    elif select_method == "by_activation":
        assert activation is not None, "activation input required for select_method=by_activation"
        Importance = activation.abs().amax(dim=0)
    elif select_method == "by_sqformat":
        assert activation is not None, "activation input required for select_method=by_sqformat"
        act_mean = activation.abs().mean(dim=0)
        wgt_sum = weight.abs().sum(dim=0)
        Importance = act_mean * wgt_sum
    else:
        raise ValueError(f"Invalid select_method: {select_method}")

    # Sort columns by importance (ascending: less important go to lower bit rate)
    sorted_indices = torch.argsort(Importance, dim=-1)
    if sorted_indices.dim() < weight.dim():
        expanded_sorted_indices = sorted_indices
        for _ in range(weight.dim() - sorted_indices.dim()):
            expanded_sorted_indices = expanded_sorted_indices.unsqueeze(0)
        expand_shape = list(weight.shape)
        expand_shape[-1] = -1
        expanded_sorted_indices = expanded_sorted_indices.expand(*weight.shape[:-1], weight.shape[-1])
        sorted_indices = expanded_sorted_indices

    sorted_weight = weight.gather(dim=-1, index=sorted_indices)

    # Mixed quantization after sorting
    low_k = int(k * l2h_ratio)
    low_index = [slice(None)] * (sorted_weight.ndim - 1) + [slice(0, low_k)]
    high_index = [slice(None)] * (sorted_weight.ndim - 1) + [slice(low_k, None)]
    low_output = fake_quant_int(
        sorted_weight[tuple(low_index)], bit_width=low_bit, group_size=group_size, sym=sym
    )
    high_output = fake_quant_int(
        sorted_weight[tuple(high_index)], bit_width=high_bit, group_size=group_size, sym=sym
    )
    sorted_weight_q = torch.cat([low_output, high_output], dim=-1)

    # Restore original order
    reversed_indices = torch.argsort(sorted_indices, dim=-1)
    weight_q = sorted_weight_q.gather(dim=-1, index=reversed_indices)

    return weight_q

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

    # Integrate fake_quant_mix_int_bymask's logic here
    high_in = weight * hmask
    low_in = weight * (1 - hmask)
    high_out = fake_quant_int(high_in, bit_width=high_bit, group_size=-1, sym=sym)
    low_out = fake_quant_int(low_in, bit_width=low_bit, group_size=-1, sym=sym)
    weight_q = high_out + low_out
    return weight_q

def fake_quant_from_config(
    weight: torch.Tensor,
    activation: torch.Tensor = None,
    config: 'QuantConfig' = None,
):
    """
    统一入口：根据 QuantConfig 配置自动选择量化流程。

    Args:
        weight:         (..., k)    torch.bfloat16      (必需)
        activation:     (..., k)    torch.bfloat16 or None (可选，部分量化方式需传)
        config:         QuantConfig
    Returns:
        量化后的 tensor
    """
    assert config is not None, "Must provide config: QuantConfig"

    # 如果是int量化（不混合）
    if not config.mixed_mode or config.mixed_mode is None:
        return fake_quant_int(
            weight,
            bit_width=config.qbit,
            group_size=config.gsize,
            sym=config.sym,
        )
    # 按列混合量化
    elif config.mixed_mode == 'by_column':
        return fake_quant_mixed_bycolumn(
            weight=weight,
            activation=activation,
            low_bit=config.low_bit,
            high_bit=config.high_bit,
            l2h_ratio=config.l2h_ratio,
            group_size=config.gsize,   # 复用gsize
            sym=config.sym,
            select_method=config.column_criterion if config.column_criterion else "by_weight",
        )
    # 按mask混合量化
    elif config.mixed_mode == 'by_mask':
        return fake_quant_mixed_bymask(
            activation=activation if activation is not None else weight,
            weight=weight,
            low_bit=config.low_bit,
            high_bit=config.high_bit,
            l2h_ratio=config.l2h_ratio,
            sym=config.sym,
        )
    else:
        raise ValueError(f"Unknown mixed_mode: {config.mixed_mode}")

def gen_config(
    n: int = None,
    allow_mixed: bool = True,
    allow_column_modes=None,
    allow_mask_modes=None,
    qbits=[8, 4, 3, 2],
    gsizes=[-1, 512, 128, 64],
    syms=[True, False],
    low_bits=[2],
    high_bits=[4],
    l2h_ratios=[0.5],
    column_criteria=["by_weight", "by_activation", "by_sqformat"],
    seed=None,
):
    """
    批量生成 n 个合法的 QuantConfig，用于 fake_quant_from_config。

    如果 n 未指定，则穷举所有可能的组合，优先 int 量化、然后 by_column、by_mask。
    如果 n 指定，则从所有可能的组合中随机采样 n 个。

    Returns:
        List[QuantConfig]
    """
    import random
    from itertools import product

    if seed is not None:
        random.seed(seed)

    # --------------------------
    # 枚举所有可能组合
    configs_int = [
        QuantConfig(qbit=qbit, gsize=gsize, sym=sym)
        for qbit, gsize, sym in product(qbits, gsizes, syms)
    ]
    configs_column = [
        QuantConfig(
            mixed_mode='by_column',
            low_bit=lb,
            high_bit=hb,
            l2h_ratio=ratio,
            gsize=gsize,
            sym=sym,
            column_criterion=ccrit,
        )
        for lb, hb, ratio, gsize, sym, ccrit in product(
            low_bits, high_bits, l2h_ratios, gsizes, syms,
            allow_column_modes if allow_column_modes is not None else column_criteria
        )
    ]
    configs_mask = [
        QuantConfig(
            mixed_mode='by_mask',
            low_bit=lb,
            high_bit=hb,
            l2h_ratio=ratio,
            sym=sym,
        )
        for lb, hb, ratio, sym in product(
            low_bits, high_bits, l2h_ratios, syms
        )
    ]
    all_configs = configs_int + configs_column + configs_mask

    if n is not None:
        # 随机采样 n 个组合
        if n > len(all_configs):
            raise ValueError(f"Requested n={n} > total unique configs ({len(all_configs)})")
        configs = random.sample(all_configs, n)
        return configs
    else:
        # 穷举全部可能，int->by_column->by_mask 顺序
        return all_configs

def test_fake_quant_mixed_bymask():
    activation = torch.randn(1024, 4096).to(torch.bfloat16)
    weight = torch.randn(1024, 4096).to(torch.bfloat16)
    low_bit = 2
    high_bit = 4
    l2h_ratio = 0.5
    sym = True
    output = fake_quant_mixed_bymask(activation, weight, low_bit, high_bit, l2h_ratio, sym)
    mse = torch.mean((weight - output) ** 2).item()
    print(f"MSE: {mse}")
    return mse

def test_fake_quant_mixed_bycolumn():
    weight = torch.randn(1024, 4096).to(torch.bfloat16)
    activation = torch.randn(1024, 4096).to(torch.bfloat16)
    low_bit = 2
    high_bit = 4
    l2h_ratio = 0.5
    group_size = -1
    sym = True
    output = fake_quant_mixed_bycolumn(weight, activation, low_bit, high_bit, l2h_ratio, group_size, sym)
    mse = torch.mean((weight - output) ** 2).item()
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

def test_fake_quant_from_config():
    """
    遍历gen_config生成的所有量化配置，分别用fake_quant_from_config测试，并汇总/排序结果。
    """
    weight = torch.randn(1024, 4096).to(torch.bfloat16)
    activation = torch.randn(1024, 4096).to(torch.bfloat16)
    configs = gen_config()
    results = []
    for config in configs:
        # try both with and without activation if needed
        try:
            if config.mixed_mode in ('by_column', 'by_mask'):
                output = fake_quant_from_config(weight, activation, config)
            else:
                output = fake_quant_from_config(weight, None, config)
            mse = torch.mean((weight - output) ** 2).item()
            mae = torch.mean(torch.abs(weight - output)).item()
            max_abs_err = torch.max(torch.abs(weight - output)).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                weight.flatten(), output.flatten(), dim=0
            ).item()
            pearson_corr = torch.corrcoef(
                torch.stack([weight.flatten(), output.flatten()])
            )[0, 1].item()
            # 压缩config列描述
            if config.mixed_mode == "by_column":
                scheme = f"byCol(lb{config.low_bit}|hb{config.high_bit}|ratio{config.l2h_ratio}|g{config.gsize}-{'sym' if config.sym else 'asym'}-{config.column_criterion})"
            elif config.mixed_mode == "by_mask":
                scheme = f"byMask(lb{config.low_bit}|hb{config.high_bit}|ratio{config.l2h_ratio}-{'sym' if config.sym else 'asym'})"
            else:
                scheme = f"int{config.qbit}-g{config.gsize}-{'sym' if config.sym else 'asym'}"
            results.append({
                "scheme": scheme,
                "mse": mse,
                "mae": mae,
                "max_abs_err": max_abs_err,
                "cos_sim": cos_sim,
                "pearson_corr": pearson_corr,
                "config": config,
            })
        except Exception as e:
            print(f"Config {config.__dict__} failed: {str(e)}")

    # 按mse升序
    results = sorted(results, key=lambda x: x["mse"])
    best = results[0]

    print("=" * 140)
    print("Quantization Error Comparison (all QuantConfigs from gen_config)")
    print(
        f"{'Rank':<6}{'Scheme':<50}{'MSE':>14}{'MAE':>14}{'MaxAbsErr':>14}"
        f"{'CosSim':>14}{'Pearson':>14}"
    )
    print("-" * 140)

    for idx, item in enumerate(results, start=1):
        marker = "  <= best" if item["scheme"] == best["scheme"] else ""
        print(
            f"{idx:<6}{item['scheme']:<50}{item['mse']:>14.8f}{item['mae']:>14.8f}"
            f"{item['max_abs_err']:>14.8f}{item['cos_sim']:>14.8f}"
            f"{item['pearson_corr']:>14.8f}{marker}"
        )

    print("-" * 140)
    print(
        f"Best by MSE: {best['scheme']} | MSE={best['mse']:.8f}, "
        f"MAE={best['mae']:.8f}, MaxAbsErr={best['max_abs_err']:.8f}"
    )
    print("=" * 140)

def parse_model(model_path):
    model = load_file(model_path)
    # model = torch.load("/root/workspace/low_bit_quant/model_inference/dump/20260319_112603/layer_000_up_proj__call_0000.pt")

    for key, value in model.items():
        print(f"{key}: {value.shape}, {value.dtype}")
    return model

def main():
    parse_model("/root/fshare/models/Qwen/Qwen3-0.6B/model.safetensors")
    # test_fake_quant_int_multi()
    # test_fake_quant_mixed_bycolumn()
    # test_fake_quant_mixed_bymask()
    # test_fake_quant_from_config()

if __name__ == "__main__":
    main()