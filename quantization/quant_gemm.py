import torch
import math
from safetensors.torch import save_file
from safetensors.torch import load_file

from quant_utils import QuantConfig, fake_quant_from_config, gen_config

def quant_gemm(A, qConfigA, B, qConfigB):
    """
    Args:
        A: (..., k)    torch.bfloat16
        B: (n, k)    torch.bfloat16
        qConfigA: QuantConfig
        qConfigB: QuantConfig
    Returns:
        C: (m, n)    torch.bfloat16
    """
    assert A.shape[-1] == B.shape[-1], "The number of columns in A must be equal to the number of columns in B"

    # For weight mixed quant modes that rely on activation statistics,
    # collapse activation to 2D so column importance can be computed on K.
    act_for_weight = A.reshape(-1, A.shape[-1]) if A.dim() > 2 else A

    A_q = fake_quant_from_config(weight=A, activation=None, config=qConfigA)
    B_q = fake_quant_from_config(weight=B, activation=act_for_weight, config=qConfigB)

    C = torch.matmul(A_q, B_q.T)

    return C


def _config_to_scheme(prefix: str, cfg: QuantConfig) -> str:
    if cfg.mixed_mode == "by_column":
        criterion = cfg.column_criterion if cfg.column_criterion else "by_weight"
        return (
            f"{prefix}(byCol-lb{cfg.low_bit}-hb{cfg.high_bit}-"
            f"ratio{cfg.l2h_ratio}-g{cfg.gsize}-"
            f"{'sym' if cfg.sym else 'asym'}-{criterion})"
        )
    if cfg.mixed_mode == "by_mask":
        return (
            f"{prefix}(byMask-lb{cfg.low_bit}-hb{cfg.high_bit}-"
            f"ratio{cfg.l2h_ratio}-{'sym' if cfg.sym else 'asym'})"
        )
    return f"{prefix}(i{cfg.qbit}-g{cfg.gsize}-{'sym' if cfg.sym else 'asym'})"

def gemm_compare(A, qConfigA, B, qConfigB):
    """
    Args:
        A: (..., k)    torch.bfloat16
        B: (n, k)    torch.bfloat16
        qConfigA: QuantConfig
        qConfigB: QuantConfig
    Returns:
        dict: {
            "scheme": scheme,
            "mse": mse,
            "mae": mae,
            "max_abs_err": max_abs_err,
            "cos_sim": cos_sim,
            "pearson_corr": pearson_corr,
        }
    """
    out = torch.matmul(A, B.T)
    qout = quant_gemm(A, qConfigA, B, qConfigB)

    mse = torch.mean((out - qout) ** 2).item()
    mae = torch.mean(torch.abs(out - qout)).item()
    max_abs_err = torch.max(torch.abs(out - qout)).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out.flatten(), qout.flatten(), dim=0
    ).item()
    pearson_corr = torch.corrcoef(
        torch.stack([out.flatten(), qout.flatten()])
    )[0, 1].item()
    scheme = f"{_config_to_scheme('w', qConfigB)}{_config_to_scheme('a', qConfigA)}"

    return {
        "scheme": scheme,
        "mse": mse,
        "mae": mae,
        "max_abs_err": max_abs_err,
        "cos_sim": cos_sim,
        "pearson_corr": pearson_corr,
    }

def test_gemm_compare():
    # ACTIVATION_PATH = "/root/jmj/model_infer/proj_inputs/qwen3_0p6b_up_down_proj_inputs.pt"
    ACTIVATION_PATH = "/root/workspace/low_bit_quant/model_inference/dump/20260319_112603/layer_000_up_proj__call_0000.pt"
    WEIGHT_PATH = "/root/fshare/models/Qwen/Qwen3-0.6B/model.safetensors"
    activation = torch.load(ACTIVATION_PATH)
    weight = load_file(WEIGHT_PATH)

    # act = activation["model.layers.0.mlp.up_proj"].to(torch.bfloat16) # shape: (b, s, d)
    act = torch.load(ACTIVATION_PATH).to(torch.bfloat16)
    wgt = weight["model.layers.0.mlp.up_proj.weight"] # shape: (m, k)

    qConfigA = QuantConfig(qbit=8, gsize=-1, sym=True)
    configs = gen_config()
    results = [
        gemm_compare(act, qConfigA, wgt, qConfigB)
        for qConfigB in configs
    ]

    results = sorted(results, key=lambda x: x["mse"])
    best = results[0]
    print("=" * 124)
    print("Quantization Error Comparison (same input across all schemes)")

    # Define max width for scheme column dynamically for perfect alignment
    scheme_width = max(32, max(len(item["scheme"]) for item in results) + 2)
    header_fmt = f"{{:<6}}{{:<{scheme_width}}}{{:>16}}{{:>16}}{{:>16}}{{:>16}}{{:>16}}"
    row_fmt    = f"{{:<6}}{{:<{scheme_width}}}{{:>16.8f}}{{:>16.8f}}{{:>16.8f}}{{:>16.8f}}{{:>16.8f}}{{}}"

    print(header_fmt.format("Rank", "Scheme", "MSE", "MAE", "MaxAbsErr", "CosSim", "Pearson"))
    print("-" * 124)
    for idx, item in enumerate(results, start=1):
        marker = "  <= best" if item["scheme"] == best["scheme"] else ""
        print(row_fmt.format(
            idx,
            item['scheme'],
            item['mse'],
            item['mae'],
            item['max_abs_err'],
            item['cos_sim'],
            item['pearson_corr'],
            marker
        ))
    print("=" * 124)


def main():
    test_gemm_compare()

if __name__ == "__main__":
    main()
