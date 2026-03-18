import torch
import math
from safetensors.torch import save_file
from safetensors.torch import load_file

from quant_utils import QuantConfig, fake_quant_int

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

    A_q = fake_quant_int(A, qConfigA.qbit, qConfigA.gsize, qConfigA.sym)
    B_q = fake_quant_int(B, qConfigB.qbit, qConfigB.gsize, qConfigB.sym)

    C = torch.matmul(A_q, B_q.T)

    return C

def gemm_compare(A, qConfigA, B, qConfigB):
    """
    Args:
        A: (..., k)    torch.bfloat16
        B: (n, k)    torch.bfloat16
        qConfigA: QuantConfig
        qConfigB: QuantConfig
    Returns:
        dict: {
            "scheme": f"w(i{qConfigB.qbit}-g{qConfigB.gsize}-{'sym' if qConfigB.sym else 'asym'})a(i{qConfigA.qbit}-g{qConfigA.gsize}-{'sym' if qConfigA.sym else 'asym'})",
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

    return {
        "scheme": f"w(i{qConfigB.qbit}-g{qConfigB.gsize}-{'sym' if qConfigB.sym else 'asym'})a(i{qConfigA.qbit}-g{qConfigA.gsize}-{'sym' if qConfigA.sym else 'asym'})",
        "mse": mse,
        "mae": mae,
        "max_abs_err": max_abs_err,
        "cos_sim": cos_sim,
        "pearson_corr": pearson_corr,
    }

def test_gemm_compare():
    ACTIVATION_PATH = "/root/jmj/model_infer/proj_inputs/qwen3_0p6b_up_down_proj_inputs.pt"
    WEIGHT_PATH = "/root/fshare/models/Qwen/Qwen3-0.6B/model.safetensors"
    activation = torch.load(ACTIVATION_PATH)
    weight = load_file(WEIGHT_PATH)

    act = activation["model.layers.0.mlp.up_proj"].to(torch.bfloat16) # shape: (b, s, d)
    wgt = weight["model.layers.0.mlp.up_proj.weight"] # shape: (m, k)

    qConfigA = QuantConfig(qbit=8, gsize=-1, sym=True)
    qbits = [8, 4, 3, 2]
    gsizes = [-1, 512, 128, 64, 16]
    syms = [True, False]
    configs = [
        QuantConfig(qbit=qbit, gsize=gsize, sym=sym)
        for qbit in qbits for gsize in gsizes for sym in syms
    ]
    results = [
        gemm_compare(act, qConfigA, wgt, qConfigB)
        for qConfigB in configs
    ]

    results = sorted(results, key=lambda x: x["mse"])
    best = results[0]
    print("=" * 124)
    print("Quantization Error Comparison (same input across all schemes)")
    header_fmt = "{:<6}{:<32}{:>16}{:>16}{:>16}{:>16}{:>16}"
    row_fmt    = "{:<6}{:<32}{:>16.8f}{:>16.8f}{:>16.8f}{:>16.8f}{:>16.8f}{}"

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
