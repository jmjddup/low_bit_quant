import torch
import math
from safetensors.torch import save_file
from safetensors.torch import load_file

from quant_utils import QuantConfig, fake_quant_from_config, gen_config


def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform on the last dimension.
    Output is orthonormalized by sqrt(n), i.e., H * H^T = I.
    """
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard transform requires power-of-two last dim, got {n}")

    y = x.contiguous().reshape(-1, n).float()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    y = y.view(-1, n) / math.sqrt(n)
    return y.reshape(x.shape).to(dtype=x.dtype)


def hadamard_act_wgt(A: torch.Tensor, B: torch.Tensor):
    """
    Apply the same orthonormal Hadamard transform on K dimension:
      A' = A H, B' = B H
    which keeps A @ B^T unchanged.
    """
    return _fwht_last_dim(A), _fwht_last_dim(B)


def smooth_act_wgt(A: torch.Tensor, B: torch.Tensor, alpha: float = 0.5, eps: float = 1e-6):
    """
    Smooth activation/weight on K dimension while keeping A @ B^T mathematically unchanged:
      A' = A / s, B' = B * s
    where s is computed per-column from activation/weight magnitudes.
    """
    act_2d = A.reshape(-1, A.shape[-1]) if A.dim() > 2 else A
    act_max = act_2d.abs().amax(dim=0).clamp_min(eps)
    wgt_max = B.abs().amax(dim=0).clamp_min(eps)
    scales = (act_max.pow(alpha) / wgt_max.pow(1.0 - alpha)).clamp_min(eps)

    view_shape = [1] * A.dim()
    view_shape[-1] = -1
    A_smooth = A / scales.view(*view_shape)
    B_smooth = B * scales.view(1, -1)
    return A_smooth, B_smooth


def quant_gemm(
    A,
    qConfigA,
    B,
    qConfigB,
    enable_smooth: bool = False,
    enable_hadamard: bool = False,
):
    """
    Args:
        A: (..., k)    torch.bfloat16
        B: (n, k)    torch.bfloat16
        qConfigA: QuantConfig
        qConfigB: QuantConfig
        enable_smooth: bool
        enable_hadamard: bool
    Returns:
        C: (m, n)    torch.bfloat16
    """
    assert A.shape[-1] == B.shape[-1], "The number of columns in A must be equal to the number of columns in B"

    if enable_hadamard:
        A, B = hadamard_act_wgt(A, B)

    if enable_smooth:
        A, B = smooth_act_wgt(A, B)

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

def gemm_compare(
    A,
    qConfigA,
    B,
    qConfigB,
    enable_smooth: bool = False,
    enable_hadamard: bool = False,
):
    """
    Args:
        A: (..., k)    torch.bfloat16
        B: (n, k)    torch.bfloat16
        qConfigA: QuantConfig
        qConfigB: QuantConfig
    Returns:
        dict: {
            "scheme": scheme,
            "enable_smooth": enable_smooth,
            "enable_hadamard": enable_hadamard,
            "mse": mse,
        }
    """
    out = torch.matmul(A, B.T)
    qout = quant_gemm(
        A,
        qConfigA,
        B,
        qConfigB,
        enable_smooth=enable_smooth,
        enable_hadamard=enable_hadamard,
    )

    mse = torch.mean((out - qout) ** 2).item()
    scheme = f"{_config_to_scheme('w', qConfigB)}{_config_to_scheme('a', qConfigA)}"

    return {
        "scheme": scheme,
        "enable_smooth": enable_smooth,
        "enable_hadamard": enable_hadamard,
        "mse": mse,
    }

def test_gemm_compare():
    # ACTIVATION_PATH = "/root/jmj/model_infer/proj_inputs/qwen3_0p6b_up_down_proj_inputs.pt"
    ACTIVATION_PATH = "/root/workspace/low_bit_quant/model_inference/dump/20260319_170151/layer_000_up_proj__call_0000.pt"
    WEIGHT_PATH = "/root/fshare/models/Qwen/Qwen3-0.6B/model.safetensors"
    activation = torch.load(ACTIVATION_PATH)
    weight = load_file(WEIGHT_PATH)

    # act = activation["model.layers.0.mlp.up_proj"].to(torch.bfloat16) # shape: (b, s, d)
    act = torch.load(ACTIVATION_PATH).to(torch.bfloat16) # shape: (b, s, d)
    wgt = weight["model.layers.0.mlp.up_proj.weight"] # shape: (m, k)

    qConfigA = QuantConfig(qbit=8, gsize=-1, sym=True)
    configs = gen_config()
    paired_results = []
    skipped = 0
    for qConfigB in configs:
        try:
            base = gemm_compare(act, qConfigA, wgt, qConfigB, enable_smooth=False)
            smooth = gemm_compare(act, qConfigA, wgt, qConfigB, enable_smooth=True)
            hadamard = gemm_compare(
                act,
                qConfigA,
                wgt,
                qConfigB,
                enable_smooth=False,
                enable_hadamard=True,
            )
            paired_results.append(
                {
                    "scheme": base["scheme"],
                    "mse_base": base["mse"],
                    "mse_smooth": smooth["mse"],
                    "mse_delta_smooth": smooth["mse"] - base["mse"],
                    "mse_hadamard": hadamard["mse"],
                    "mse_delta_hadamard": hadamard["mse"] - base["mse"],
                }
            )
        except Exception:
            skipped += 1

    if not paired_results:
        raise RuntimeError("No valid quantization config remained after filtering.")

    paired_results = sorted(paired_results, key=lambda x: x["mse_base"])
    best = min(paired_results, key=lambda x: x["mse_smooth"])

    print("=" * 140)
    print("Quantization Error Comparison (same scheme, smooth/hadamard vs baseline)")
    if skipped:
        print(f"Skipped {skipped} invalid configs.")

    scheme_width = max(32, max(len(item["scheme"]) for item in paired_results) + 2)
    header_fmt = f"{{:<6}}{{:<{scheme_width}}}{{:>14}}{{:>14}}{{:>16}}{{:>14}}{{:>18}}"
    row_fmt = f"{{:<6}}{{:<{scheme_width}}}{{:>14.8f}}{{:>14.8f}}{{:>16.8f}}{{:>14.8f}}{{:>18.8f}}{{}}"
    print(
        header_fmt.format(
            "Rank",
            "Scheme",
            "MSE(base)",
            "MSE(smooth)",
            "Delta(smooth)",
            "MSE(hadamard)",
            "Delta(hadamard)",
        )
    )
    print("-" * 140)
    for idx, item in enumerate(paired_results, start=1):
        marker = "  <= best MSE(on)" if item["scheme"] == best["scheme"] else ""
        print(
            row_fmt.format(
                idx,
                item["scheme"],
                item["mse_base"],
                item["mse_smooth"],
                item["mse_delta_smooth"],
                item["mse_hadamard"],
                item["mse_delta_hadamard"],
                marker,
            )
        )
    print("=" * 140)


def main():
    test_gemm_compare()

if __name__ == "__main__":
    main()
