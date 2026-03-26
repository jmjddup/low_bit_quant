# low_bit_quant

这是一个用于快速验证大模型低比特量化精度效果的仓库，当前围绕 Qwen3-0.6B 实现了从低到高的四层验证链路：

1. 单 tensor 量化精度
2. 单 op（GEMM）量化精度
3. 模型端到端输出精度
4. 模型数据集评测精度（lm-eval）

## 目录结构

```text
low_bit_quant/
├── model_inference/
│   ├── offline_basic.py
│   └── offline_basic_dump.py
├── quantization/
│   ├── quant_utils.py
│   └── quant_gemm.py
├── eval/
│   ├── download_datasets.py
│   ├── run_lm_eval.py
│   └── run_lm_eval_offline.py
└── README.md
```

## 依赖环境

- Python: 使用 `python3`
- 核心依赖（按需）：
  - `torch`
  - `transformers`
  - `vllm`
  - `datasets`
  - `lm-eval`
  - `safetensors`

建议在同一环境中安装上述依赖后再执行各脚本。

## 模型路径约定

当前脚本默认模型路径：

- `/root/fshare/models/Qwen/Qwen3-0.6B`

如需替换模型，可修改对应脚本中的 `MODEL_PATH` 或命令行参数（`eval` 脚本支持 `--model`）。

## 四层验证流程

### 1) 单 tensor 量化精度

量化算子与配置入口在：

- `quantization/quant_utils.py`

其中 `fake_quant_from_config` 统一支持：

- int 量化（`qbit/gsize/sym`）
- mixed by_column
- mixed by_mask

可直接在该文件中使用已有测试函数扩展单 tensor 对比实验。

### 2) 单 op（GEMM）量化精度

GEMM 对比脚本：

- `quantization/quant_gemm.py`

已支持能力：

- 量化入口统一为 `fake_quant_from_config`
- 可选 `smooth`（`enable_smooth`）
- 可选 Hadamard 变换（`enable_hadamard`）
- 同一量化方案下输出 `base/smooth/hadamard` 的 MSE 与差值对比

运行：

```bash
python3 quantization/quant_gemm.py
```

### 3) 模型端到端输出精度

基础推理：

```bash
python3 model_inference/offline_basic.py
```

带中间输入 dump 的推理（dump `mlp.up_proj/down_proj` 输入）：

```bash
python3 model_inference/offline_basic_dump.py
```

dump 结果默认输出到：

- `model_inference/dump/<timestamp>/`
- 索引文件：`dump_index.json`

### 4) 模型数据集评测精度（lm-eval）

在线评测（vLLM 后端）：

```bash
python3 eval/run_lm_eval.py
```

常用参数示例：

```bash
python3 eval/run_lm_eval.py --tasks hellaswag,arc_easy --batch_size auto --device cuda:0
```

结果默认输出到：

- `eval/results/`

## 离线评测流程

### 下载并保存离线数据集

```bash
python3 eval/download_datasets.py
```

数据集保存目录：

- `eval/datasets/`

### 使用离线数据集评测

```bash
python3 eval/run_lm_eval_offline.py
```

常用参数示例：

```bash
python3 eval/run_lm_eval_offline.py --tasks hellaswag,gsm8k --limit 100
```

离线评测结果默认输出到：

- `eval/results_offline/`

## 结果与数据管理

仓库已通过 `.gitignore` 屏蔽以下内容，避免误提交大文件与中间产物：

- `model_inference/dump/`
- `eval/datasets/`
- `eval/results/`
- `*.txt`

如果某些大文件已被 Git 跟踪，需额外执行 `git rm --cached` 取消跟踪。
