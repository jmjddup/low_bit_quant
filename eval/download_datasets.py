# download_datasets.py
from datasets import load_dataset
import os

# ===================== 配置项 =====================
# 1. 要下载的数据集（对应 lm_eval 的 task 名称）
# 格式：{lm_eval任务名: (数据集名, 子集名)}
# 可通过 `lm_eval --list-tasks` 查看任务对应的数据集信息
TASKS_TO_DOWNLOAD = {
    "mmlu": ("cais/mmlu", "all"),                          # MMLU 全子集
    "gsm8k": ("openai/gsm8k", "main"),                    # GSM8K 主数据集
    "truthfulqa_mc2": ("truthfulqa/truthful_qa", "multiple_choice"),  # TruthfulQA
    "hellaswag": ("Rowan/hellaswag", None),                # HellaSwag（无子集）
}
# 2. 本地保存路径（需拷贝到离线机器）
SAVE_BASE_PATH = "/root/workspace/low_bit_quant/eval/datasets"  # 如 /data/offline-datasets

# ===================== 下载+保存数据集 =====================
def download_and_save_dataset(dataset_name, subset_name, save_path):
    """下载数据集并保存到本地磁盘"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据集（自动下载到 datasets 缓存）
    if subset_name:
        dataset = load_dataset(dataset_name, subset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    # 保存为本地磁盘格式（关键：离线加载的核心）
    dataset.save_to_disk(save_path)
    print(f"✅ 数据集 {dataset_name}({subset_name}) 已保存到：{save_path}")

# 批量下载所有指定数据集
for task_name, (ds_name, ds_subset) in TASKS_TO_DOWNLOAD.items():
    save_path = os.path.join(SAVE_BASE_PATH, task_name)
    download_and_save_dataset(ds_name, ds_subset, save_path)

# 可选：下载 lm_eval 任务配置（避免离线加载任务时联网）
# 拷贝 lm_eval 的任务配置文件（有网机器）
# cp -r $(python -c "import lm_eval; print(lm_eval.__path__[0])")/tasks /path/to/offline-lm_eval-tasks