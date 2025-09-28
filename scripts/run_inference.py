# 文件路径: scripts/run_inference.py

import sys
import os
import pandas as pd

# ==================== 1. 配置区域 ====================
DATASET_NAME = 'tpcds'
TRAIN_MODE = 'estimated_train' # 指定加载哪个版本的模型
USE_ESTIMATES_MODE = True # 指定评估时用什么数据
# =======================================================

# --- 2. 项目根目录设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import structure
structure.USE_HASH_TABLE_SIZE_FEATURE = False # 确保这里是 False

# --- 3. 动态路径定义 ---
if DATASET_NAME == 'tpch':
    TEST_DATA_SOURCE = "tpch_output_22"
elif DATASET_NAME == 'tpcds':
    TEST_DATA_SOURCE = "tpcds_100g_new_test"
else:
    raise ValueError(f"未知的数据集名称: {DATASET_NAME}")

DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
TEST_DATA_DIR = os.path.join(DATA_DIR, TEST_DATA_SOURCE)

# 动态定义输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", DATASET_NAME)
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predictions")
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

# 模型路径是通用的，不与数据集绑定
# --- 修改模型加载路径 ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "output", "models", TRAIN_MODE)
no_dop_model_dir_path = os.path.join(MODEL_DIR, "operator_non_dop_aware")
dop_model_dir_path = os.path.join(MODEL_DIR, "operator_dop_aware")

# --- 4. 导入模块 ---
try:
    from inference.predict_queries import run_inference
except ImportError as e:
    print(f"导入错误: {e}. 请检查项目结构和 __init__.py 文件。")
    sys.exit(1)

# --- 5. 主逻辑 ---
def main():
    """执行查询推理"""
    plan_csv_path = os.path.join(TEST_DATA_DIR, "plan_info.csv")
    query_csv_path = os.path.join(TEST_DATA_DIR, "query_info.csv")
    output_csv_path = os.path.join(PREDICTION_OUTPUT_DIR, "operator_level_inference_results.csv")

    if not os.path.exists(plan_csv_path) or not os.path.exists(query_csv_path):
        print(f"错误: 推理所需的输入文件未找到。")
        sys.exit(1)

    print(f"--- 开始为数据集 '{DATASET_NAME}' 执行算子级别模型推理 ---")
    
    run_inference(
        plan_csv_path=plan_csv_path,
        query_csv_path=query_csv_path,
        use_estimates=USE_ESTIMATES_MODE, # <-- 传递开关
        output_csv_path=output_csv_path,
        no_dop_model_dir=no_dop_model_dir_path,
        dop_model_dir=dop_model_dir_path
    )
    print(f"--- 推理完成。结果已保存到: {output_csv_path} ---")

if __name__ == "__main__":
    main()