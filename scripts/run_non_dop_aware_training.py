# 文件路径: scripts/run_non_dop_aware_training.py

import sys
import os
import pandas as pd

# ==================== 1. 配置区域 ====================
# 在这里切换您要运行的数据集: 'tpch' 或 'tpcds'
DATASET_NAME = 'tpcds'
# 训练模式: 'exact_train' 或 'estimated_train'
TRAIN_MODE = 'estimated_train' 
# =======================================================

# --- 2. 项目根目录设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 3. 动态路径定义 ---
# 根据 DATASET_NAME 动态选择数据源
if DATASET_NAME == 'tpch':
    # !! 请确认您的 TPC-H 训练和测试数据目录名 !!
    TRAIN_DATA_SOURCE = "tpch_output_500"
    TEST_DATA_SOURCE = "tpch_output_22"
elif DATASET_NAME == 'tpcds':
    # !! 请确认您的 TPC-DS 训练和测试数据目录名 !!
    TRAIN_DATA_SOURCE = "tpcds_100g_output"
    TEST_DATA_SOURCE = "tpcds_100g_new_test"
else:
    raise ValueError(f"未知的数据集名称: {DATASET_NAME}")

DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng") # 统一数据源目录
TRAIN_DATA_DIR = os.path.join(DATA_DIR, TRAIN_DATA_SOURCE)
TEST_DATA_DIR = os.path.join(DATA_DIR, TEST_DATA_SOURCE)

# 动态定义输出目录，避免结果混淆
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", DATASET_NAME)
# 模型输出路径不与数据集绑定，因为模型是通用的
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "models") 
os.makedirs(os.path.join(MODEL_OUTPUT_DIR, "operator_non_dop_aware"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "evaluations", "operator_non_dop_aware", "operator_comparisons"), exist_ok=True)


# --- 4. 导入模块 ---
try:
    from training.operator_non_dop_aware.train import train_all_operators
except ImportError as e:
    print(f"导入错误: {e}. 请检查项目结构和 __init__.py 文件。")
    sys.exit(1)

# --- 5. 主逻辑 ---
def main():
    """执行非 DOP 感知算子训练"""
    train_plan_info_csv = os.path.join(TRAIN_DATA_DIR, "plan_info.csv")
    test_plan_info_csv = os.path.join(TEST_DATA_DIR, "plan_info.csv")

    if not os.path.exists(train_plan_info_csv) or not os.path.exists(test_plan_info_csv):
        print(f"错误: 训练或测试数据文件未找到。")
        print(f"  - 训练文件检查: {train_plan_info_csv}")
        print(f"  - 测试文件检查: {test_plan_info_csv}")
        sys.exit(1)

    print(f"--- 开始为数据集 '{DATASET_NAME}' 训练非DOP感知模型 ---")
    print(f"训练数据源: {TRAIN_DATA_DIR}")
    print(f"测试数据源: {TEST_DATA_DIR}")

    train_data = pd.read_csv(train_plan_info_csv, delimiter=';', encoding='utf-8')
    test_data = pd.read_csv(test_plan_info_csv, delimiter=';', encoding='utf-8')
    
    # 假设 train_all_operators 函数已经适配，可以处理输出路径
    # 如果没有，需要修改 train_all_operators 函数以接受 output_dir 参数
    train_all_operators(
        train_data=train_data,
        test_data=test_data,
        total_queries=500, # 这个参数似乎是固定的
        train_ratio=1,     # 这个参数似乎是固定的
        # (可选) 传递动态输出目录给训练函数
        # output_dir=OUTPUT_DIR 
        # 训练时是否使用估计值，由 TRAIN_MODE 决定
        use_estimates=(TRAIN_MODE == 'estimated_train') 
    )
    print(f"--- 数据集 '{DATASET_NAME}' 的非DOP感知模型训练完成 ---")
    print(f"请在 'output/models/operator_non_d_aware/' 和 '{os.path.join(OUTPUT_DIR, 'evaluations')}' 查看结果。")

if __name__ == "__main__":
    main()