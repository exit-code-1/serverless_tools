# 文件路径: scripts/run_query_level_training.py

import sys
import os

# ==================== 1. 配置区域 ====================
DATASET_NAME = 'tpcds'
# --- 新增配置 ---
# 训练模式: 'exact_train' 或 'estimated_train'
TRAIN_MODE = 'estimated_train'
# 评估时是否使用估计值
USE_ESTIMATES_MODE = True
# =======================================================

# --- 2. 项目根目录设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 3. 动态路径定义 ---               
if DATASET_NAME == 'tpch':
    TRAIN_DATA_SOURCE = "tpch_output_500"
    TEST_DATA_SOURCE = "tpch_output_22"
elif DATASET_NAME == 'tpcds':
    TRAIN_DATA_SOURCE = "tpch_output_500"
    TEST_DATA_SOURCE = "tpcds_100g_new"
else:
    raise ValueError(f"未知的数据集名称: {DATASET_NAME}")

DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, TRAIN_DATA_SOURCE)
TEST_DATA_DIR = os.path.join(DATA_DIR, TEST_DATA_SOURCE)

# 动态定义输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", DATASET_NAME)
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models", TRAIN_MODE, "query_level")
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predictions")
EVALUATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluations")
EVALUATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluations")
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

# --- 4. 导入模块 ---
try:
    from training.query_level.train import (
        train_and_save_xgboost_onnx,
        test_onnx_xgboost,
        compute_qerror_by_bins
    )
except ImportError as e:
    print(f"导入错误: {e}. 请检查项目结构和 __init__.py 文件。")
    sys.exit(1)

# --- 5. 主逻辑 ---
def main():
    """封装主要的执行流程"""
    feature_csv = os.path.join(TRAIN_DATA_DIR, "plan_info.csv")
    true_val_csv = os.path.join(TRAIN_DATA_DIR, "query_info.csv")
    test_feature_csv = os.path.join(TEST_DATA_DIR, "plan_info.csv")
    test_execution_csv = os.path.join(TEST_DATA_DIR, "query_info.csv")

    execution_onnx_path = os.path.join(MODEL_OUTPUT_DIR, "execution_time_model.onnx")
    memory_onnx_path = os.path.join(MODEL_OUTPUT_DIR, "memory_usage_model.onnx")
    prediction_output_file = os.path.join(PREDICTION_OUTPUT_DIR, f"query_level_predictions_{TRAIN_MODE}_eval_{'estimated' if USE_ESTIMATES_MODE else 'exact'}.csv")
    qerror_output_file = os.path.join(EVALUATION_OUTPUT_DIR, f"query_level_qerror_{TRAIN_MODE}_eval_{'estimated' if USE_ESTIMATES_MODE else 'exact'}.csv")
    cost_log_file_path = os.path.join(EVALUATION_OUTPUT_DIR, "training_costs_summary.csv")
    target_dop = 96

    print(f"--- 模式: [训练={TRAIN_MODE}, 评估={'estimated' if USE_ESTIMATES_MODE else 'exact'}] ---")
    print(f"--- 开始为数据集 '{DATASET_NAME}' 运行 Query-Level 模型流程 ---")
    
    # # 训练
    # train_and_save_xgboost_onnx(
    #     feature_csv=feature_csv,
    #     true_val_csv=true_val_csv,
    #     execution_onnx_path=execution_onnx_path,
    #     memory_onnx_path=memory_onnx_path,
    #     n_trials=30,
    #     use_estimates=(TRAIN_MODE == 'estimated_train'), # <--- 关键修改
    #     cost_log_file=cost_log_file_path # <-- 传递路径
    # )
    # print(f"训练完成。模型已保存到: {MODEL_OUTPUT_DIR}")

    # 测试与评估
    print("开始测试和评估...")
    results_df = test_onnx_xgboost(
        execution_onnx_path=execution_onnx_path,
        memory_onnx_path=memory_onnx_path,
        feature_csv=test_feature_csv,
        true_val_csv=test_execution_csv,
        output_file=prediction_output_file,
        use_estimates=USE_ESTIMATES_MODE # 传递开关
    )
    print(f"测试完成。预测结果已保存到: {prediction_output_file}")

    print("开始计算 Q-error 统计...")
    compute_qerror_by_bins(results_df, qerror_output_file, target_dop)
    print(f"Q-error 计算完成。统计结果已保存到: {qerror_output_file}")
    print(f"--- 数据集 '{DATASET_NAME}' 的Query-Level模型流程完成 ---")

if __name__ == "__main__":
    main()