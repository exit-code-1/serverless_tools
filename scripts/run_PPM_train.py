# 文件路径: scripts/run_PPM_train.py

import sys
import os

# ==================== 1. 配置区域 ====================
# 在这里切换您要运行的模型: 'NN' 或 'GNN'
METHOD = 'GNN'
# 在这里切换您s要运行的数据集: 'tpch' 或 'tpcds'
DATASET_NAME = 'tpcds'
# 设置为 True 来运行基数估计模拟实验
# --- 新增配置 ---
TRAIN_MODE = 'estimated_train'
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
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models", TRAIN_MODE, "PPM", METHOD)
# 根据评估模式，确定最终预测结果的子目录名
eval_mode_str = "eval_estimated" if USE_ESTIMATES_MODE else "eval_exact"
# 将 训练模式 和 评估模式 都加入到预测输出路径中
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predictions", f"train_{TRAIN_MODE}", eval_mode_str, "PPM", METHOD)
EVALUATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluations")
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

# --- 4. 导入模块 ---
try:
    from training.PPM.GNN_train import train_gnn_models, evaluate_gnn_models
    from training.PPM.NN_train import train_nn_models, evaluate_nn_models
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
    cost_log_file_path = os.path.join(EVALUATION_OUTPUT_DIR, "training_costs_summary.csv")
    print(f"--- 开始为数据集 '{DATASET_NAME}' 训练PPM-{METHOD}模型 ---")

    if METHOD == 'NN':
        # print("训练PPM-NN模型...")
        # train_nn_models(
        #     feature_csv=feature_csv,
        #     query_info_train_csv=true_val_csv,
        #     execution_onnx_path=execution_onnx_path,
        #     memory_onnx_path=memory_onnx_path,
        #     use_estimates=(TRAIN_MODE == 'estimated_train'), # <--- 关键修改
        #     cost_log_file=cost_log_file_path # <-- 传递路径
        # )
        print("评估PPM-NN模型...")
        evaluate_nn_models(
            query_features_test_csv=test_feature_csv,
            query_info_test_csv=test_execution_csv,
            execution_onnx_path=execution_onnx_path,
            memory_onnx_path=memory_onnx_path,
            output_file=PREDICTION_OUTPUT_DIR,
            use_estimates=USE_ESTIMATES_MODE # 传递开关
        )
    elif METHOD == 'GNN':
        # print("训练PPM-GNN模型...")
        # train_gnn_models(
        #     query_features_train_csv=feature_csv,
        #     query_info_train_csv=true_val_csv,
        #     execution_onnx_path=execution_onnx_path,
        #     memory_onnx_path=memory_onnx_path,
        #     use_estimates=(TRAIN_MODE == 'estimated_train'), # <--- 关键修改
        #     cost_log_file=cost_log_file_path # <-- 传递路径
        # )
        print("评估PPM-GNN模型...")
        evaluate_gnn_models(
            query_features_test_csv=test_feature_csv,
            query_info_test_csv=test_execution_csv,
            execution_onnx_path=execution_onnx_path,
            memory_onnx_path=memory_onnx_path,
            output_file=PREDICTION_OUTPUT_DIR,
            use_estimates=USE_ESTIMATES_MODE # 传递开关
        )
    else:
        raise ValueError(f"未知的PPM方法: {METHOD}")
        
    print(f"--- 数据集 '{DATASET_NAME}' 的PPM-{METHOD}模型流程完成 ---")
    print(f"模型保存在: {MODEL_OUTPUT_DIR}")
    print(f"预测结果保存在: {PREDICTION_OUTPUT_DIR}")

if __name__ == "__main__":
    main()