# 文件路径: scripts/run_optimization.py

import sys
import os
import pandas as pd
# ==================== 1. 配置区域 ====================
DATASET_NAME = 'tpcds'
TRAIN_MODE = 'exact_train' # 指定加载哪个版本的模型
USE_ESTIMATES_MODE = False # 指定评估时用什么数据
# =======================================================

# --- 2. 项目根目录设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 3. 动态路径定义 ---
if DATASET_NAME == 'tpch':
    # 优化需要包含所有DOP的数据
    ALL_DOP_DATA_SOURCE = "tpch_output_500" # 假设500这个数据集包含所有DOP
elif DATASET_NAME == 'tpcds':
    # 优化需要包含所有DOP的数据
    ALL_DOP_DATA_SOURCE = "tpcds_100g_new" # 假设这个数据集包含所有DOP
else:
    raise ValueError(f"未知的数据集名称: {DATASET_NAME}")

DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
ALL_DOP_DATA_DIR = os.path.join(DATA_DIR, ALL_DOP_DATA_SOURCE)

# 动态定义输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", DATASET_NAME)
OPTIMIZATION_RESULT_DIR = os.path.join(OUTPUT_DIR, "optimization_results")
PARSED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "parsed_results")
os.makedirs(OPTIMIZATION_RESULT_DIR, exist_ok=True)
os.makedirs(PARSED_OUTPUT_DIR, exist_ok=True)

# 通用模型路径
# --- 修改模型加载路径 ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "output", "models", TRAIN_MODE)
no_dop_model_dir_path = os.path.join(MODEL_DIR, "operator_non_dop_aware")
dop_model_dir_path = os.path.join(MODEL_DIR, "operator_dop_aware")

# --- 4. 导入模块 ---
try:
    from core.onnx_manager import ONNXModelManager
    from utils.json_parser import parse_json_to_operator_map, extract_dop_from_json
    from optimization.optimizer import run_dop_optimization
    from optimization.result_processor import save_thread_counts_from_json
except ImportError as e:
    print(f"导入错误: {e}. 请检查项目结构和 __init__.py 文件。")
    sys.exit(1)

# --- 5. 主逻辑 ---
def main():
    """执行 DOP 优化流程"""
    plan_csv_path_all_dops = os.path.join(ALL_DOP_DATA_DIR, "plan_info.csv")
    output_json_file = os.path.join(OPTIMIZATION_RESULT_DIR, "query_details_optimized.json")
    output_operators_csv = os.path.join(PARSED_OUTPUT_DIR, "operators_optimized.csv")
    output_query_dop_csv = os.path.join(PARSED_OUTPUT_DIR, "query_max_dop_optimized.csv")

    if not os.path.exists(plan_csv_path_all_dops):
        print(f"错误: 优化所需的包含所有DOP的数据文件未找到: {plan_csv_path_all_dops}")
        sys.exit(1)

    optimization_params = {
        'base_dop': 64,
        'min_improvement_ratio': 0.2,
        'min_reduction_threshold': 400,
    }

    print(f"--- 开始为数据集 '{DATASET_NAME}' 运行DOP优化流程 ---")
    df_plans = pd.read_csv(plan_csv_path_all_dops, delimiter=';', encoding='utf-8')
    
    onnx_manager = ONNXModelManager(
        no_dop_model_dir=no_dop_model_dir_path,
        dop_model_dir=dop_model_dir_path
    )

    optimization_results = run_dop_optimization(
        df_plans_all_dops=df_plans,
        onnx_manager=onnx_manager,
        output_json_path=output_json_file,
        use_estimates=USE_ESTIMATES_MODE,
        **optimization_params
    )
    # --- 新增：处理并打印计时信息 ---
    if 'timing_info' in optimization_results:
        timing = optimization_results['timing_info']
        # --- 新增：处理并保存每个查询的决策时间 ---
        per_query_times = timing.get('per_query_decision_s')
        if per_query_times:
            # 转换字典为 DataFrame
            per_query_df = pd.DataFrame(list(per_query_times.items()), columns=['query_id', 'decision_time_s'])
            per_query_df = per_query_df.sort_values(by='query_id')
                # 保存到 timing_log.csv
            timing_log_path = os.path.join(OPTIMIZATION_RESULT_DIR, "timing_log.csv")
                
                # 创建一个包含总体统计的DataFrame
            summary_timing = {k: v for k, v in timing.items() if k != 'per_query_decision_s'}
            summary_df = pd.DataFrame([summary_timing])
                
                # 将详细耗时和总体耗时写入同一个文件，或分开写
                # 这里我们选择分开写，以保持日志清晰
            per_query_timing_log_path = os.path.join(OPTIMIZATION_RESULT_DIR, "per_query_decision_timing_log.csv")
            per_query_df.to_csv(per_query_timing_log_path, index=False)
            print(f"每个查询的详细决策耗时已保存到: {per_query_timing_log_path}")
    # --- 计时信息处理结束 ---
    print("优化流程执行完毕。开始解析结果...")
    if os.path.exists(output_json_file):
        # --- 新增：调用新函数来保存 thread_count ---
        # 定义线程数CSV的输出路径
        thread_count_csv_path = os.path.join(OPTIMIZATION_RESULT_DIR, "query_thread_counts.csv")
        print(f"  提取查询总线程数到: {thread_count_csv_path}")
        save_thread_counts_from_json(output_json_file, thread_count_csv_path)
        # --- 调用结束 ---
        parse_json_to_operator_map(output_json_file, output_operators_csv)
        extract_dop_from_json(output_json_file, output_query_dop_csv)
        print(f"结果解析完成。JSON保存在: {output_json_file}")
        print(f"解析后的CSV保存在: {PARSED_OUTPUT_DIR}")
    else:
        print("错误: 未找到优化输出的JSON文件。")
    print(f"--- 数据集 '{DATASET_NAME}' 的DOP优化流程完成 ---")

if __name__ == "__main__":
    main()