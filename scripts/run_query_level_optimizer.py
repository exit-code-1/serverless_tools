# 文件路径: scripts/run_optimization.py

import sys
import os
import pandas as pd

# --- 步骤 1: 将项目根目录添加到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"将项目根目录添加到 sys.path: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print("项目根目录已添加到 sys.path。")
else:
    print("项目根目录已在 sys.path 中。")

# --- 步骤 2: 导入需要的模块和函数 ---
try:
    from core.onnx_manager import ONNXModelManager
    from utils.json_parser import parse_json_to_operator_map, extract_dop_from_json, generate_query_file
    from optimization.optimizer import run_query_dop_optimization
    from optimization.result_processor import save_thread_counts_from_json
    # 如果需要运行后续的对比或选择函数，也在这里导入
    # from optimization.result_processor import compare_results, select_optimal_dops_from_prediction_file
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请检查:")
    print(f"1. 项目结构是否正确，并且 '{PROJECT_ROOT}' 确实是你的项目根目录。")
    print(f"2. 必要的模块文件是否存在 (例如 optimization/optimizer.py)。")
    print(f"3. 相关的 __init__.py 文件是否存在于各级包目录中。")
    sys.exit(1)
except FileNotFoundError:
     print(f"错误: 必要的模块文件未找到。")
     sys.exit(1)

# --- 步骤 3: 定义输入数据路径 ---
# (请根据你的实际数据存放位置修改这些示例路径)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
# 优化流程通常需要包含所有 DOP 的数据
# 假设数据在 tpch_10g_output_22 中 (如果你的训练和测试用了不同的数据集，这里需要调整)
ALL_DOP_DATA_DIR = os.path.join(DATA_DIR, "tpcds_100g_new") # 或者包含 500 和 22 合并的数据集
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
plan_csv_path_all_dops = os.path.join(ALL_DOP_DATA_DIR, "plan_info.csv")

# ... (定义输入输出路径，包括 JSON 和解析后的 CSV 路径) ...
# 检查输入文件是否存在
if not os.path.exists(plan_csv_path_all_dops):
    print(f"错误: 输入 plan 文件 (所有DOP) 未找到: {plan_csv_path_all_dops}")
    sys.exit(1)
    
algorithm = 'PPM'
data_set = 'tpcds'
# --- 步骤 4: 定义模型目录和输出路径 ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", algorithm, data_set)
MODEL_DIR = os.path.join(PROJECT_ROOT, "output/models")
OPTIMIZATION_RESULT_DIR = os.path.join(OUTPUT_DIR,  "optimization_results")
os.makedirs(OPTIMIZATION_RESULT_DIR, exist_ok=True) # 确保目录存在
input_csv_file = os.path.join(INPUT_DIR, algorithm, data_set, "test_predictions.csv")
if algorithm == 'pipeline_query':
    input_csv_file = os.path.join(OPTIMIZATION_RESULT_DIR, "query_max_dop_optimized.csv")

output_json_file = os.path.join(OPTIMIZATION_RESULT_DIR, "query_details_optimized.json")
output_operators_csv = os.path.join(OPTIMIZATION_RESULT_DIR, "operators_optimized.csv")
output_query_dop_csv = os.path.join(OPTIMIZATION_RESULT_DIR, "query_max_dop_optimized.csv")

no_dop_model_dir_path = os.path.join(MODEL_DIR, "operator_non_dop_aware")
dop_model_dir_path = os.path.join(MODEL_DIR, "operator_dop_aware")

# 定义输出 JSON 文件路径
output_json_file = os.path.join(OPTIMIZATION_RESULT_DIR, "query_details_optimized.json")
# (可选) 定义其他输出文件路径
# comparison_output_file = os.path.join(OPTIMIZATION_RESULT_DIR, "optimization_comparison.csv")
# selected_dop_output_file = os.path.join(OPTIMIZATION_RESULT_DIR, "selected_optimal_dops.csv")

# --- 步骤 5: 定义优化参数 (阈值等) ---
# 这些参数会通过 **kwargs 传递给 choose_optimal_dop
optimization_params = {
    'base_dop': 64, # 基准 DOP
    'child_time_default': 1e-6,
    'min_improvement_ratio': 0.2,
    'min_reduction_threshold': 200,
    # 可以从配置文件加载这些参数
}

# --- 步骤 6: 主逻辑函数 ---
def main():
    """执行 DOP 优化流程"""
    print(f"开始加载所有 DOP 的计划数据: {plan_csv_path_all_dops}")
    try:
        df_plans = pd.read_csv(plan_csv_path_all_dops, delimiter=';', encoding='utf-8')
        print("数据加载完成。")
    except Exception as e:
        print(f"加载计划数据时出错: {e}")
        sys.exit(1)

    print("初始化 ONNX 模型管理器...")
    onnx_manager = ONNXModelManager(
        no_dop_model_dir=no_dop_model_dir_path,
        dop_model_dir=dop_model_dir_path
    )
    if not onnx_manager.exec_sessions and not onnx_manager.mem_sessions:
         print("警告: ONNXModelManager 未加载任何模型，优化结果可能不准确。")


    print("\n开始运行 DOP 优化...")
    optimization_results = run_query_dop_optimization(
        csv_file=input_csv_file,
        df_plans_all_dops=df_plans,
        onnx_manager=onnx_manager,
        output_json_path=output_json_file, # 传递 JSON 输出路径
        algorithm=algorithm,
        **optimization_params # 传递优化参数
    )

    if optimization_results:
        print("优化流程执行完毕。")
        # --- 新增：处理并打印计时信息 ---
        if 'timing_info' in optimization_results:
            timing = optimization_results['timing_info']
            print("\n--- 优化流程开销报告 ---")
            print(f"  - 数据准备与预测耗时: {timing.get('data_prep_and_prediction_s', 0):.4f} 秒")
            print(f"  - DOP 选择耗时:        {timing.get('dop_selection_s', 0):.4f} 秒")
            print(f"  - 总优化流程耗时:     {timing.get('total_optimization_s', 0):.4f} 秒")
            print("--------------------------\n")

            # (可选) 将计时信息保存到文件
            timing_log_path = os.path.join(OPTIMIZATION_RESULT_DIR, "timing_log.csv")
            pd.DataFrame([timing]).to_csv(timing_log_path, index=False)
            print(f"详细计时日志已保存到: {timing_log_path}")
        # --- 计时信息处理结束 ---
        print("\n开始解析优化结果 JSON 并生成 CSV...")
        try:
            if os.path.exists(output_json_file):
                # --- 新增：调用新函数来保存 thread_count ---
                # 定义线程数CSV的输出路径
                thread_count_csv_path = os.path.join(OPTIMIZATION_RESULT_DIR, "query_thread_counts.csv")
                print(f"  提取查询总线程数到: {thread_count_csv_path}")
                save_thread_counts_from_json(output_json_file, thread_count_csv_path)
                # --- 调用结束 ---
                print(f"  生成算子级别 CSV 到: {output_operators_csv}")
                parse_json_to_operator_map(output_json_file, output_operators_csv)

                print(f"  提取查询最大 DOP 到: {output_query_dop_csv}")
                extract_dop_from_json(output_json_file, output_query_dop_csv)

                # (可选) 生成单个查询文件
                # target_query_id = 17
                # single_query_output = os.path.join(PARSED_OUTPUT_DIR, f"query_{target_query_id}_operators.csv")
                # if os.path.exists(output_operators_csv):
                #     generate_query_file(output_operators_csv, target_query_id, single_query_output)

            else:
                print(f"  错误: 优化结果 JSON 文件未找到，无法生成后续 CSV。")
        except NameError: # 如果 json_parser 函数导入失败
             print("  错误: 未能导入 json_parser 函数，跳过解析步骤。")
        except Exception as e:
            print(f"  解析 JSON 或生成 CSV 时出错: {e}")
        # --- 结束调用 ---
        # --- (可选) 在这里执行后续的对比或DOP选择保存 ---
        # print("\n(可选) 运行结果对比...")
        # compare_results(baseline_query_info_path="path/to/baseline_query.csv", ...) # 需要定义基线路径
        # print("\n(可选) 从预测文件选择最优DOP...")
        # select_optimal_dops_from_prediction_file(csv_file="path/to/predictions.csv", output_file=selected_dop_output_file) # 需要定义预测文件路径
    else:
        print("优化流程未能成功完成。")


# --- 步骤 7: 脚本入口点 ---
if __name__ == "__main__":
    main()