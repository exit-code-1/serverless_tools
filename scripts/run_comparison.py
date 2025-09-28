# 文件路径: scripts/run_comparison.py

import sys
import os
import pandas as pd # 保留导入

# --- 步骤 1: 将项目根目录添加到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"将项目根目录添加到 sys.path: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print("项目根目录已添加到 sys.path。")
else:
    print("项目根目录已在 sys.path 中。")

# --- 步骤 2: 导入需要的对比函数 ---
try:
    # 从 optimization 包的 result_processor 子模块中导入 compare_results
    from optimization.result_processor import compare_results
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请检查:")
    print(f"1. 项目结构是否正确，并且 '{PROJECT_ROOT}' 确实是你的项目根目录。")
    print(f"2. optimization/result_processor.py 文件是否存在且包含 compare_results 函数。")
    print(f"3. 相关的 __init__.py 文件是否存在于各级包目录中 (optimization/ 等)。")
    sys.exit(1)
except FileNotFoundError:
     print(f"错误: 模块文件 'optimization/result_processor.py' 未找到。")
     sys.exit(1)

# --- 步骤 3: 定义输入文件路径 ---
# !! 关键 !!: 将原始调用中的硬编码路径替换为基于 PROJECT_ROOT 和 data/ 或实验结果目录的路径
# 你需要将实际运行基准策略和优化策略后收集到的数据文件放在这些路径下
DATA_SET = "tpcds"
DATA_DIR = os.path.join(PROJECT_ROOT, "input/overall_performance", DATA_SET)
BASELINE_DATA_DIR = os.path.join(DATA_DIR, "tru")  # 基线
baseline_query_info_path = os.path.join(BASELINE_DATA_DIR, "query_info.csv")
baseline_plan_info_path = os.path.join(BASELINE_DATA_DIR, "plan_info.csv")

# 多个策略（方法名, query_info_path, plan_info_path）
strategies = ["auto_dop", "PPM", "PIO", "POO"]  # 可以修改为你自己的目录名列表
method_info_list = []
for method in strategies:
    method_dir = os.path.join(DATA_DIR, method)
    query_info_path = os.path.join(method_dir, "query_info.csv")
    plan_info_path = os.path.join(method_dir, "plan_info.csv")

    # 检查文件存在
    if not os.path.exists(query_info_path) or not os.path.exists(plan_info_path):
        print(f"警告: 策略 {method} 缺少必要文件，已跳过。")
        continue
    method_info_list.append((method, query_info_path, plan_info_path))

# 检查基线文件是否存在
required_files = [baseline_query_info_path, baseline_plan_info_path]
for f_path in required_files:
    if not os.path.exists(f_path):
        print(f"错误: 缺少基线输入文件: {f_path}")
        sys.exit(1)

if not method_info_list:
    print("错误: 没有任何有效的策略目录，无法进行对比分析。")
    sys.exit(1)

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "evaluations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
comparison_output_csv_path = os.path.join(OUTPUT_DIR, "optimization_comparison_report.csv")
# --- 步骤 5: 主逻辑函数 ---
def main():
    """执行对比分析"""
    print("开始运行优化效果对比...")

    print(f"  基准目录: {BASELINE_DATA_DIR}")
    print(f"  策略数量: {len(method_info_list)}")
    print(f"  输出路径: {comparison_output_csv_path}")

    try:
        # 调用多策略对比函数
        compare_results(
            baseline_query_info_path=baseline_query_info_path,
            baseline_plan_info_path=baseline_plan_info_path,
            method_info_list=method_info_list,
            output_csv_path=comparison_output_csv_path
        )
    except Exception as e:
        print(f"\n运行 compare_multiple_results 时发生错误: {e}")
        import traceback
        traceback.print_exc()

# --- 步骤 6: 脚本入口点 ---
if __name__ == "__main__":
    main()