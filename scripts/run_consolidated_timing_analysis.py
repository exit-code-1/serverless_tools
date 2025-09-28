# scripts/run_consolidated_timing_analysis.py (新版本)

import sys
import os
import pandas as pd
import glob

def process_one_experiment(dataset_name, experiment_name, project_root):
    """
    处理单个实验场景（如 'exact_exact'）的计时数据整合。
    
    Args:
        dataset_name (str): 'tpch' 或 'tpcds'
        experiment_name (str): 实验场景的目录名，如 'exact_exact'
        project_root (str): 项目的根目录
    """
    print(f"\n{'='*20} 开始处理实验: {dataset_name} / {experiment_name} {'='*20}")

    # --- 1. 定义输入文件路径 ---
    # 新的目录结构，例如：output/tpcds/optimization/exact_exact/
    base_dir = os.path.join(project_root, "output", dataset_name)
    
    # 决策时间日志路径
    decision_timing_file = os.path.join(base_dir, "optimization", experiment_name, "optimization_results", "per_query_decision_timing_log.csv")
    
    # 推理时间日志路径
    # 我们需要根据 experiment_name 构建正确的文件名
    # 例如 'exact_exact' -> 'operator_level_exact_exact_inference_results.csv'
    inference_timing_filename = f"operator_level_{experiment_name}_inference_results.csv"
    inference_timing_file = os.path.join(base_dir, "predictions", "operator_level", inference_timing_filename)

    # 定义最终输出文件
    evaluation_dir = os.path.join(base_dir, "evaluations")
    os.makedirs(evaluation_dir, exist_ok=True)
    output_summary_file = os.path.join(evaluation_dir, f"total_online_cost_summary_{experiment_name}.csv")

    # --- 2. 检查输入文件是否存在 ---
    if not os.path.exists(decision_timing_file):
        print(f"  -> 警告: 决策时间日志文件未找到，跳过此实验: {decision_timing_file}")
        return
        
    if not os.path.exists(inference_timing_file):
        print(f"  -> 警告: 推理时间日志文件未找到，跳过此实验: {inference_timing_file}")
        return

    print(f"  读取决策时间从: {decision_timing_file}")
    print(f"  读取推理时间从: {inference_timing_file}")

    # --- 3. 加载、合并并计算 ---
    try:
        df_decision = pd.read_csv(decision_timing_file)
        df_inference = pd.read_csv(inference_timing_file, sep=';')
        
        inference_cols_to_keep = ['Query ID', 'Time Calculation Duration (s)', 'Memory Calculation Duration (s)']
        if not all(col in df_inference.columns for col in inference_cols_to_keep):
            print(f"  -> 错误: 推理时间日志文件 '{inference_timing_file}' 缺少必要的列。")
            return
            
        df_inference = df_inference[inference_cols_to_keep]
        
        df_decision.rename(columns={'query_id': 'Query ID'}, inplace=True)
        df_inference.rename(columns={
            'Time Calculation Duration (s)': 'pred_exec_time_s',
            'Memory Calculation Duration (s)': 'pred_mem_time_s'
        }, inplace=True)
        
        df_merged = pd.merge(df_decision, df_inference, on='Query ID', how='outer').fillna(0)
        
        df_merged['total_online_cost_s'] = df_merged['decision_time_s'] + \
                                           df_merged['pred_exec_time_s'] + \
                                           df_merged['pred_mem_time_s']
                                           
        df_merged.sort_values(by='Query ID', inplace=True)
        
        df_merged.to_csv(output_summary_file, index=False, sep=';')
        
        print(f"  -> 成功！整合报告已保存到: {output_summary_file}")
        
    except Exception as e:
        print(f"  -> 处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    自动检测所有实验场景，并为每个场景生成整合的在线开销报告。
    """
    # ==================== 配置区域 ====================
    DATASET_NAME = 'tpcds'
    # =================================================

    # --- 项目根目录设置 ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # --- 自动检测所有实验场景 ---
    # 查找 optimization 目录下的所有子目录作为实验名
    optimization_base_dir = os.path.join(PROJECT_ROOT, "output", DATASET_NAME, "optimization")
    if not os.path.isdir(optimization_base_dir):
        print(f"错误: 优化结果目录未找到: {optimization_base_dir}")
        print("请先运行至少一个优化实验。")
        sys.exit(1)
        
    experiment_names = [d for d in os.listdir(optimization_base_dir) if os.path.isdir(os.path.join(optimization_base_dir, d))]
    
    if not experiment_names:
        print(f"在 '{optimization_base_dir}' 中没有找到任何实验子目录 (如 'exact_exact')。")
        sys.exit(1)
        
    print(f"在数据集 '{DATASET_NAME}' 上检测到以下实验场景: {experiment_names}")

    # --- 为每个实验场景运行整合分析 ---
    for exp_name in experiment_names:
        process_one_experiment(DATASET_NAME, exp_name, PROJECT_ROOT)

if __name__ == "__main__":
    main()