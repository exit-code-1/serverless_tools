# 文件路径: scripts/run_evaluation.py (最终修复版)

import sys
import os
import pandas as pd
import numpy as np
import glob
import re

# ==================== 1. 项目根目录设置 ====================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =================================================================

def calculate_stats(df, time_col, mem_col):
    """从DataFrame计算Q-error统计数据"""
    stats = {}
    
    # --- 时间 Q-error ---
    if time_col in df.columns:
        time_errors = pd.to_numeric(df[time_col], errors='coerce').dropna()
        time_errors = time_errors[time_errors >= 0]
        if not time_errors.empty:
            stats['p50_q_error_time'] = np.percentile(time_errors, 50)
            stats['p90_q_error_time'] = np.percentile(time_errors, 90)
            stats['avg_q_error_time'] = time_errors.mean()
    
    # --- 内存 Q-error ---
    if mem_col in df.columns:
        mem_errors = pd.to_numeric(df[mem_col], errors='coerce').dropna()
        mem_errors = mem_errors[mem_errors >= 0]
        if not mem_errors.empty:
            stats['p50_q_error_mem'] = np.percentile(mem_errors, 50)
            stats['p90_q_error_mem'] = np.percentile(mem_errors, 90)
            stats['avg_q_error_mem'] = mem_errors.mean()
            
    return stats

def main():
    """
    遍历指定目录下的所有预测结果CSV，为每个文件计算 p50, p90, avg Q-error，
    并生成一个汇总的对比报告。
    """
    # ==================== 2. 配置区域 ====================
    DATASET_NAME = 'tpcds' # 在这里切换数据集
    # =======================================================
    
    PREDICTION_DIR = os.path.join(PROJECT_ROOT, "output", DATASET_NAME, "predictions")
    EVALUATION_DIR = os.path.join(PROJECT_ROOT, "output", DATASET_NAME, "evaluations")
    SUMMARY_REPORT_PATH = os.path.join(EVALUATION_DIR, "qerror_summary_report.csv")
    
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    all_stats = []
    print(f"--- 开始为数据集 '{DATASET_NAME}' 计算Q-error统计 ---")
    print(f"扫描目录: {PREDICTION_DIR}")

    # --- Case 1: 处理 operator_level (Auto-DOP) 模型 ---
    op_level_pattern = os.path.join(PREDICTION_DIR, 'operator_level', 'operator_level_*_inference_results.csv')
    for file_path in glob.glob(op_level_pattern):
        filename = os.path.basename(file_path)
        print(f"\n正在处理文件 (Auto-DOP): {filename}")
        
        match = re.search(r'operator_level_(.+)_inference_results.csv', filename)
        if not match: continue
        experiment_type = match.group(1)
        model_name = '(Operator-Level)'

        try:
            df = pd.read_csv(file_path, sep=';')
            stats = calculate_stats(df, 'Execution Time Q-error', 'Memory Q-error')
            if stats:
                stats['model'] = model_name
                stats['experiment_type'] = experiment_type
                all_stats.append(stats)
                print(f"  > 模型: {model_name}, 实验: {experiment_type}, p90时间Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
        except Exception as e:
            print(f"  处理文件时出错: {e}")

    # --- Case 2: 处理 query_level (XGBoost) 模型 ---
    query_level_pattern = os.path.join(PREDICTION_DIR, 'query_level', 'query_level_predictions_*.csv')
    for file_path in glob.glob(query_level_pattern):
        filename = os.path.basename(file_path)
        print(f"\n正在处理文件 (Query-Level): {filename}")
        
        match = re.search(r'query_level_predictions_(.+).csv', filename)
        if not match: continue
        experiment_type = match.group(1)
        model_name = 'Query-Level (XGBoost)'

        try:
            df = pd.read_csv(file_path, sep=';')
            stats = calculate_stats(df, 'Execution Time Q-error', 'Memory Q-error')
            if stats:
                stats['model'] = model_name
                stats['experiment_type'] = experiment_type
                all_stats.append(stats)
                print(f"  > 模型: {model_name}, 实验: {experiment_type}, p90时间Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
        except Exception as e:
            print(f"  处理文件时出错: {e}")

    # --- Case 3: 处理 PPM 模型 ---
    # !! 修改: 匹配 test_predictions.csv (NN) 和 comparison.csv (GNN) !!
    ppm_patterns = [
        os.path.join(PREDICTION_DIR, 'PPM', '**', 'test_predictions.csv'),
        os.path.join(PREDICTION_DIR, 'PPM', '**', 'comparison.csv')
    ]
    
    all_ppm_files = []
    for pattern in ppm_patterns:
        all_ppm_files.extend(glob.glob(pattern, recursive=True))

    for file_path in all_ppm_files:
        print(f"\n正在处理文件 (PPM): {os.path.relpath(file_path, PREDICTION_DIR)}")
        
        path_parts = file_path.split(os.sep)
        try:
            model_type = path_parts[-2] # NN or GNN
            train_mode = path_parts[-5].split('_', 1)[1] # exact_train
            eval_mode_folder = path_parts[-4] # eval_exact or eval_estimated
            
            model_name = f'PPM-{model_type}'
            experiment_type = f'{train_mode}_{eval_mode_folder}'

            df = pd.read_csv(file_path, sep=';')
            
            # !! 修改: GNN的列名是 'Execution Time Q-error'，NN是 'Execution Time Q-error' !!
            # 为了统一，我们检查两种可能性
            time_q_error_col = 'Execution Time Q-error' if 'Execution Time Q-error' in df.columns else 'q_error_time'
            mem_q_error_col = 'Memory Q-error' if 'Memory Q-error' in df.columns else 'q_error_mem'
            
            stats = calculate_stats(df, time_q_error_col, mem_q_error_col)
            if stats:
                stats['model'] = model_name
                stats['experiment_type'] = experiment_type
                all_stats.append(stats)
                print(f"  > 模型: {model_name}, 实验: {experiment_type}, p90时间Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
        except (IndexError, FileNotFoundError) as e:
            print(f"  解析PPM路径或文件时出错: {file_path}, 错误: {e}")
        except Exception as e:
            print(f"  处理PPM文件时出错: {e}")
            
    # --- 保存汇总报告 ---
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        cols_order = ['model', 'experiment_type', 'p50_q_error_time', 'p90_q_error_time', 'avg_q_error_time', 'p50_q_error_mem', 'p90_q_error_mem', 'avg_q_error_mem']
        for col in cols_order:
            if col not in summary_df:
                summary_df[col] = np.nan
        summary_df = summary_df[cols_order]
        summary_df.sort_values(by=['model', 'experiment_type'], inplace=True)
        
        summary_df.to_csv(SUMMARY_REPORT_PATH, index=False, sep=';', float_format='%.4f')
        print(f"\n--- Q-error 汇总报告已成功保存到 ---\n{SUMMARY_REPORT_PATH}")
        print("\n报告内容预览:")
        print(summary_df.to_string())
    else:
        print("\n未能生成任何统计数据。")

if __name__ == "__main__":
    main()