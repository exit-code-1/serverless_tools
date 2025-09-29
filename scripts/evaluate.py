# -*- coding: utf-8 -*-
"""
统一评估入口
整合所有评估方法
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import glob
import re

# 导入配置和工具
from config import DATASETS, DEFAULT_CONFIG
from utils import (
    setup_environment, log_experiment_start, log_experiment_end, Timer,
    get_output_paths, load_csv_safe, save_csv_safe
)

def calculate_qerror_stats(df: pd.DataFrame, time_col: str, mem_col: str) -> dict:
    """计算Q-error统计数据"""
    stats = {}
    
    # 时间 Q-error
    if time_col in df.columns:
        time_errors = pd.to_numeric(df[time_col], errors='coerce').dropna()
        time_errors = time_errors[time_errors >= 0]
        if not time_errors.empty:
            stats['p50_q_error_time'] = np.percentile(time_errors, 50)
            stats['p90_q_error_time'] = np.percentile(time_errors, 90)
            stats['avg_q_error_time'] = time_errors.mean()
    
    # 内存 Q-error
    if mem_col in df.columns:
        mem_errors = pd.to_numeric(df[mem_col], errors='coerce').dropna()
        mem_errors = mem_errors[mem_errors >= 0]
        if not mem_errors.empty:
            stats['p50_q_error_mem'] = np.percentile(mem_errors, 50)
            stats['p90_q_error_mem'] = np.percentile(mem_errors, 90)
            stats['avg_q_error_mem'] = mem_errors.mean()
    
    return stats

def evaluate_predictions(dataset: str, train_mode: str):
    """评估预测结果"""
    print(f"开始评估预测结果...")
    
    # 获取输出路径
    output_paths = get_output_paths(dataset, 'evaluation', train_mode)
    prediction_dir = output_paths['prediction_dir']
    evaluation_dir = output_paths['evaluation_dir']
    
    all_stats = []
    
    with Timer("预测结果评估"):
        # 处理算子级别预测结果
        op_level_pattern = os.path.join(prediction_dir, 'operator_level', 'operator_level_*_inference_results.csv')
        for file_path in glob.glob(op_level_pattern):
            filename = os.path.basename(file_path)
            print(f"处理算子级别文件: {filename}")
            
            match = re.search(r'operator_level_(.+)_inference_results.csv', filename)
            if not match:
                continue
            
            experiment_type = match.group(1)
            model_name = '(Operator-Level)'
            
            try:
                df = load_csv_safe(file_path, description=f"算子级别文件 {filename}")
                if df is not None:
                    stats = calculate_qerror_stats(df, 'Execution Time Q-error', 'Memory Q-error')
                    if stats:
                        stats['model'] = model_name
                        stats['experiment_type'] = experiment_type
                        all_stats.append(stats)
                        print(f"  > 模型: {model_name}, 实验: {experiment_type}, p90时间Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
            except Exception as e:
                print(f"处理文件时出错: {e}")
        
        # 处理查询级别预测结果
        query_level_pattern = os.path.join(prediction_dir, 'query_level', 'query_level_predictions_*.csv')
        for file_path in glob.glob(query_level_pattern):
            filename = os.path.basename(file_path)
            print(f"处理查询级别文件: {filename}")
            
            match = re.search(r'query_level_predictions_(.+).csv', filename)
            if not match:
                continue
            
            experiment_type = match.group(1)
            model_name = 'Query-Level (XGBoost)'
            
            try:
                df = load_csv_safe(file_path, description=f"查询级别文件 {filename}")
                if df is not None:
                    stats = calculate_qerror_stats(df, 'Execution Time Q-error', 'Memory Q-error')
                    if stats:
                        stats['model'] = model_name
                        stats['experiment_type'] = experiment_type
                        all_stats.append(stats)
                        print(f"  > 模型: {model_name}, 实验: {experiment_type}, p90时间Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
            except Exception as e:
                print(f"处理文件时出错: {e}")
        
        # 处理PPM预测结果
        ppm_patterns = [
            os.path.join(prediction_dir, 'PPM', '**', 'test_predictions.csv'),
            os.path.join(prediction_dir, 'PPM', '**', 'comparison.csv')
        ]
        
        all_ppm_files = []
        for pattern in ppm_patterns:
            all_ppm_files.extend(glob.glob(pattern, recursive=True))
        
        for file_path in all_ppm_files:
            print(f"处理PPM文件: {os.path.relpath(file_path, prediction_dir)}")
            
            path_parts = file_path.split(os.sep)
            try:
                model_type = path_parts[-2]  # NN or GNN
                train_mode_from_path = path_parts[-5].split('_', 1)[1]  # exact_train
                eval_mode_folder = path_parts[-4]  # eval_exact or eval_estimated
                
                model_name = f'PPM-{model_type}'
                experiment_type = f'{train_mode_from_path}_{eval_mode_folder}'
                
                df = load_csv_safe(file_path, description=f"PPM文件 {filename}")
                if df is not None:
                    # 检查列名
                    time_q_error_col = 'Execution Time Q-error' if 'Execution Time Q-error' in df.columns else 'q_error_time'
                    mem_q_error_col = 'Memory Q-error' if 'Memory Q-error' in df.columns else 'q_error_mem'
                    
                    stats = calculate_qerror_stats(df, time_q_error_col, mem_q_error_col)
                    if stats:
                        stats['model'] = model_name
                        stats['experiment_type'] = experiment_type
                        all_stats.append(stats)
                        print(f"  > 模型: {model_name}, 实验: {experiment_type}, p90时间Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
            except (IndexError, FileNotFoundError) as e:
                print(f"解析PPM路径或文件时出错: {file_path}, 错误: {e}")
            except Exception as e:
                print(f"处理PPM文件时出错: {e}")
    
    # 保存汇总报告
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        cols_order = ['model', 'experiment_type', 'p50_q_error_time', 'p90_q_error_time', 'avg_q_error_time', 
                     'p50_q_error_mem', 'p90_q_error_mem', 'avg_q_error_mem']
        
        for col in cols_order:
            if col not in summary_df:
                summary_df[col] = np.nan
        
        summary_df = summary_df[cols_order]
        summary_df.sort_values(by=['model', 'experiment_type'], inplace=True)
        
        summary_file = os.path.join(evaluation_dir, "qerror_summary_report.csv")
        if save_csv_safe(summary_df, summary_file, description="Q-error汇总报告"):
            print(f"\nQ-error汇总报告已保存: {summary_file}")
            print("\n报告内容预览:")
            print(summary_df.to_string())
            return True
    
    print("\n未能生成任何统计数据。")
    return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一评估入口')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='数据集名称')
    parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                       help='训练模式')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 记录实验开始
    log_experiment_start(args.dataset, 'evaluation', args.train_mode)
    
    # 执行评估
    success = evaluate_predictions(args.dataset, args.train_mode)
    
    # 记录实验结束
    if success:
        log_experiment_end(args.dataset, 'evaluation')
        print("评估完成！")
    else:
        print("评估失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
