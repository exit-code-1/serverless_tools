# -*- coding: utf-8 -*-
"""
对比分析脚本
整合多方法对比功能
"""

import argparse
import sys
import os

# 导入配置和工具
from config import DATASETS, DEFAULT_CONFIG
from utils import (
    setup_environment, log_experiment_start, log_experiment_end, Timer,
    safe_import, check_file_exists
)

def run_comparison(dataset: str):
    """运行多方法对比分析"""
    print(f"开始多方法对比分析...")
    
    # 导入对比函数
    compare_func = safe_import('optimization.result_processor', 'compare_results')
    if compare_func is None:
        return False
    
    # 定义输入文件路径
    input_dir = os.path.join(PROJECT_ROOT, "input/overall_performance", dataset)
    baseline_dir = os.path.join(input_dir, "tru")  # 基线
    
    baseline_query_info_path = os.path.join(baseline_dir, "query_info.csv")
    baseline_plan_info_path = os.path.join(baseline_dir, "plan_info.csv")
    
    # 检查基线文件
    if not check_file_exists(baseline_query_info_path, "基线查询信息文件"):
        return False
    if not check_file_exists(baseline_plan_info_path, "基线计划信息文件"):
        return False
    
    # 定义对比方法
    strategies = ["auto_dop", "PPM", "PIO", "POO"]
    method_info_list = []
    
    for method in strategies:
        method_dir = os.path.join(input_dir, method)
        query_info_path = os.path.join(method_dir, "query_info.csv")
        plan_info_path = os.path.join(method_dir, "plan_info.csv")
        
        if check_file_exists(query_info_path, f"方法 {method} 查询信息文件") and \
           check_file_exists(plan_info_path, f"方法 {method} 计划信息文件"):
            method_info_list.append((method, query_info_path, plan_info_path))
        else:
            print(f"警告: 方法 {method} 缺少必要文件，已跳过")
    
    if not method_info_list:
        print("错误: 没有任何有效的策略目录，无法进行对比分析")
        return False
    
    # 定义输出路径
    output_dir = os.path.join(PROJECT_ROOT, "output", "evaluations")
    os.makedirs(output_dir, exist_ok=True)
    comparison_output_csv_path = os.path.join(output_dir, "optimization_comparison_report.csv")
    
    # 执行对比
    with Timer("多方法对比分析"):
        try:
            compare_func(
                baseline_query_info_path=baseline_query_info_path,
                baseline_plan_info_path=baseline_plan_info_path,
                method_info_list=method_info_list,
                output_csv_path=comparison_output_csv_path
            )
            print(f"对比分析完成，结果已保存到: {comparison_output_csv_path}")
            return True
        except Exception as e:
            print(f"运行对比分析时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='对比分析脚本')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='数据集名称')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 记录实验开始
    log_experiment_start(args.dataset, 'comparison', 'N/A')
    
    # 执行对比
    success = run_comparison(args.dataset)
    
    # 记录实验结束
    if success:
        log_experiment_end(args.dataset, 'comparison')
        print("对比分析完成！")
    else:
        print("对比分析失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
