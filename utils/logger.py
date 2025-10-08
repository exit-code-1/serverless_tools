# -*- coding: utf-8 -*-
"""
日志工具模块
包含日志记录、实验跟踪等功能
"""

def log_experiment_start(dataset: str, method: str, train_mode: str, eval_mode: str = None):
    """记录实验开始"""
    print("=" * 60)
    print(f"开始实验: {method.upper()} 方法")
    print(f"数据集: {dataset}")
    print(f"训练模式: {train_mode}")
    if eval_mode:
        print(f"评估模式: {eval_mode}")
    print("=" * 60)

def log_experiment_end(dataset: str, method: str, output_path: str = None):
    """记录实验结束"""
    print("=" * 60)
    print(f"实验完成: {method.upper()} 方法")
    print(f"数据集: {dataset}")
    if output_path:
        print(f"输出路径: {output_path}")
    print("=" * 60)
