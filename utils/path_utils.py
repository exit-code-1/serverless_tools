# -*- coding: utf-8 -*-
"""
路径工具模块
包含路径管理、目录创建、模型路径获取等功能
"""

import os
from typing import Dict

# ==================== 基础路径配置 ====================
def get_data_dir():
    """获取数据目录"""
    from config import PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "data_kunpeng")

def get_output_dir(dataset_name):
    """获取输出目录"""
    from config import PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "output", dataset_name)

def get_model_dir(train_mode):
    """获取模型目录"""
    from config import PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "output", "models", train_mode)

# get_dataset_paths 函数已移动到 utils/dataset_loader.py 中，使用固定路径

# ==================== 目录管理 ====================
def ensure_dir_exists(dir_path: str) -> None:
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)
    print(f"目录已创建: {dir_path}")

def get_experiment_name(train_mode: str, eval_mode: str) -> str:
    """生成实验名称"""
    return f"{train_mode}_{eval_mode}"

# ==================== 模型路径管理 ====================
def get_model_paths(train_mode: str, method: str) -> Dict[str, str]:
    """获取模型路径"""
    model_dir = get_model_dir(train_mode)
    
    if method in ['dop_aware', 'non_dop_aware']:
        model_type = f"operator_{method}"
        return {
            'model_dir': os.path.join(model_dir, model_type),
            'exec_model': os.path.join(model_dir, model_type, "execution_time_model.onnx"),
            'mem_model': os.path.join(model_dir, model_type, "memory_usage_model.onnx")
        }
    elif method == 'query_level':
        return {
            'model_dir': os.path.join(model_dir, "query_level"),
            'exec_model': os.path.join(model_dir, "query_level", "execution_time_model.onnx"),
            'mem_model': os.path.join(model_dir, "query_level", "memory_usage_model.onnx")
        }
    elif method == 'ppm':
        return {
            'model_dir': os.path.join(model_dir, "PPM"),
            'exec_model': None,  # PPM有自己的模型结构
            'mem_model': None
        }
    else:
        raise ValueError(f"Unknown method: {method}")

def get_output_paths(dataset: str, method: str, train_mode: str, eval_mode: str = None) -> Dict[str, str]:
    """获取输出路径"""
    output_dir = get_output_dir(dataset)
    experiment_name = get_experiment_name(train_mode, eval_mode) if eval_mode else train_mode
    
    paths = {
        'output_dir': output_dir,
        'model_dir': os.path.join(output_dir, "models", train_mode),
        'prediction_dir': os.path.join(output_dir, "predictions"),
        'evaluation_dir': os.path.join(output_dir, "evaluations"),
        'optimization_dir': os.path.join(output_dir, "optimization_results")
    }
    
    # 确保所有目录存在
    for path in paths.values():
        ensure_dir_exists(path)
    
    return paths
