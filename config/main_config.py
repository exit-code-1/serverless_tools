# -*- coding: utf-8 -*-
"""
统一配置管理模块
包含所有脚本的公共配置和路径定义
"""

import os
import sys

# ==================== 项目根目录设置 ====================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================== 数据集配置 ====================
DATASETS = {
    'tpch': {
        'train_dir': 'tpch_output_500',
        'test_dir': 'tpch_output_22',
        'plan_info_file': 'plan_info.csv',
        'query_info_file': 'query_info.csv'
    },
    'tpcds': {
        'train_dir': 'tpcds_100g_output_train',
        'test_dir': 'tpcds_100g_output_test',
        'plan_info_file': 'plan_info.csv',
        'query_info_file': 'query_info.csv'
    }
}

# ==================== 方法配置 ====================
METHODS = {
    'dop_aware': {
        'name': 'DOP感知算子模型',
        'module': 'training.operator_dop_aware.train',
        'function': 'train_all_operators'
    },
    'non_dop_aware': {
        'name': '非DOP感知算子模型',
        'module': 'training.operator_non_dop_aware.train', 
        'function': 'train_all_operators'
    },
    'ppm': {
        'name': 'PPM方法',
        'module': 'training.PPM.GNN_train',  # 或 NN_train
        'function': 'main'
    },
    'query_level': {
        'name': '查询级别模型',
        'module': 'training.query_level.train',
        'function': 'train_and_save_xgboost_onnx'
    },
    'mci': {
        'name': 'MCI方法',
        'module': 'scripts.main',
        'function': 'run_mci_training'
    }
}

# ==================== 训练模式配置 ====================
TRAIN_MODES = {
    'exact_train': {
        'name': '精确训练',
        'use_estimates': False
    },
    'estimated_train': {
        'name': '估计训练', 
        'use_estimates': True
    }
}

# ==================== 优化算法配置 ====================
OPTIMIZATION_ALGORITHMS = {
    'pipeline': {
        'name': 'Pipeline优化',
        'module': 'optimization.optimizer',
        'function': 'run_dop_optimization'
    },
    'query_level': {
        'name': '查询级别优化',
        'module': 'optimization.optimizer', 
        'function': 'run_query_dop_optimization'
    },
    'auto_dop': {
        'name': 'Auto-DOP',
        'algorithm': 'tru'
    },
    'ppm': {
        'name': 'PPM方法',
        'algorithm': 'PPM'
    }
}

# ==================== 系统配置 ====================
# DOP相关配置
DEFAULT_DOP = 64
THREAD_COST = 6
THREAD_MEM = 8194
DOP_SETS = {1, 8, 16, 32, 64, 96}

# 特征开关配置
USE_HASH_TABLE_SIZE_FEATURE = True

# ==================== 默认配置 ====================
DEFAULT_CONFIG = {
    'dataset': 'tpcds',
    'train_mode': 'estimated_train',
    'use_estimates_mode': True,
    'base_dop': DEFAULT_DOP,
    'min_improvement_ratio': 0.2,
    'min_reduction_threshold': 200,
    'target_dop': 96,
    'train_ratio': 1.0,
    'total_queries': 500
}

# ==================== 向后兼容导入 ====================
# 路径配置函数已移动到 utils/path_utils.py
# 数据集路径配置已移动到 utils/dataset_loader.py

# 验证函数已移动到 utils/system_utils.py
# 使用延迟导入避免循环导入
def validate_dataset(dataset_name):
    """验证数据集名称是否有效"""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    return True

def validate_method(method_name):
    """验证方法名称是否有效"""
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHODS.keys())}")
    return True

def validate_train_mode(train_mode):
    """验证训练模式是否有效"""
    if train_mode not in TRAIN_MODES:
        raise ValueError(f"Unknown train mode: {train_mode}. Available: {list(TRAIN_MODES.keys())}")
    return True
