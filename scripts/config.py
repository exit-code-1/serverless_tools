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
        'train_source': 'tpch_output_500',
        'test_source': 'tpch_output_22',
        'all_dop_source': 'tpch_output_500'  # 包含所有DOP的数据
    },
    'tpcds': {
        'train_source': 'tpcds_100g_output',
        'test_source': 'tpcds_100g_new_test', 
        'all_dop_source': 'tpcds_100g_new'  # 包含所有DOP的数据
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

# ==================== 路径配置 ====================
def get_data_dir():
    """获取数据目录"""
    return os.path.join(PROJECT_ROOT, "data_kunpeng")

def get_output_dir(dataset_name):
    """获取输出目录"""
    return os.path.join(PROJECT_ROOT, "output", dataset_name)

def get_model_dir(train_mode):
    """获取模型目录"""
    return os.path.join(PROJECT_ROOT, "output", "models", train_mode)

def get_dataset_paths(dataset_name, mode='train'):
    """获取数据集路径"""
    data_dir = get_data_dir()
    dataset_config = DATASETS[dataset_name]
    
    if mode == 'train':
        source = dataset_config['train_source']
    elif mode == 'test':
        source = dataset_config['test_source']
    elif mode == 'all_dop':
        source = dataset_config['all_dop_source']
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    dataset_dir = os.path.join(data_dir, source)
    plan_info_path = os.path.join(dataset_dir, "plan_info.csv")
    query_info_path = os.path.join(dataset_dir, "query_info.csv")
    
    return {
        'dir': dataset_dir,
        'plan_info': plan_info_path,
        'query_info': query_info_path
    }

# ==================== 默认配置 ====================
DEFAULT_CONFIG = {
    'dataset': 'tpcds',
    'train_mode': 'estimated_train',
    'use_estimates_mode': True,
    'base_dop': 64,
    'min_improvement_ratio': 0.2,
    'min_reduction_threshold': 200,
    'target_dop': 96,
    'train_ratio': 1.0,
    'total_queries': 500
}

# ==================== 验证函数 ====================
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
