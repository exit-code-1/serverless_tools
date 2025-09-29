# -*- coding: utf-8 -*-
"""
统一训练入口
整合所有训练方法，通过参数控制使用哪种方法
"""

import argparse
import sys
import os

# 导入配置和工具
from config import DATASETS, METHODS, TRAIN_MODES, DEFAULT_CONFIG
from utils import (
    setup_environment, setup_config_structure, validate_experiment_config,
    log_experiment_start, log_experiment_end, safe_import, Timer,
    get_dataset_paths, get_output_paths, load_csv_safe
)

def train_dop_aware_models(dataset: str, train_mode: str, **kwargs):
    """训练DOP感知算子模型"""
    print(f"开始训练DOP感知算子模型...")
    
    # 获取数据路径
    train_paths = get_dataset_paths(dataset, 'train')
    test_paths = get_dataset_paths(dataset, 'test')
    
    # 加载数据
    train_data = load_csv_safe(train_paths['plan_info'], description="训练数据")
    test_data = load_csv_safe(test_paths['plan_info'], description="测试数据")
    
    if train_data is None or test_data is None:
        return False
    
    # 导入训练函数
    train_func = safe_import('training.operator_dop_aware.train', 'train_all_operators')
    if train_func is None:
        return False
    
    # 执行训练
    with Timer("DOP感知模型训练"):
        train_func(
            train_data=train_data,
            test_data=test_data,
            total_queries=kwargs.get('total_queries', DEFAULT_CONFIG['total_queries']),
            train_ratio=kwargs.get('train_ratio', DEFAULT_CONFIG['train_ratio']),
            use_estimates=TRAIN_MODES[train_mode]['use_estimates']
        )
    
    return True

def train_non_dop_aware_models(dataset: str, train_mode: str, **kwargs):
    """训练非DOP感知算子模型"""
    print(f"开始训练非DOP感知算子模型...")
    
    # 获取数据路径
    train_paths = get_dataset_paths(dataset, 'train')
    test_paths = get_dataset_paths(dataset, 'test')
    
    # 加载数据
    train_data = load_csv_safe(train_paths['plan_info'], description="训练数据")
    test_data = load_csv_safe(test_paths['plan_info'], description="测试数据")
    
    if train_data is None or test_data is None:
        return False
    
    # 导入训练函数
    train_func = safe_import('training.operator_non_dop_aware.train', 'train_all_operators')
    if train_func is None:
        return False
    
    # 执行训练
    with Timer("非DOP感知模型训练"):
        train_func(
            train_data=train_data,
            test_data=test_data,
            total_queries=kwargs.get('total_queries', DEFAULT_CONFIG['total_queries']),
            train_ratio=kwargs.get('train_ratio', DEFAULT_CONFIG['train_ratio']),
            use_estimates=TRAIN_MODES[train_mode]['use_estimates']
        )
    
    return True

def train_ppm_models(dataset: str, train_mode: str, method_type: str = 'GNN', **kwargs):
    """训练PPM模型"""
    print(f"开始训练PPM模型 ({method_type})...")
    
    # 根据方法类型选择训练模块
    if method_type == 'GNN':
        train_func = safe_import('training.PPM.GNN_train', 'main')
    elif method_type == 'NN':
        train_func = safe_import('training.PPM.NN_train', 'main')
    else:
        print(f"错误: 未知的PPM方法类型: {method_type}")
        return False
    
    if train_func is None:
        return False
    
    # 执行训练
    with Timer(f"PPM模型训练 ({method_type})"):
        # PPM训练函数可能需要不同的参数
        # 这里需要根据实际的PPM训练函数调整
        train_func()
    
    return True

def train_query_level_models(dataset: str, train_mode: str, **kwargs):
    """训练查询级别模型"""
    print(f"开始训练查询级别模型...")
    
    # 获取数据路径
    train_paths = get_dataset_paths(dataset, 'train')
    test_paths = get_dataset_paths(dataset, 'test')
    
    # 获取输出路径
    output_paths = get_output_paths(dataset, 'query_level', train_mode)
    
    # 导入训练函数
    train_func = safe_import('training.query_level.train', 'train_and_save_xgboost_onnx')
    test_func = safe_import('training.query_level.train', 'test_onnx_xgboost')
    qerror_func = safe_import('training.query_level.train', 'compute_qerror_by_bins')
    
    if train_func is None or test_func is None or qerror_func is None:
        return False
    
    # 执行训练
    with Timer("查询级别模型训练"):
        train_func(
            feature_csv=train_paths['plan_info'],
            true_val_csv=train_paths['query_info'],
            execution_onnx_path=os.path.join(output_paths['model_dir'], "execution_time_model.onnx"),
            memory_onnx_path=os.path.join(output_paths['model_dir'], "memory_usage_model.onnx"),
            n_trials=kwargs.get('n_trials', 30),
            use_estimates=TRAIN_MODES[train_mode]['use_estimates']
        )
    
    # 执行测试
    with Timer("查询级别模型测试"):
        results_df = test_func(
            execution_onnx_path=os.path.join(output_paths['model_dir'], "execution_time_model.onnx"),
            memory_onnx_path=os.path.join(output_paths['model_dir'], "memory_usage_model.onnx"),
            feature_csv=test_paths['plan_info'],
            true_val_csv=test_paths['query_info'],
            output_file=os.path.join(output_paths['prediction_dir'], f"query_level_predictions_{train_mode}.csv"),
            use_estimates=kwargs.get('use_estimates_mode', DEFAULT_CONFIG['use_estimates_mode'])
        )
    
    # 计算Q-error
    if results_df is not None:
        with Timer("Q-error计算"):
            qerror_func(
                results_df, 
                os.path.join(output_paths['evaluation_dir'], f"query_level_qerror_{train_mode}.csv"),
                kwargs.get('target_dop', DEFAULT_CONFIG['target_dop'])
            )
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一训练入口')
    parser.add_argument('--method', type=str, required=True, 
                       choices=list(METHODS.keys()),
                       help='训练方法: dop_aware, non_dop_aware, ppm, query_level')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='数据集名称')
    parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                       choices=list(TRAIN_MODES.keys()),
                       help='训练模式')
    parser.add_argument('--ppm_type', type=str, default='GNN',
                       choices=['GNN', 'NN'],
                       help='PPM方法类型 (仅当method=ppm时有效)')
    parser.add_argument('--total_queries', type=int, default=DEFAULT_CONFIG['total_queries'],
                       help='总查询数量')
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_CONFIG['train_ratio'],
                       help='训练比例')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='XGBoost优化试验次数 (仅当method=query_level时有效)')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    setup_config_structure()
    
    # 验证配置
    if not validate_experiment_config(args.dataset, args.method, args.train_mode):
        sys.exit(1)
    
    # 记录实验开始
    log_experiment_start(args.dataset, args.method, args.train_mode)
    
    # 执行训练
    success = False
    if args.method == 'dop_aware':
        success = train_dop_aware_models(args.dataset, args.train_mode, 
                                       total_queries=args.total_queries,
                                       train_ratio=args.train_ratio)
    elif args.method == 'non_dop_aware':
        success = train_non_dop_aware_models(args.dataset, args.train_mode,
                                          total_queries=args.total_queries,
                                          train_ratio=args.train_ratio)
    elif args.method == 'ppm':
        success = train_ppm_models(args.dataset, args.train_mode, args.ppm_type)
    elif args.method == 'query_level':
        success = train_query_level_models(args.dataset, args.train_mode,
                                         n_trials=args.n_trials)
    
    # 记录实验结束
    if success:
        log_experiment_end(args.dataset, args.method)
        print("训练完成！")
    else:
        print("训练失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
