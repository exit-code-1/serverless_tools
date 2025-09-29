# -*- coding: utf-8 -*-
"""
统一推理入口
整合所有推理方法
"""

import argparse
import sys
import os

# 导入配置和工具
from config import DATASETS, TRAIN_MODES, DEFAULT_CONFIG
from utils import (
    setup_environment, setup_config_structure, validate_experiment_config,
    log_experiment_start, log_experiment_end, safe_import, Timer,
    get_dataset_paths, get_output_paths, load_csv_safe
)

def run_inference(dataset: str, train_mode: str, use_estimates_mode: bool = True):
    """运行算子级别推理"""
    print(f"开始算子级别推理...")
    
    # 获取数据路径
    test_paths = get_dataset_paths(dataset, 'test')
    
    # 获取输出路径
    output_paths = get_output_paths(dataset, 'inference', train_mode)
    
    # 获取模型路径
    from utils import get_model_paths
    model_paths = get_model_paths(train_mode, 'dop_aware')  # 使用DOP感知模型
    
    # 导入推理函数
    inference_func = safe_import('inference.predict_queries', 'run_inference')
    if inference_func is None:
        return False
    
    # 执行推理
    with Timer("算子级别推理"):
        inference_func(
            plan_csv_path=test_paths['plan_info'],
            query_csv_path=test_paths['query_info'],
            use_estimates=use_estimates_mode,
            output_csv_path=os.path.join(output_paths['prediction_dir'], 
                                       f"operator_level_{train_mode}_inference_results.csv"),
            no_dop_model_dir=os.path.join(model_paths['model_dir'], '..', 'operator_non_dop_aware'),
            dop_model_dir=model_paths['model_dir']
        )
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一推理入口')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='数据集名称')
    parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                       choices=list(TRAIN_MODES.keys()),
                       help='训练模式')
    parser.add_argument('--use_estimates', action='store_true', 
                       default=DEFAULT_CONFIG['use_estimates_mode'],
                       help='是否使用估计值')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    setup_config_structure()
    
    # 验证配置
    if not validate_experiment_config(args.dataset, 'dop_aware', args.train_mode):
        sys.exit(1)
    
    # 记录实验开始
    eval_mode = 'estimated' if args.use_estimates else 'exact'
    log_experiment_start(args.dataset, 'inference', args.train_mode, eval_mode)
    
    # 执行推理
    success = run_inference(args.dataset, args.train_mode, args.use_estimates)
    
    # 记录实验结束
    if success:
        log_experiment_end(args.dataset, 'inference')
        print("推理完成！")
    else:
        print("推理失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
