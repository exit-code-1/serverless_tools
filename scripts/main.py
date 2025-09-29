# -*- coding: utf-8 -*-
"""
主控制脚本
提供统一的命令行接口来运行所有功能
"""

import argparse
import sys
import os

# 导入配置和工具
from config import DATASETS, METHODS, TRAIN_MODES, OPTIMIZATION_ALGORITHMS, DEFAULT_CONFIG
from utils import setup_environment, validate_experiment_config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Serverless Predictor 主控制脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练DOP感知模型
  python main.py train --method dop_aware --dataset tpcds --train_mode estimated_train
  
  # 训练PPM模型
  python main.py train --method ppm --dataset tpcds --ppm_type GNN
  
  # 运行推理
  python main.py inference --dataset tpcds --train_mode estimated_train
  
  # 运行优化
  python main.py optimize --algorithm pipeline --dataset tpcds
  
  # 运行评估
  python main.py evaluate --dataset tpcds
  
  # 运行对比分析
  python main.py compare --dataset tpcds
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练子命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--method', type=str, required=True,
                             choices=list(METHODS.keys()),
                             help='训练方法')
    train_parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                             choices=list(DATASETS.keys()),
                             help='数据集名称')
    train_parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                             choices=list(TRAIN_MODES.keys()),
                             help='训练模式')
    train_parser.add_argument('--ppm_type', type=str, default='GNN',
                             choices=['GNN', 'NN'],
                             help='PPM方法类型')
    train_parser.add_argument('--total_queries', type=int, default=DEFAULT_CONFIG['total_queries'],
                             help='总查询数量')
    train_parser.add_argument('--train_ratio', type=float, default=DEFAULT_CONFIG['train_ratio'],
                             help='训练比例')
    train_parser.add_argument('--n_trials', type=int, default=30,
                             help='XGBoost优化试验次数')
    
    # 推理子命令
    inference_parser = subparsers.add_parser('inference', help='运行推理')
    inference_parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                                 choices=list(DATASETS.keys()),
                                 help='数据集名称')
    inference_parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                                 choices=list(TRAIN_MODES.keys()),
                                 help='训练模式')
    inference_parser.add_argument('--use_estimates', action='store_true',
                                 default=DEFAULT_CONFIG['use_estimates_mode'],
                                 help='是否使用估计值')
    
    # 优化子命令
    optimize_parser = subparsers.add_parser('optimize', help='运行优化')
    optimize_parser.add_argument('--algorithm', type=str, required=True,
                                choices=list(OPTIMIZATION_ALGORITHMS.keys()),
                                help='优化算法')
    optimize_parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                                choices=list(DATASETS.keys()),
                                help='数据集名称')
    optimize_parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                                help='训练模式')
    optimize_parser.add_argument('--base_dop', type=int, default=DEFAULT_CONFIG['base_dop'],
                                help='基准DOP')
    optimize_parser.add_argument('--min_improvement_ratio', type=float,
                                default=DEFAULT_CONFIG['min_improvement_ratio'],
                                help='最小改进比例')
    optimize_parser.add_argument('--min_reduction_threshold', type=int,
                                default=DEFAULT_CONFIG['min_reduction_threshold'],
                                help='最小减少阈值')
    optimize_parser.add_argument('--use_estimates', action='store_true',
                                default=DEFAULT_CONFIG['use_estimates_mode'],
                                help='是否使用估计值')
    
    # 评估子命令
    evaluate_parser = subparsers.add_parser('evaluate', help='运行评估')
    evaluate_parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                               choices=list(DATASETS.keys()),
                               help='数据集名称')
    evaluate_parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                               help='训练模式')
    
    # 对比子命令
    compare_parser = subparsers.add_parser('compare', help='运行对比分析')
    compare_parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                               choices=list(DATASETS.keys()),
                               help='数据集名称')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 根据子命令执行相应功能
    if args.command == 'train':
        # 验证配置
        if not validate_experiment_config(args.dataset, args.method, args.train_mode):
            sys.exit(1)
        
        # 构建参数
        train_args = [
            '--method', args.method,
            '--dataset', args.dataset,
            '--train_mode', args.train_mode,
            '--total_queries', str(args.total_queries),
            '--train_ratio', str(args.train_ratio)
        ]
        
        if args.method == 'ppm':
            train_args.extend(['--ppm_type', args.ppm_type])
        elif args.method == 'query_level':
            train_args.extend(['--n_trials', str(args.n_trials)])
        
        # 调用训练脚本
        from train import main as train_main
        sys.argv = ['train.py'] + train_args
        train_main()
        
    elif args.command == 'inference':
        # 验证配置
        if not validate_experiment_config(args.dataset, 'dop_aware', args.train_mode):
            sys.exit(1)
        
        # 构建参数
        inference_args = [
            '--dataset', args.dataset,
            '--train_mode', args.train_mode
        ]
        
        if args.use_estimates:
            inference_args.append('--use_estimates')
        
        # 调用推理脚本
        from inference import main as inference_main
        sys.argv = ['inference.py'] + inference_args
        inference_main()
        
    elif args.command == 'optimize':
        # 验证配置
        if not validate_experiment_config(args.dataset, 'dop_aware', args.train_mode):
            sys.exit(1)
        
        # 构建参数
        optimize_args = [
            '--algorithm', args.algorithm,
            '--dataset', args.dataset,
            '--train_mode', args.train_mode,
            '--base_dop', str(args.base_dop),
            '--min_improvement_ratio', str(args.min_improvement_ratio),
            '--min_reduction_threshold', str(args.min_reduction_threshold)
        ]
        
        if args.use_estimates:
            optimize_args.append('--use_estimates')
        
        # 调用优化脚本
        from optimize import main as optimize_main
        sys.argv = ['optimize.py'] + optimize_args
        optimize_main()
        
    elif args.command == 'evaluate':
        # 构建参数
        evaluate_args = [
            '--dataset', args.dataset,
            '--train_mode', args.train_mode
        ]
        
        # 调用评估脚本
        from evaluate import main as evaluate_main
        sys.argv = ['evaluate.py'] + evaluate_args
        evaluate_main()
        
    elif args.command == 'compare':
        # 构建参数
        compare_args = ['--dataset', args.dataset]
        
        # 调用对比脚本
        from compare import main as compare_main
        sys.argv = ['compare.py'] + compare_args
        compare_main()

if __name__ == "__main__":
    main()
