# -*- coding: utf-8 -*-
"""
优化模块
提供统一的优化接口
"""

import os
import sys
import pandas as pd
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.optimizer import run_dop_optimization, run_query_dop_optimization
from utils import (
    get_output_paths, get_model_paths, create_dataset_loader, 
    log_experiment_start, log_experiment_end, Timer
)
from config import OPTIMIZATION_ALGORITHMS

def run_pipeline_optimization(dataset: str, algorithm: str, train_mode: str, 
                            base_dop: int = 64, min_improvement_ratio: float = 0.2,
                            min_reduction_threshold: int = 200, use_estimates: bool = False) -> bool:
    """
    运行Pipeline优化
    
    Args:
        dataset: 数据集名称
        algorithm: 优化算法 ('pipeline', 'auto_dop', 'ppm')
        train_mode: 训练模式
        base_dop: 基准DOP
        min_improvement_ratio: 最小改进比例
        min_reduction_threshold: 最小减少阈值
        use_estimates: 是否使用估计值
        
    Returns:
        bool: 是否成功
    """
    try:
        log_experiment_start(dataset, f"Pipeline优化-{algorithm}", train_mode)
        
        # 使用统一的数据加载器
        loader = create_dataset_loader(dataset)
        output_paths = get_output_paths(dataset, 'dop_aware', train_mode)
        
        # 加载数据
        df_plans_all_dops = loader.load_test_data(use_estimates)  # 使用测试数据作为优化输入
        if df_plans_all_dops is None:
            return False
        
        # 设置输出路径
        output_json_path = os.path.join(output_paths['optimization_dir'], f"pipeline_optimization_{algorithm}.json")
        
        # 导入ONNX模型管理器
        from core.onnx_manager import ONNXModelManager
        
        # 初始化模型管理器
        model_paths = get_model_paths(train_mode, 'dop_aware')
        onnx_manager = ONNXModelManager(
            no_dop_model_dir=os.path.join(model_paths['model_dir'], 'non_dop_aware'),
            dop_model_dir=os.path.join(model_paths['model_dir'], 'dop_aware')
        )
        
        # 运行优化
        with Timer(f"Pipeline优化-{algorithm}"):
            if algorithm == 'pipeline':
                result = run_dop_optimization(
                    df_plans_all_dops, onnx_manager, output_json_path,
                    base_dop=base_dop, use_estimates=use_estimates,
                    min_improvement_ratio=min_improvement_ratio,
                    min_reduction_threshold=min_reduction_threshold
                )
            elif algorithm in ['auto_dop', 'ppm']:
                # 对于auto_dop和ppm，使用查询级别优化
                result = run_query_dop_optimization(
                    None, df_plans_all_dops, onnx_manager, output_json_path,
                    algorithm=algorithm, base_dop=base_dop
                )
            else:
                print(f"❌ 未知的优化算法: {algorithm}")
                return False
        
        if result is not None:
            print(f"✅ Pipeline优化-{algorithm} 完成")
            log_experiment_end(dataset, f"Pipeline优化-{algorithm}", output_json_path)
            return True
        else:
            print(f"❌ Pipeline优化-{algorithm} 失败")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline优化-{algorithm} 出错: {e}")
        return False

def run_query_level_optimization(dataset: str, algorithm: str, train_mode: str,
                               base_dop: int = 64) -> bool:
    """
    运行查询级别优化
    
    Args:
        dataset: 数据集名称
        algorithm: 优化算法 ('query_level')
        train_mode: 训练模式
        base_dop: 基准DOP
        
    Returns:
        bool: 是否成功
    """
    try:
        log_experiment_start(dataset, "查询级别优化", train_mode)
        
        # 使用统一的数据加载器
        loader = create_dataset_loader(dataset)
        output_paths = get_output_paths(dataset, 'query_level', train_mode)
        
        # 加载数据
        df_plans_all_dops = loader.load_test_data(use_estimates)  # 使用测试数据作为优化输入
        if df_plans_all_dops is None:
            return False
        
        # 设置输出路径
        output_json_path = os.path.join(output_paths['optimization_dir'], "query_level_optimization.json")
        
        # 导入ONNX模型管理器
        from core.onnx_manager import ONNXModelManager
        
        # 初始化模型管理器
        model_paths = get_model_paths(train_mode, 'query_level')
        onnx_manager = ONNXModelManager(
            no_dop_model_dir=os.path.join(model_paths['model_dir'], 'non_dop_aware'),
            dop_model_dir=os.path.join(model_paths['model_dir'], 'dop_aware')
        )
        
        # 运行优化
        with Timer("查询级别优化"):
            result = run_query_dop_optimization(
                None, df_plans_all_dops, onnx_manager, output_json_path,
                algorithm='query_level', base_dop=base_dop
            )
        
        if result is not None:
            print("✅ 查询级别优化完成")
            log_experiment_end(dataset, "查询级别优化", output_json_path)
            return True
        else:
            print("❌ 查询级别优化失败")
            return False
            
    except Exception as e:
        print(f"❌ 查询级别优化出错: {e}")
        return False
