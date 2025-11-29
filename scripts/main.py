# -*- coding: utf-8 -*-
"""
主控制脚本
通过修改文件中的变量来控制运行参数，无需命令行参数
"""

import sys
import os
from tkinter import TRUE
import torch

# Add parent directory to Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置和工具
from config.main_config import DATASETS, METHODS, TRAIN_MODES, OPTIMIZATION_ALGORITHMS, DEFAULT_CONFIG
from utils import setup_environment, validate_experiment_config

def main():
    """主函数 - 通过修改下面的变量来控制运行参数"""
    
    # ==================== 配置区域 ====================
    # 在这里修改参数来控制运行什么功能
    
    # 基础配置
    DATASET = 'tpcds'  # 测试数据集: 'tpch' 或 'tpcds'
    MODEL_DATASET = 'tpch'  # 训练模型数据集: 'tpch' 或 'tpcds' (用于指定加载哪个数据集的模型)
    TRAIN_MODE = 'exact_train'  # 训练模式: 'exact_train' 或 'estimated_train'
    USE_ESTIMATES_MODE = False  # 是否使用估计值
    
    # 训练配置
    TRAIN_METHOD = 'ppm'  # 训练方法: 'dop_aware', 'non_dop_aware', 'ppm', 'query_level', 'mci'
    PPM_TYPE = 'NN'  # PPM类型: 'GNN' 或 'NN' (仅当TRAIN_METHOD='ppm'时有效)
    TOTAL_QUERIES = 500  # 总查询数量
    TRAIN_RATIO = 1.0  # 训练比例
    N_TRIALS = 30  # XGBoost优化试验次数 (仅当TRAIN_METHOD='query_level'时有效)
    
    # MCI配置
    MCI_CONFIG_FILE = 'mci_config_small.json'  # MCI配置文件路径
    # 优化配置
    OPTIMIZATION_ALGORITHM = 'ppm'  # 优化算法: 'pipeline', 'query_level', 'auto_dop', 'ppm', 'mci', 'base'
    BASE_DOP = 64  # 基准DOP
    MIN_IMPROVEMENT_RATIO = 0.1  # 最小改进比例
    MIN_REDUCTION_THRESHOLD = 200  # 最小减少阈值
    
    # MOO优化配置 (仅当OPTIMIZATION_ALGORITHM='pipeline'时有效)
    USE_MOO = True  # 是否使用多目标优化(MOO)算法
    USE_CONTINUOUS_DOP = True  # MOO是否在连续区间搜索DOP (True=区间搜索, False=候选列表搜索)
    MOO_POPULATION_SIZE = 30  # MOO种群大小
    MOO_GENERATIONS = 20  # MOO进化代数
    MOO_WEIGHT_LATENCY = 0.7  # Latency权重 (注: 流速匹配通过候选区间体现,不作为目标)
    MOO_WEIGHT_COST = 0.3  # Cost权重
    INTERVAL_TOLERANCE = 0.3  # 流速匹配区间容差(±30%) - 用于choose_optimal_dop
    
    # 运行控制 - 设置要运行的功能 (True/False)
    RUN_TRAIN =  False  # 是否运行训练
    RUN_INFERENCE = False  # 是否运行推理
    RUN_OPTIMIZE = False  # 是否运行优化
    RUN_EVALUATE = False  # 是否运行评估
    RUN_COMPARE = True  # 是否运行对比分析
    RUN_COMPARE_INFERENCE_METHODS = False  # 是否运行推理方法比较
    # =======================================================
    
    # 设置环境
    setup_environment()
    
    print("=" * 60)
    print("Serverless Predictor 主控制脚本")
    print("=" * 60)
    print(f"测试数据集: {DATASET}")
    print(f"模型数据集: {MODEL_DATASET}")
    print(f"训练模式: {TRAIN_MODE}")
    print(f"使用估计值: {USE_ESTIMATES_MODE}")
    print("=" * 60)
    
    # 运行训练
    if RUN_TRAIN:
        print(f"\n🚀 开始训练: {TRAIN_METHOD}")
        # 验证配置
        if not validate_experiment_config(DATASET, TRAIN_METHOD, TRAIN_MODE):
            print("❌ 训练配置验证失败")
            return
        
        # 调用训练函数
        if TRAIN_METHOD == 'dop_aware':
            from train import train_dop_aware_models
            success = train_dop_aware_models(DATASET, TRAIN_MODE, 
                                           total_queries=TOTAL_QUERIES,
                                           train_ratio=TRAIN_RATIO)
        elif TRAIN_METHOD == 'non_dop_aware':
            from train import train_non_dop_aware_models
            success = train_non_dop_aware_models(DATASET, TRAIN_MODE,
                                               total_queries=TOTAL_QUERIES,
                                               train_ratio=TRAIN_RATIO)
        elif TRAIN_METHOD == 'ppm':
            from train import train_ppm_models
            success = train_ppm_models(DATASET, TRAIN_MODE, PPM_TYPE)
        elif TRAIN_METHOD == 'query_level':
            from train import train_query_level_models
            success = train_query_level_models(DATASET, TRAIN_MODE, n_trials=N_TRIALS)
        elif TRAIN_METHOD == 'mci':
            success = run_mci_training(DATASET, MCI_CONFIG_FILE)
        
        if success:
            print("✅ 训练完成")
        else:
            print("❌ 训练失败")
            return
    
    # 运行推理
    if RUN_INFERENCE:
        print(f"\n🔍 开始推理: {TRAIN_METHOD}")
        
        # 调用推理函数
        if TRAIN_METHOD == 'dop_aware' or TRAIN_METHOD == 'non_dop_aware':
            # 验证配置
            if not validate_experiment_config(DATASET, TRAIN_METHOD, TRAIN_MODE):
                print("❌ 推理配置验证失败")
                return
            from inference import run_inference
            success = run_inference(DATASET, TRAIN_MODE, USE_ESTIMATES_MODE)
        elif TRAIN_METHOD == 'ppm':
            # 验证配置
            if not validate_experiment_config(DATASET, 'ppm', TRAIN_MODE):
                print("❌ 推理配置验证失败")
                return
            from inference import run_ppm_inference
            success = run_ppm_inference(DATASET, TRAIN_MODE, PPM_TYPE)
        elif TRAIN_METHOD == 'query_level':
            # 验证配置
            if not validate_experiment_config(DATASET, 'query_level', TRAIN_MODE):
                print("❌ 推理配置验证失败")
                return
            from inference import run_query_level_inference
            success = run_query_level_inference(DATASET, TRAIN_MODE)
        elif TRAIN_METHOD == 'mci':
            success = run_mci_inference(DATASET, MCI_CONFIG_FILE)
        else:
            print(f"❌ 未知的训练方法: {TRAIN_METHOD}")
            return
        
        if success:
            print("✅ 推理完成")
        else:
            print("❌ 推理失败")
            return
    
    # 运行优化
    if RUN_OPTIMIZE:
        print(f"\n⚡ 开始优化: {OPTIMIZATION_ALGORITHM}")
        
        # 调用优化函数 - based on OPTIMIZATION_ALGORITHM, not TRAIN_METHOD
        if OPTIMIZATION_ALGORITHM == 'base':
            # Baseline: all parallel operators use BASE_DOP (default 64)
            from optimize import run_baseline_optimization
            success = run_baseline_optimization(
                DATASET, TRAIN_MODE,
                base_dop=BASE_DOP,
                use_estimates=USE_ESTIMATES_MODE,
                model_dataset=MODEL_DATASET
            )
        elif OPTIMIZATION_ALGORITHM == 'mci':
            success = run_mci_optimization(DATASET, MCI_CONFIG_FILE)
        elif OPTIMIZATION_ALGORITHM == 'pipeline':
            # Pipeline: operator-level DOP optimization
            # Validate that operator models exist (trained with dop_aware method)
            if not validate_experiment_config(MODEL_DATASET, 'dop_aware', TRAIN_MODE):
                print(f"❌ 优化配置验证失败 - 需要先训练 {MODEL_DATASET} 数据集的 dop_aware 模型")
                return
            from optimize import run_pipeline_optimization
            success = run_pipeline_optimization(
                DATASET, TRAIN_MODE,
                base_dop=BASE_DOP,
                min_improvement_ratio=MIN_IMPROVEMENT_RATIO,
                min_reduction_threshold=MIN_REDUCTION_THRESHOLD,
                use_estimates=USE_ESTIMATES_MODE,
                interval_tolerance=INTERVAL_TOLERANCE,
                use_moo=USE_MOO,
                use_continuous_dop=USE_CONTINUOUS_DOP,
                moo_population_size=MOO_POPULATION_SIZE,
                moo_generations=MOO_GENERATIONS,
                model_dataset=MODEL_DATASET  # Pass model dataset separately
            )
        elif OPTIMIZATION_ALGORITHM in ['query_level', 'auto_dop', 'ppm']:
            # Query-level: uniform DOP for entire query
            # Validate that operator models exist (needed for performance prediction)
            if not validate_experiment_config(MODEL_DATASET, 'dop_aware', TRAIN_MODE):
                print(f"❌ 优化配置验证失败 - 需要先训练 {MODEL_DATASET} 数据集的 dop_aware 模型")
                return
            from optimize import run_query_level_optimization
            success = run_query_level_optimization(
                DATASET, OPTIMIZATION_ALGORITHM, TRAIN_MODE,
                base_dop=BASE_DOP,
                use_estimates=USE_ESTIMATES_MODE,
                model_dataset=MODEL_DATASET  # Pass model dataset separately
            )
        else:
            print(f"❌ 未知的优化算法: {OPTIMIZATION_ALGORITHM}")
            return
        
        if success:
            print("✅ 优化完成")
        else:
            print("❌ 优化失败")
            return
    
    # 运行评估
    if RUN_EVALUATE:
        print(f"\n📊 开始评估")
        # 调用评估函数
        from evaluate import evaluate_predictions
        success = evaluate_predictions(DATASET, TRAIN_MODE)
        
        if success:
            print("✅ 评估完成")
        else:
            print("❌ 评估失败")
            return
    
    # 运行对比分析
    if RUN_COMPARE:
        print(f"\n📈 开始对比分析")
        # 调用对比函数
        from compare import run_comparison
        success = run_comparison(DATASET)
        
        if success:
            print("✅ 对比分析完成")
        else:
            print("❌ 对比分析失败")
            return
    
    # 运行推理方法比较
    if RUN_COMPARE_INFERENCE_METHODS:
        print(f"\n🔍 开始推理方法比较")
        # 调用推理方法比较函数
        from compare_inference_methods import compare_inference_methods
        success = compare_inference_methods(DATASET, TRAIN_MODE, USE_ESTIMATES_MODE, MODEL_DATASET)
        
        if success:
            print("✅ 推理方法比较完成")
        else:
            print("❌ 推理方法比较失败")
            return
    
    print("\n" + "=" * 60)
    print("🎉 所有任务完成！")
    print("=" * 60)


# ==================== MCI 功能函数 ====================

def _load_mci_modules(mci_path: str, module_names: list):
    """动态加载MCI模块"""
    import importlib.util
    
    modules = {}
    for module_name in module_names:
        module_path = os.path.join(mci_path, f'{module_name}.py')
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[module_name] = module
    
    return modules


def _setup_mci_config(dataset: str, config_file: str, mci_path: str, create_config_file, MCIConfig):
    """设置MCI配置"""
    # 创建配置文件（如果不存在）
    config_file_path = os.path.join(mci_path, config_file)
    if not os.path.exists(config_file_path):
        print(f"Creating MCI config file: {config_file_path}")
        create_config_file(config_file_path, "small")
    
    # 加载配置
    config = MCIConfig.load_config(config_file_path)
    
    # Override dataset from parameter
    config.data.dataset = dataset
    print(f"Using dataset: {dataset}")
    
    # 更新数据集路径
    from utils import create_dataset_loader
    loader = create_dataset_loader(dataset)
    train_paths = loader.get_file_paths('train')
    test_paths = loader.get_file_paths('test')
    
    config.data.train_csv = train_paths['plan_info']
    config.data.train_info_csv = train_paths['query_info']
    config.data.test_csv = test_paths['plan_info']
    config.data.test_info_csv = test_paths['query_info']
    
    return config


def run_mci_training(dataset: str, config_file: str) -> bool:
    """运行MCI训练 - 调用标准训练流程"""
    print("Starting MCI model training...")
    
    # 添加MCI模块路径
    mci_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'MCI')
    sys.path.insert(0, mci_path)
    
    # 加载MCI模块
    modules = _load_mci_modules(mci_path, ['mci_train', 'mci_model', 'mci_data_loader'])
    
    # 导入MCI配置
    from config.mci_config import MCIConfig
    
    # 设置配置
    config = _setup_mci_config(dataset, config_file, mci_path, None, MCIConfig)
    
    # Setup device
    if config.training.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.training.device
    
    print(f"Using device: {device}")
    print(f"Output directory: {config.data.output_dir}")
    
    # Record start time
    import time
    start_time = time.time()
    
    # Train execution time model
    print("\n" + "="*60)
    print("Starting Execution Time Model Training...")
    print("="*60)
    exec_start = time.time()
    exec_results = modules['mci_train'].train_mci_execution_model(config, device)
    exec_time = time.time() - exec_start
    print(f"Execution time model training completed in {exec_time:.2f}s")
    
    # Train memory model
    print("\n" + "="*60)
    print("Starting Memory Model Training...")
    print("="*60)
    mem_start = time.time()
    mem_results = modules['mci_train'].train_mci_memory_model(config, device)
    mem_time = time.time() - mem_start
    print(f"Memory model training completed in {mem_time:.2f}s")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save results for both models
    exec_model_name = f"{config.data.model_name}_exec"
    mem_model_name = f"{config.data.model_name}_mem"
    
    modules['mci_train'].save_model_and_results(
        exec_results['model'], 
        exec_results, 
        config.data.output_dir, 
        exec_model_name,
        config
    )
    modules['mci_train'].save_model_and_results(
        mem_results['model'], 
        mem_results, 
        config.data.output_dir, 
        mem_model_name,
        config
    )
    
    print("\n" + "=" * 60)
    print("✅ MCI训练完成")
    print("=" * 60)
    print("Execution time model saved as PyTorch model")
    print("Memory model saved as PyTorch model")
    print()
    print("Training Time Summary:")
    print(f"  Execution model: {exec_time:.2f}s ({exec_time/60:.2f} min)")
    print(f"  Memory model: {mem_time:.2f}s ({mem_time/60:.2f} min)")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 60)
    
    return True


def run_mci_inference(dataset: str, config_file: str) -> bool:
    """运行MCI推理"""
    try:
        print("Starting MCI model inference...")
        
        # 添加MCI模块路径
        mci_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'MCI')
        sys.path.insert(0, mci_path)
        
        # 导入MCI配置
        from config.mci_config import MCIConfig
        
        # 加载配置
        config_path = os.path.join(mci_path, config_file)
        print(f"Loading configuration from: {config_path}")
        config = MCIConfig.load_config(config_path)
        
        # Override dataset from parameter
        config.data.dataset = dataset
        print(f"Using dataset: {dataset}")
        
        # Setup device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # 导入推理模块 (动态导入，运行时可用)
        from mci_inference import load_pytorch_model, predict_with_pytorch_model  # type: ignore
        from mci_data_loader import create_mci_data_loaders_exec, create_mci_data_loaders_mem  # type: ignore
        
        # 确定模型路径
        exec_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_exec.pth")
        mem_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_mem.pth")
        
        print(f"Execution time model: {exec_model_path}")
        print(f"Memory model: {mem_model_path}")
        
        # 检查模型是否存在
        if not os.path.exists(exec_model_path):
            print(f"❌ Execution time model not found: {exec_model_path}")
            return False
        if not os.path.exists(mem_model_path):
            print(f"❌ Memory model not found: {mem_model_path}")
            return False
        
        # 加载PyTorch模型
        print("Loading PyTorch models...")
        exec_model = load_pytorch_model(exec_model_path, config, device)
        mem_model = load_pytorch_model(mem_model_path, config, device)
        
        # 加载测试数据 - 使用全部数据进行推理
        print("Loading test data...")
        from mci_data_loader import load_mci_pipeline_data_exec, load_mci_pipeline_data_mem  # type: ignore
        from torch_geometric.loader import DataLoader as PyGDataLoader
        
        # Load all test data without train/val split
        # Use target_dop_levels=None to load ALL DOPs in test data (not just training DOPs)
        all_test_data_exec = load_mci_pipeline_data_exec(
            csv_file=config.get_test_csv_path(),
            query_info_file=config.get_test_info_csv_path(),
            target_dop_levels=None,  # Load all DOPs in test data
            use_estimates=config.training.use_estimates
        )
        
        all_test_data_mem = load_mci_pipeline_data_mem(
            csv_file=config.get_test_csv_path(),
            query_info_file=config.get_test_info_csv_path(),
            target_dop_levels=None,  # Load all DOPs in test data
            use_estimates=config.training.use_estimates
        )
        
        print(f"Loaded {len(all_test_data_exec)} execution time pipelines")
        print(f"Loaded {len(all_test_data_mem)} memory pipelines")
        
        # Create data loaders with batch_size=1 for pipeline-level inference
        # This allows precise tracking of each pipeline's prediction
        test_loader_exec = PyGDataLoader(
            all_test_data_exec,
            batch_size=1,  # Process one pipeline at a time
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        test_loader_mem = PyGDataLoader(
            all_test_data_mem,
            batch_size=1,  # Process one pipeline at a time
            shuffle=False,
            num_workers=0
        )
        
        # 运行推理
        print("Running execution time inference...")
        exec_results = predict_with_pytorch_model(exec_model, test_loader_exec, device)
        
        print("Running memory inference...")
        mem_results = predict_with_pytorch_model(mem_model, test_loader_mem, device)
        
        # 打印结果
        print("=" * 60)
        print("MCI Model Inference Results")
        print("=" * 60)
        print(f"Execution Time Model:")
        print(f"  MSE: {exec_results['mse']:.6f}")
        print(f"  MAE: {exec_results['mae']:.6f}")
        print(f"  Q-error (mean): {exec_results['q_error_mean']:.6f}")
        print(f"  Q-error (median): {exec_results['q_error_median']:.6f}")
        print()
        print(f"Memory Model:")
        print(f"  MSE: {mem_results['mse']:.6f}")
        print(f"  MAE: {mem_results['mae']:.6f}")
        print(f"  Q-error (mean): {mem_results['q_error_mean']:.6f}")
        print(f"  Q-error (median): {mem_results['q_error_median']:.6f}")
        print("=" * 60)
        
        # Save results to CSV format
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        
        # Get pipeline-level predictions and inference times
        exec_predictions = exec_results['predictions'].tolist() if hasattr(exec_results['predictions'], 'tolist') else exec_results['predictions']
        exec_targets = exec_results['targets'].tolist() if hasattr(exec_results['targets'], 'tolist') else exec_results['targets']
        exec_inference_times = exec_results['inference_times'].tolist() if hasattr(exec_results['inference_times'], 'tolist') else exec_results['inference_times']
        
        mem_predictions = mem_results['predictions'].tolist() if hasattr(mem_results['predictions'], 'tolist') else mem_results['predictions']
        mem_targets = mem_results['targets'].tolist() if hasattr(mem_results['targets'], 'tolist') else mem_results['targets']
        mem_inference_times = mem_results['inference_times'].tolist() if hasattr(mem_results['inference_times'], 'tolist') else mem_results['inference_times']
        
        # Extract query_id and query DOP for each pipeline from the test data
        # Use query_dop (query's original DOP) not dop_level (pipeline's DOP which may be 1)
        pipeline_info = []
        for data in all_test_data_exec:
            query_id = data.pipeline_metadata.get('query_id', 'unknown')
            query_dop = data.pipeline_metadata.get('query_dop', data.dop_level.item())  # Use query's original DOP
            pipeline_info.append((query_id, query_dop))
        
        print(f"\nAggregating {len(pipeline_info)} pipelines to query level...")
        
        # Diagnose NaN predictions - find which queries/pipelines have NaN
        nan_exec_indices = [i for i, pred in enumerate(exec_predictions) if np.isnan(pred)]
        nan_mem_indices = [i for i, pred in enumerate(mem_predictions) if np.isnan(pred)]
        
        if len(nan_exec_indices) > 0 or len(nan_mem_indices) > 0:
            print(f"\n⚠️  检测到 NaN 预测值，定位到具体 query:")
            
            if len(nan_exec_indices) > 0:
                print(f"\n⚠️  执行时间模型 - {len(nan_exec_indices)} 个 NaN:")
                nan_queries_exec = {}
                for idx in nan_exec_indices:
                    qid, dop = pipeline_info[idx]
                    key = (qid, dop)
                    if key not in nan_queries_exec:
                        nan_queries_exec[key] = 0
                    nan_queries_exec[key] += 1
                
                print(f"    涉及的 (query_id, DOP) 组合:")
                for (qid, dop), count in sorted(nan_queries_exec.items())[:20]:
                    print(f"      - Query {qid}, DOP={dop}: {count} 个 pipeline")
            
            if len(nan_mem_indices) > 0:
                print(f"\n⚠️  内存模型 - {len(nan_mem_indices)} 个 NaN:")
                nan_queries_mem = {}
                for idx in nan_mem_indices:
                    qid, dop = pipeline_info[idx]
                    key = (qid, dop)
                    if key not in nan_queries_mem:
                        nan_queries_mem[key] = 0
                    nan_queries_mem[key] += 1
                
                print(f"    涉及的 (query_id, DOP) 组合:")
                for (qid, dop), count in sorted(nan_queries_mem.items())[:20]:
                    print(f"      - Query {qid}, DOP={dop}: {count} 个 pipeline")
            
            print()
        
        # Group by (query_id, DOP) and sum predictions/targets/inference_times
        query_aggregated = defaultdict(lambda: {
            'exec_pred': 0.0, 'exec_target': 0.0,
            'mem_pred': 0.0, 'mem_target': 0.0,
            'exec_inference_time': 0.0,
            'mem_inference_time': 0.0,
            'pipeline_count': 0
        })
        
        for i, (query_id, dop) in enumerate(pipeline_info):
            key = (query_id, dop)
            query_aggregated[key]['exec_pred'] += exec_predictions[i]
            query_aggregated[key]['exec_target'] += exec_targets[i]
            query_aggregated[key]['mem_pred'] += mem_predictions[i]
            query_aggregated[key]['mem_target'] += mem_targets[i]
            query_aggregated[key]['exec_inference_time'] += exec_inference_times[i]
            query_aggregated[key]['mem_inference_time'] += mem_inference_times[i]
            query_aggregated[key]['pipeline_count'] += 1
        
        # Calculate Q-errors for each query
        def calc_q_error(actual, pred):
            # Check for nan or invalid values
            if np.isnan(actual) or np.isnan(pred):
                return np.nan
            if actual == 0 or pred == 0:
                return float('inf')
            return max(pred / actual, actual / pred) - 1
        
        # Create DataFrame with query-level results
        rows = []
        for (query_id, dop), agg_data in sorted(query_aggregated.items()):
            exec_q_error = calc_q_error(agg_data['exec_target'], agg_data['exec_pred'])
            mem_q_error = calc_q_error(agg_data['mem_target'], agg_data['mem_pred'])
            
            # Check if this query has NaN predictions
            has_nan_exec = np.isnan(agg_data['exec_pred'])
            has_nan_mem = np.isnan(agg_data['mem_pred'])
            
            rows.append({
                'query_id': query_id,
                'dop': dop,
                'actual_time': agg_data['exec_target'],
                'predicted_time': agg_data['exec_pred'],
                'Execution Time Q-error': exec_q_error,
                'actual_memory': agg_data['mem_target'],
                'predicted_memory': agg_data['mem_pred'],
                'Memory Q-error': mem_q_error,
                'pipeline_count': agg_data['pipeline_count'],
                'exec_inference_time': agg_data['exec_inference_time'],
                'mem_inference_time': agg_data['mem_inference_time'],
                'total_inference_time': agg_data['exec_inference_time'] + agg_data['mem_inference_time'],
                'has_nan_exec': has_nan_exec,
                'has_nan_mem': has_nan_mem
            })
        
        df = pd.DataFrame(rows)
        
        print(f"Aggregated to {len(df)} query-level results (from {len(pipeline_info)} pipelines)")
        
        # Report NaN statistics at query level
        nan_exec_queries = df['has_nan_exec'].sum()
        nan_mem_queries = df['has_nan_mem'].sum()
        if nan_exec_queries > 0 or nan_mem_queries > 0:
            print(f"\n⚠️  Query 级别 NaN 统计:")
            print(f"    - 执行时间 NaN 的 query 数: {nan_exec_queries}/{len(df)} ({nan_exec_queries/len(df)*100:.1f}%)")
            print(f"    - 内存 NaN 的 query 数: {nan_mem_queries}/{len(df)} ({nan_mem_queries/len(df)*100:.1f}%)")
            
            # Show which queries have NaN
            if nan_exec_queries > 0:
                nan_exec_list = df[df['has_nan_exec']]['query_id'].unique()
                print(f"    - 执行时间 NaN 的 query_id: {sorted(nan_exec_list)[:20]}")
            if nan_mem_queries > 0:
                nan_mem_list = df[df['has_nan_mem']]['query_id'].unique()
                print(f"    - 内存 NaN 的 query_id: {sorted(nan_mem_list)[:20]}")
        
        # Check for nan predictions
        nan_exec_count = df['predicted_time'].isna().sum()
        nan_mem_count = df['predicted_memory'].isna().sum()
        if nan_exec_count > 0 or nan_mem_count > 0:
            print("\n" + "="*60)
            print("WARNING: Some predictions contain NaN values")
            print(f"  Execution time NaN count: {nan_exec_count}/{len(df)}")
            print(f"  Memory NaN count: {nan_mem_count}/{len(df)}")
            print("  This may indicate feature mismatch between training and test data")
            print("  (e.g., using tpch-trained model on tpcds data)")
            print("="*60 + "\n")
        
        # Save to CSV with semicolon separator
        results_path = os.path.join(config.data.output_dir, "mci_inference_results.csv")
        df.to_csv(results_path, index=False, sep=';')
        
        print(f"Results saved to: {results_path}")
        print("✅ MCI推理完成")
        return True
        
    except Exception as e:
        print(f"❌ MCI推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_mci_optimization(dataset: str, config_file: str) -> bool:
    """运行MCI优化"""
    try:
        print("Starting MCI MOO optimization...")
        
        # 添加MCI模块路径
        mci_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'MCI')
        sys.path.insert(0, mci_path)
        
        # 加载MCI模块
        modules = _load_mci_modules(mci_path, ['mci_optimize'])
        
        # 导入MCI配置
        from config.mci_config import MCIConfig, create_config_file, PresetConfigs
        mci_config = type('MCIConfigModule', (), {
            'MCIConfig': MCIConfig,
            'create_config_file': create_config_file,
            'PresetConfigs': PresetConfigs
        })()
        
        # 设置配置
        config = _setup_mci_config(dataset, config_file, mci_path, create_config_file, MCIConfig)
        
        # 构建PyTorch模型路径 (使用执行时间模型)
        pytorch_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_exec.pth")
        
        if not os.path.exists(pytorch_model_path):
            print(f"❌ PyTorch模型文件不存在: {pytorch_model_path}")
            print("请先运行训练以生成模型文件")
            return False
        
        # 运行优化
        results = modules['mci_optimize'].run_moo_optimization(
            config=config,
            pytorch_model_path=pytorch_model_path,
            output_dir=config.data.output_dir,
            population_size=10,
            generations=6,
            weight_latency=0.9,
            weight_cost=0.1
        )
        
        print("✅ MCI优化完成")
        return True
        
    except Exception as e:
        print(f"❌ MCI优化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
