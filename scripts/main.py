# -*- coding: utf-8 -*-
"""
主控制脚本
通过修改文件中的变量来控制运行参数，无需命令行参数
"""

import sys
import os
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
    DATASET = 'tpch'  # 数据集: 'tpch' 或 'tpcds'
    TRAIN_MODE = 'exact_train'  # 训练模式: 'exact_train' 或 'estimated_train'
    USE_ESTIMATES_MODE = False  # 是否使用估计值
    
    # 训练配置
    TRAIN_METHOD = 'mci'  # 训练方法: 'dop_aware', 'non_dop_aware', 'ppm', 'query_level', 'mci'
    PPM_TYPE = 'GNN'  # PPM类型: 'GNN' 或 'NN' (仅当TRAIN_METHOD='ppm'时有效)
    TOTAL_QUERIES = 500  # 总查询数量
    TRAIN_RATIO = 1.0  # 训练比例
    N_TRIALS = 30  # XGBoost优化试验次数 (仅当TRAIN_METHOD='query_level'时有效)
    
    # MCI配置
    MCI_CONFIG_FILE = 'mci_config_small.json'  # MCI配置文件路径
    
    # 优化配置
    OPTIMIZATION_ALGORITHM = 'mci'  # 优化算法: 'pipeline', 'query_level', 'auto_dop', 'ppm', 'mci'
    BASE_DOP = 64  # 基准DOP
    MIN_IMPROVEMENT_RATIO = 0.2  # 最小改进比例
    MIN_REDUCTION_THRESHOLD = 200  # 最小减少阈值
    
    # 运行控制 - 设置要运行的功能 (True/False)
    RUN_TRAIN = False  # 是否运行训练
    RUN_INFERENCE = True  # 是否运行推理
    RUN_OPTIMIZE = False  # 是否运行优化
    RUN_EVALUATE = False  # 是否运行评估
    RUN_COMPARE = False  # 是否运行对比分析
    # =======================================================
    
    # 设置环境
    setup_environment()
    
    print("=" * 60)
    print("Serverless Predictor 主控制脚本")
    print("=" * 60)
    print(f"数据集: {DATASET}")
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
        print(f"\n⚡ 开始优化: {TRAIN_METHOD} - {OPTIMIZATION_ALGORITHM}")
        
        # 调用优化函数
        if TRAIN_METHOD == 'mci':
            success = run_mci_optimization(DATASET, MCI_CONFIG_FILE)
        elif OPTIMIZATION_ALGORITHM in ['pipeline', 'auto_dop', 'ppm']:
            # 验证配置
            if not validate_experiment_config(DATASET, TRAIN_METHOD, TRAIN_MODE):
                print("❌ 优化配置验证失败")
                return
            from optimize import run_pipeline_optimization
            success = run_pipeline_optimization(
                DATASET, OPTIMIZATION_ALGORITHM, TRAIN_MODE,
                base_dop=BASE_DOP,
                min_improvement_ratio=MIN_IMPROVEMENT_RATIO,
                min_reduction_threshold=MIN_REDUCTION_THRESHOLD,
                use_estimates=USE_ESTIMATES_MODE
            )
        elif OPTIMIZATION_ALGORITHM == 'query_level':
            # 验证配置
            if not validate_experiment_config(DATASET, 'query_level', TRAIN_MODE):
                print("❌ 优化配置验证失败")
                return
            from optimize import run_query_level_optimization
            success = run_query_level_optimization(
                DATASET, OPTIMIZATION_ALGORITHM, TRAIN_MODE,
                base_dop=BASE_DOP
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
    
    # Train execution time model
    exec_results = modules['mci_train'].train_mci_execution_model(config, device)
    
    # Train memory model
    mem_results = modules['mci_train'].train_mci_memory_model(config, device)
    
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
    
    print("=" * 60)
    print("✅ MCI训练完成")
    print("Execution time model saved as PyTorch model")
    print("Memory model saved as PyTorch model")
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
        
        # 加载测试数据
        print("Loading test data...")
        test_loader_exec = create_mci_data_loaders_exec(
            csv_file=config.get_test_csv_path(),
            query_info_file=config.get_test_info_csv_path(),
            target_dop_levels=config.training.dop_levels,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            use_estimates=config.training.use_estimates
        )[1]  # Get test loader
        
        test_loader_mem = create_mci_data_loaders_mem(
            csv_file=config.get_test_csv_path(),
            query_info_file=config.get_test_info_csv_path(),
            target_dop_levels=config.training.dop_levels,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            use_estimates=config.training.use_estimates
        )[1]  # Get test loader
        
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
        
        # 保存结果
        results = {
            'execution_model': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in exec_results.items()},
            'memory_model': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in mem_results.items()},
        }
        
        results_path = os.path.join(config.data.output_dir, "mci_inference_results.json")
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
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
        
        # 加载pipeline信息
        pipeline_infos = modules['mci_optimize'].load_pipeline_info_for_moo(config, config.model.onnx_path)
        
        # 运行优化
        results = modules['mci_optimize'].run_moo_optimization(config, pipeline_infos)
        
        print("✅ MCI优化完成")
        return True
        
    except Exception as e:
        print(f"❌ MCI优化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
