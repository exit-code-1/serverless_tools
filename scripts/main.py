# -*- coding: utf-8 -*-
"""
Main control script
Control runtime parameters by modifying variables in the file, no command-line arguments needed
"""

import sys
import os
from tkinter import TRUE
import torch

# Add parent directory to Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and utilities
from config.main_config import DATASETS, METHODS, TRAIN_MODES, OPTIMIZATION_ALGORITHMS, DEFAULT_CONFIG
from utils import setup_environment, validate_experiment_config

def main():
    """Main function - control runtime parameters by modifying variables below"""
    
    # ==================== Configuration Area ====================
    # Modify parameters here to control which functions to run
    
    # Basic configuration
    DATASET = 'tpch'  # Test dataset: 'tpch' or 'tpcds'
    MODEL_DATASET = 'tpch'  # Training model dataset: 'tpch' or 'tpcds' (used to specify which dataset's model to load)
    TRAIN_MODE = 'exact_train'  # Training mode: 'exact_train' or 'estimated_train'
    USE_ESTIMATES_MODE = False  # Whether to use estimates
    
    # Training configuration
    TRAIN_METHOD = 'dop_aware'  # Training method: 'dop_aware', 'non_dop_aware', 'ppm', 'query_level', 'mci'
    PPM_TYPE = 'NN'  # PPM type: 'GNN' or 'NN' (only effective when TRAIN_METHOD='ppm')
    TOTAL_QUERIES = 500  # Total number of queries
    TRAIN_RATIO = 1.0  # Training ratio
    N_TRIALS = 30  # XGBoost optimization trial count (only effective when TRAIN_METHOD='query_level')
    
    # MCI configuration
    MCI_CONFIG_FILE = 'mci_config_small.json'  # MCI config file path
    # Optimization configuration
    OPTIMIZATION_ALGORITHM = 'pipeline'  # Optimization algorithm: 'pipeline', 'query_level', 'auto_dop', 'ppm', 'mci', 'base'
    BASE_DOP = 64  # Baseline DOP
    MIN_IMPROVEMENT_RATIO = 0.1  # Minimum improvement ratio
    MIN_REDUCTION_THRESHOLD = 200  # Minimum reduction threshold
    
    # MOO optimization configuration (only effective when OPTIMIZATION_ALGORITHM='pipeline')
    USE_MOO = True  # Whether to use Multi-Objective Optimization (MOO) algorithm
    USE_CONTINUOUS_DOP = True  # Whether MOO searches DOP in continuous interval (True=interval search, False=candidate list search)
    MOO_POPULATION_SIZE = 30  # MOO population size
    MOO_GENERATIONS = 20  # MOO evolution generations
    MOO_WEIGHT_LATENCY = 0.7  # Latency weight (Note: throughput matching is reflected through candidate intervals, not as an objective)
    MOO_WEIGHT_COST = 0.3  # Cost weight
    INTERVAL_TOLERANCE = 0.3  # Throughput matching interval tolerance (±30%) - used for choose_optimal_dop
    # PDG-MOO: competitive Top-K expansion + bounded elite pool (takes precedence over USE_MOO when True)
    USE_PDG_MOO = True  # Whether to use PDG-MOO (segment-aware iterative DOP optimization)
    PDG_MOO_WL = 0.7  # PDG-MOO latency weight (WL + WC = 1)
    PDG_MOO_WC = 0.3  # PDG-MOO cost weight
    PDG_MOO_B = 20  # Elite pool capacity
    PDG_MOO_T = 20  # Max rounds
    PDG_MOO_K = 5   # Top-K actions per parent
    PDG_MOO_B_SLOW = 4   # Top-b slowest segs for A_balance
    PDG_MOO_P = 15  # Top-p stages for A_profit
    PDG_MOO_LAMBDA_I = 0.1  # Interference penalty weight I/L
    
    # Runtime control - set which functions to run (True/False)
    RUN_TRAIN =  False  # Whether to run training
    RUN_INFERENCE = False  # Whether to run inference
    RUN_OPTIMIZE = False  # Whether to run optimization
    RUN_EVALUATE = False  # Whether to run evaluation
    RUN_COMPARE = False  # Whether to run comparison analysis
    RUN_COMPARE_INFERENCE_METHODS = True  # Whether to run inference method comparison
    # =======================================================
    
    # Setup environment
    setup_environment()
    
    print("=" * 60)
    print("Serverless Predictor Main Control Script")
    print("=" * 60)
    print(f"Test dataset: {DATASET}")
    print(f"Model dataset: {MODEL_DATASET}")
    print(f"Training mode: {TRAIN_MODE}")
    print(f"Use estimates: {USE_ESTIMATES_MODE}")
    print("=" * 60)
    
    # Run training
    if RUN_TRAIN:
        print(f"\n🚀 Starting training: {TRAIN_METHOD}")
        # Validate configuration
        if not validate_experiment_config(DATASET, TRAIN_METHOD, TRAIN_MODE):
            print("❌ Training configuration validation failed")
            return
        
        # Call training function
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
            success = train_ppm_models(DATASET, TRAIN_MODE, PPM_TYPE, train_ratio=TRAIN_RATIO)
        elif TRAIN_METHOD == 'query_level':
            from train import train_query_level_models
            success = train_query_level_models(DATASET, TRAIN_MODE, n_trials=N_TRIALS, train_ratio=TRAIN_RATIO)
        elif TRAIN_METHOD == 'mci':
            success = run_mci_training(DATASET, MCI_CONFIG_FILE)
        
        if success:
            print("✅ Training completed")
        else:
            print("❌ Training failed")
            return
    
    # Run inference
    if RUN_INFERENCE:
        print(f"\n🔍 Starting inference: {TRAIN_METHOD}")
        
        # Call inference function
        if TRAIN_METHOD == 'dop_aware' or TRAIN_METHOD == 'non_dop_aware':
            # Validate configuration
            if not validate_experiment_config(DATASET, TRAIN_METHOD, TRAIN_MODE):
                print("❌ Inference configuration validation failed")
                return
            from inference import run_inference
            success = run_inference(DATASET, TRAIN_MODE, USE_ESTIMATES_MODE, train_ratio=TRAIN_RATIO)
        elif TRAIN_METHOD == 'ppm':
            # Validate configuration
            if not validate_experiment_config(DATASET, 'ppm', TRAIN_MODE):
                print("❌ Inference configuration validation failed")
                return
            from inference import run_ppm_inference
            success = run_ppm_inference(DATASET, TRAIN_MODE, PPM_TYPE, train_ratio=TRAIN_RATIO)
        elif TRAIN_METHOD == 'query_level':
            # Validate configuration
            if not validate_experiment_config(DATASET, 'query_level', TRAIN_MODE):
                print("❌ Inference configuration validation failed")
                return
            from inference import run_query_level_inference
            success = run_query_level_inference(DATASET, TRAIN_MODE, train_ratio=TRAIN_RATIO)
        elif TRAIN_METHOD == 'mci':
            success = run_mci_inference(DATASET, MCI_CONFIG_FILE)
        else:
            print(f"❌ Unknown training method: {TRAIN_METHOD}")
            return
        
        if success:
            print("✅ Inference completed")
        else:
            print("❌ Inference failed")
            return
    
    # Run optimization
    if RUN_OPTIMIZE:
        print(f"\n⚡ Starting optimization: {OPTIMIZATION_ALGORITHM}")
        
        # Call optimization function - based on OPTIMIZATION_ALGORITHM, not TRAIN_METHOD
        if OPTIMIZATION_ALGORITHM == 'base':
            # Baseline: all parallel operators use BASE_DOP (default 64)
            from optimize import run_baseline_optimization
            success = run_baseline_optimization(
                DATASET, TRAIN_MODE,
                base_dop=BASE_DOP,
                use_estimates=USE_ESTIMATES_MODE,
                model_dataset=MODEL_DATASET,
                train_ratio=TRAIN_RATIO
            )
        elif OPTIMIZATION_ALGORITHM == 'mci':
            success = run_mci_optimization(DATASET, MCI_CONFIG_FILE)
        elif OPTIMIZATION_ALGORITHM == 'pipeline':
            # Pipeline: operator-level DOP optimization
            # Validate that operator models exist (trained with dop_aware method)
            if not validate_experiment_config(MODEL_DATASET, 'dop_aware', TRAIN_MODE):
                print(f"❌ Optimization configuration validation failed - need to train {MODEL_DATASET} dataset's dop_aware model first")
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
                use_pdg_moo=USE_PDG_MOO,
                pdg_moo_WL=PDG_MOO_WL,
                pdg_moo_WC=PDG_MOO_WC,
                pdg_moo_B=PDG_MOO_B,
                pdg_moo_T=PDG_MOO_T,
                pdg_moo_K=PDG_MOO_K,
                pdg_moo_b=PDG_MOO_B_SLOW,
                pdg_moo_p=PDG_MOO_P,
                pdg_moo_lambda_I=PDG_MOO_LAMBDA_I,
                model_dataset=MODEL_DATASET,  # Pass model dataset separately
                train_ratio=TRAIN_RATIO
            )
        elif OPTIMIZATION_ALGORITHM in ['query_level', 'auto_dop', 'ppm']:
            # Query-level: uniform DOP for entire query
            # Validate that operator models exist (needed for performance prediction)
            if not validate_experiment_config(MODEL_DATASET, 'dop_aware', TRAIN_MODE):
                print(f"❌ Optimization configuration validation failed - need to train {MODEL_DATASET} dataset's dop_aware model first")
                return
            from optimize import run_query_level_optimization
            success = run_query_level_optimization(
                DATASET, OPTIMIZATION_ALGORITHM, TRAIN_MODE,
                base_dop=BASE_DOP,
                use_estimates=USE_ESTIMATES_MODE,
                model_dataset=MODEL_DATASET,  # Pass model dataset separately
                train_ratio=TRAIN_RATIO
            )
        else:
            print(f"❌ Unknown optimization algorithm: {OPTIMIZATION_ALGORITHM}")
            return
        
        if success:
            print("✅ Optimization completed")
        else:
            print("❌ Optimization failed")
            return
    
    # Run evaluation
    if RUN_EVALUATE:
        print(f"\n📊 Starting evaluation")
        # Call evaluation function
        from evaluate import evaluate_predictions
        success = evaluate_predictions(DATASET, TRAIN_MODE)
        
        if success:
            print("✅ Evaluation completed")
        else:
            print("❌ Evaluation failed")
            return
    
    # Run comparison analysis
    if RUN_COMPARE:
        print(f"\n📈 Starting comparison analysis")
        # Call comparison function
        from compare import run_comparison
        success = run_comparison(DATASET)
        
        if success:
            print("✅ Comparison analysis completed")
        else:
            print("❌ Comparison analysis failed")
            return
    
    # Run inference method comparison
    if RUN_COMPARE_INFERENCE_METHODS:
        print(f"\n🔍 Starting inference method comparison")
        # Call inference method comparison function
        from compare_inference_methods import compare_inference_methods
        success = compare_inference_methods(DATASET, TRAIN_MODE, USE_ESTIMATES_MODE, MODEL_DATASET)
        
        if success:
            print("✅ Inference method comparison completed")
        else:
            print("❌ Inference method comparison failed")
            return
    
    print("\n" + "=" * 60)
    print("🎉 All tasks completed!")
    print("=" * 60)


# ==================== MCI Function Functions ====================

def _load_mci_modules(mci_path: str, module_names: list):
    """Dynamically load MCI modules"""
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
    """Setup MCI configuration"""
    # Create config file if it doesn't exist
    config_file_path = os.path.join(mci_path, config_file)
    if not os.path.exists(config_file_path):
        print(f"Creating MCI config file: {config_file_path}")
        create_config_file(config_file_path, "small")
    
    # Load configuration
    config = MCIConfig.load_config(config_file_path)
    
    # Override dataset from parameter
    config.data.dataset = dataset
    print(f"Using dataset: {dataset}")
    
    # Update dataset paths
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
    """Run MCI training - call standard training process"""
    print("Starting MCI model training...")
    
    # Add MCI module path
    mci_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'MCI')
    sys.path.insert(0, mci_path)
    
    # Load MCI modules
    modules = _load_mci_modules(mci_path, ['mci_train', 'mci_model', 'mci_data_loader'])
    
    # Import MCI configuration
    from config.mci_config import MCIConfig
    
    # Setup configuration
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
    print("✅ MCI training completed")
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
    """Run MCI inference"""
    try:
        print("Starting MCI model inference...")
        
        # Add MCI module path
        mci_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'MCI')
        sys.path.insert(0, mci_path)
        
        # Import MCI configuration
        from config.mci_config import MCIConfig
        
        # Load configuration
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
        
        # Import inference modules (dynamic import, available at runtime)
        from mci_inference import load_pytorch_model, predict_with_pytorch_model  # type: ignore
        from mci_data_loader import create_mci_data_loaders_exec, create_mci_data_loaders_mem  # type: ignore
        
        # Determine model paths
        exec_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_exec.pth")
        mem_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_mem.pth")
        
        print(f"Execution time model: {exec_model_path}")
        print(f"Memory model: {mem_model_path}")
        
        # Check if models exist
        if not os.path.exists(exec_model_path):
            print(f"❌ Execution time model not found: {exec_model_path}")
            return False
        if not os.path.exists(mem_model_path):
            print(f"❌ Memory model not found: {mem_model_path}")
            return False
        
        # Load PyTorch models
        print("Loading PyTorch models...")
        exec_model = load_pytorch_model(exec_model_path, config, device)
        mem_model = load_pytorch_model(mem_model_path, config, device)
        
        # Load test data - use all data for inference
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
        
        # Run inference
        print("Running execution time inference...")
        exec_results = predict_with_pytorch_model(exec_model, test_loader_exec, device)
        
        print("Running memory inference...")
        mem_results = predict_with_pytorch_model(mem_model, test_loader_mem, device)
        
        # Print results
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
            print(f"\n⚠️  Detected NaN predictions, locating specific queries:")
            
            if len(nan_exec_indices) > 0:
                print(f"\n⚠️  Execution time model - {len(nan_exec_indices)} NaNs:")
                nan_queries_exec = {}
                for idx in nan_exec_indices:
                    qid, dop = pipeline_info[idx]
                    key = (qid, dop)
                    if key not in nan_queries_exec:
                        nan_queries_exec[key] = 0
                    nan_queries_exec[key] += 1
                
                print(f"    Involved (query_id, DOP) combinations:")
                for (qid, dop), count in sorted(nan_queries_exec.items())[:20]:
                    print(f"      - Query {qid}, DOP={dop}: {count} pipelines")
            
            if len(nan_mem_indices) > 0:
                print(f"\n⚠️  Memory model - {len(nan_mem_indices)} NaNs:")
                nan_queries_mem = {}
                for idx in nan_mem_indices:
                    qid, dop = pipeline_info[idx]
                    key = (qid, dop)
                    if key not in nan_queries_mem:
                        nan_queries_mem[key] = 0
                    nan_queries_mem[key] += 1
                
                print(f"    Involved (query_id, DOP) combinations:")
                for (qid, dop), count in sorted(nan_queries_mem.items())[:20]:
                    print(f"      - Query {qid}, DOP={dop}: {count} pipelines")
            
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
            print(f"\n⚠️  Query-level NaN statistics:")
            print(f"    - Number of queries with execution time NaN: {nan_exec_queries}/{len(df)} ({nan_exec_queries/len(df)*100:.1f}%)")
            print(f"    - Number of queries with memory NaN: {nan_mem_queries}/{len(df)} ({nan_mem_queries/len(df)*100:.1f}%)")
            
            # Show which queries have NaN
            if nan_exec_queries > 0:
                nan_exec_list = df[df['has_nan_exec']]['query_id'].unique()
                print(f"    - Query IDs with execution time NaN: {sorted(nan_exec_list)[:20]}")
            if nan_mem_queries > 0:
                nan_mem_list = df[df['has_nan_mem']]['query_id'].unique()
                print(f"    - Query IDs with memory NaN: {sorted(nan_mem_list)[:20]}")
        
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
        print("✅ MCI inference completed")
        return True
        
    except Exception as e:
        print(f"❌ MCI inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_mci_optimization(dataset: str, config_file: str) -> bool:
    """Run MCI optimization"""
    try:
        print("Starting MCI MOO optimization...")
        
        # Add MCI module path
        mci_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'MCI')
        sys.path.insert(0, mci_path)
        
        # Load MCI modules
        modules = _load_mci_modules(mci_path, ['mci_optimize'])
        
        # Import MCI configuration
        from config.mci_config import MCIConfig, create_config_file, PresetConfigs
        mci_config = type('MCIConfigModule', (), {
            'MCIConfig': MCIConfig,
            'create_config_file': create_config_file,
            'PresetConfigs': PresetConfigs
        })()
        
        # Setup configuration
        config = _setup_mci_config(dataset, config_file, mci_path, create_config_file, MCIConfig)
        
        # Build PyTorch model path (using execution time model)
        pytorch_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_exec.pth")
        
        if not os.path.exists(pytorch_model_path):
            print(f"❌ PyTorch model file does not exist: {pytorch_model_path}")
            print("Please run training first to generate model file")
            return False
        
        # Run optimization
        results = modules['mci_optimize'].run_moo_optimization(
            config=config,
            pytorch_model_path=pytorch_model_path,
            output_dir=config.data.output_dir,
            population_size=10,
            generations=6,
            weight_latency=0.9,
            weight_cost=0.1
        )
        
        print("✅ MCI optimization completed")
        return True
        
    except Exception as e:
        print(f"❌ MCI optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
