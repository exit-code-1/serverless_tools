# -*- coding: utf-8 -*-
"""
Unified training entry point
Integrates all training methods, controlled by parameters to select which method to use
"""

import argparse
import sys
import os

# Import configuration and utilities
from config.main_config import DATASETS, METHODS, TRAIN_MODES, DEFAULT_CONFIG
from utils import (
    setup_environment, setup_config_structure, validate_experiment_config,
    log_experiment_start, log_experiment_end, safe_import, Timer,
    get_output_paths, create_dataset_loader
)

def train_dop_aware_models(dataset: str, train_mode: str, **kwargs):
    """Train DOP-aware operator models"""
    print(f"Starting DOP-aware operator model training...")
    
    # Use unified data loader
    loader = create_dataset_loader(dataset)
    use_estimates = TRAIN_MODES[train_mode]['use_estimates']
    
    # Load data
    train_data = loader.load_train_data(use_estimates)
    test_data = loader.load_test_data(use_estimates)
    
    if train_data is None or test_data is None:
        return False
    
    # Import training function
    train_func = safe_import('training.operator_dop_aware.train', 'train_all_operators')
    if train_func is None:
        return False
    
    # Execute training
    with Timer("DOP-aware model training"):
        train_func(
            train_data=train_data,
            test_data=test_data,
            total_queries=kwargs.get('total_queries', DEFAULT_CONFIG['total_queries']),
            train_ratio=kwargs.get('train_ratio', DEFAULT_CONFIG['train_ratio']),
            use_estimates=TRAIN_MODES[train_mode]['use_estimates']
        )
    
    return True

def train_non_dop_aware_models(dataset: str, train_mode: str, **kwargs):
    """Train non-DOP-aware operator models"""
    print(f"Starting non-DOP-aware operator model training...")
    
    # Use unified data loader
    loader = create_dataset_loader(dataset)
    use_estimates = TRAIN_MODES[train_mode]['use_estimates']
    
    # Load data
    train_data = loader.load_train_data(use_estimates)
    test_data = loader.load_test_data(use_estimates)
    
    if train_data is None or test_data is None:
        return False
    
    # Import training function
    train_func = safe_import('training.operator_non_dop_aware.train', 'train_all_operators')
    if train_func is None:
        return False
    
    # Execute training
    with Timer("Non-DOP-aware model training"):
        train_func(
            train_data=train_data,
            test_data=test_data,
            total_queries=kwargs.get('total_queries', DEFAULT_CONFIG['total_queries']),
            train_ratio=kwargs.get('train_ratio', DEFAULT_CONFIG['train_ratio']),
            use_estimates=TRAIN_MODES[train_mode]['use_estimates']
        )
    
    return True

def train_ppm_models(dataset: str, train_mode: str, method_type: str = 'GNN', **kwargs):
    """Train PPM models"""
    print(f"Starting PPM model training ({method_type})...")
    
    # Select training module based on method type
    if method_type == 'GNN':
        train_func = safe_import('training.PPM.GNN_train', 'main')
    elif method_type == 'NN':
        train_func = safe_import('training.PPM.NN_train', 'main')
    else:
        print(f"Error: Unknown PPM method type: {method_type}")
        return False
    
    if train_func is None:
        return False
    
    # Execute training
    with Timer(f"PPM model training ({method_type})"):
        # Save original argv and set new argv to pass dataset and train_mode
        import sys
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], dataset, train_mode]
        try:
            train_func()
        finally:
            # Restore original argv
            sys.argv = original_argv
    
    return True

def train_query_level_models(dataset: str, train_mode: str, **kwargs):
    """Train query-level models"""
    print(f"Starting query-level model training...")
    
    # Use unified data loader
    loader = create_dataset_loader(dataset)
    use_estimates = TRAIN_MODES[train_mode]['use_estimates']
    
    # Get output paths
    output_paths = get_output_paths(dataset, 'query_level', train_mode)
    
    # Import training functions
    train_func = safe_import('training.query_level.train', 'train_and_save_xgboost_onnx')
    test_func = safe_import('training.query_level.train', 'test_onnx_xgboost')
    qerror_func = safe_import('training.query_level.train', 'compute_qerror_by_bins')
    
    if train_func is None or test_func is None or qerror_func is None:
        return False
    
    # Get file paths
    train_paths = loader.get_file_paths('train')
    
    # Execute training
    with Timer("Query-level model training"):
        train_func(
            feature_csv=train_paths['plan_info'],
            true_val_csv=train_paths['query_info'],
            execution_onnx_path=os.path.join(output_paths['model_dir'], "execution_time_model.onnx"),
            memory_onnx_path=os.path.join(output_paths['model_dir'], "memory_usage_model.onnx"),
            n_trials=kwargs.get('n_trials', 30),
            use_estimates=use_estimates
        )
    
    # Get test file paths
    test_paths = loader.get_file_paths('test')
    
    # Execute testing
    with Timer("Query-level model testing"):
        results_df = test_func(
            execution_onnx_path=os.path.join(output_paths['model_dir'], "execution_time_model.onnx"),
            memory_onnx_path=os.path.join(output_paths['model_dir'], "memory_usage_model.onnx"),
            feature_csv=test_paths['plan_info'],
            true_val_csv=test_paths['query_info'],
            output_file=os.path.join(output_paths['prediction_dir'], f"query_level_predictions_{train_mode}.csv"),
            use_estimates=use_estimates
        )
    
    # Calculate Q-error
    if results_df is not None:
        with Timer("Q-error calculation"):
            qerror_func(
                results_df, 
                os.path.join(output_paths['evaluation_dir'], f"query_level_qerror_{train_mode}.csv"),
                kwargs.get('target_dop', DEFAULT_CONFIG['target_dop'])
            )
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified training entry point')
    parser.add_argument('--method', type=str, required=True, 
                       choices=list(METHODS.keys()),
                       help='Training method: dop_aware, non_dop_aware, ppm, query_level')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='Dataset name')
    parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                       choices=list(TRAIN_MODES.keys()),
                       help='Training mode')
    parser.add_argument('--ppm_type', type=str, default='GNN',
                       choices=['GNN', 'NN'],
                       help='PPM method type (only effective when method=ppm)')
    parser.add_argument('--total_queries', type=int, default=DEFAULT_CONFIG['total_queries'],
                       help='Total number of queries')
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_CONFIG['train_ratio'],
                       help='Training ratio')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='XGBoost optimization trial count (only effective when method=query_level)')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    setup_config_structure()
    
    # Validate configuration
    if not validate_experiment_config(args.dataset, args.method, args.train_mode):
        sys.exit(1)
    
    # Log experiment start
    log_experiment_start(args.dataset, args.method, args.train_mode)
    
    # Execute training
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
    
    # Log experiment end
    if success:
        log_experiment_end(args.dataset, args.method)
        print("Training completed!")
    else:
        print("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
