# -*- coding: utf-8 -*-
"""
Unified inference entry point
Integrates all inference methods
"""

import argparse
import sys
import os

# Import configuration and utilities
from config.main_config import DATASETS, TRAIN_MODES, DEFAULT_CONFIG
from utils import (
    setup_environment, setup_config_structure, validate_experiment_config,
    log_experiment_start, log_experiment_end, safe_import, Timer,
    get_output_paths, create_dataset_loader
)

def run_inference(dataset: str, train_mode: str, use_estimates_mode: bool = True):
    """Run operator-level inference"""
    print(f"Starting operator-level inference...")
    
    # Use unified data loader
    loader = create_dataset_loader(dataset)
    
    # Get output paths
    output_paths = get_output_paths(dataset, 'inference', train_mode)
    
    # Get model paths - inference needs both dop_aware and non_dop_aware models
    from utils import get_model_paths
    model_paths = get_model_paths(dataset, train_mode, 'pipeline')
    
    # Import inference function
    inference_func = safe_import('inference.predict_queries', 'run_inference')
    if inference_func is None:
        return False
    
    # Get test file paths
    test_paths = loader.get_file_paths('test')
    
    # Execute inference
    with Timer("Operator-level inference"):
        inference_func(
            plan_csv_path=test_paths['plan_info'],
            query_csv_path=test_paths['query_info'],
            use_estimates=use_estimates_mode,
            output_csv_path=os.path.join(output_paths['prediction_dir'], 
                                       f"operator_level_{train_mode}_inference_results.csv"),
            no_dop_model_dir=model_paths['non_dop_aware_dir'],
            dop_model_dir=model_paths['dop_aware_dir']
        )
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified inference entry point')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='Dataset name')
    parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                       choices=list(TRAIN_MODES.keys()),
                       help='Training mode')
    parser.add_argument('--use_estimates', action='store_true', 
                       default=DEFAULT_CONFIG['use_estimates_mode'],
                       help='Whether to use estimates')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    setup_config_structure()
    
    # Validate configuration
    if not validate_experiment_config(args.dataset, 'dop_aware', args.train_mode):
        sys.exit(1)
    
    # Log experiment start
    eval_mode = 'estimated' if args.use_estimates else 'exact'
    log_experiment_start(args.dataset, 'inference', args.train_mode, eval_mode)
    
    # Execute inference
    success = run_inference(args.dataset, args.train_mode, args.use_estimates)
    
    # Log experiment end
    if success:
        log_experiment_end(args.dataset, 'inference')
        print("Inference completed!")
    else:
        print("Inference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
