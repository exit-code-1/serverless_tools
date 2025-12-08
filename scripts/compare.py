# -*- coding: utf-8 -*-
"""
Comparison analysis script
Integrates multi-method comparison functionality
"""

import argparse
import sys
import os

# Import configuration and utilities
from config.main_config import DATASETS, DEFAULT_CONFIG, PROJECT_ROOT
from utils import (
    setup_environment, log_experiment_start, log_experiment_end, Timer,
    safe_import, check_file_exists
)

def run_comparison(dataset: str):
    """Run multi-method comparison analysis"""
    print(f"Starting multi-method comparison analysis...")
    
    # Import comparison function
    compare_func = safe_import('optimization.result_processor', 'compare_results')
    if compare_func is None:
        return False
    
    # Define input file paths
    input_dir = os.path.join(PROJECT_ROOT, "input/overall_performance", dataset)
    baseline_dir = os.path.join(input_dir, "default")  # Baseline
    
    baseline_query_info_path = os.path.join(baseline_dir, "query_info.csv")
    baseline_plan_info_path = os.path.join(baseline_dir, "plan_info.csv")
    
    # Check baseline files
    if not check_file_exists(baseline_query_info_path, "Baseline query info file"):
        return False
    if not check_file_exists(baseline_plan_info_path, "Baseline plan info file"):
        return False
    
    # Define comparison methods
    strategies = ["PPM", "FGO", "OPRO"]
    method_info_list = []
    
    for method in strategies:
        method_dir = os.path.join(input_dir, method)
        query_info_path = os.path.join(method_dir, "query_info.csv")
        plan_info_path = os.path.join(method_dir, "plan_info.csv")
        
        if check_file_exists(query_info_path, f"Method {method} query info file") and \
           check_file_exists(plan_info_path, f"Method {method} plan info file"):
            method_info_list.append((method, query_info_path, plan_info_path))
        else:
            print(f"Warning: Method {method} missing required files, skipped")
    
    if not method_info_list:
        print("Error: No valid strategy directories found, cannot perform comparison analysis")
        return False
    
    # Define output path
    output_dir = os.path.join(PROJECT_ROOT, "output", "evaluations")
    os.makedirs(output_dir, exist_ok=True)
    comparison_output_csv_path = os.path.join(output_dir, "optimization_comparison_report.csv")
    
    # Execute comparison
    with Timer("Multi-method comparison analysis"):
        try:
            compare_func(
                baseline_query_info_path=baseline_query_info_path,
                baseline_plan_info_path=baseline_plan_info_path,
                method_info_list=method_info_list,
                output_csv_path=comparison_output_csv_path
            )
            print(f"Comparison analysis completed, results saved to: {comparison_output_csv_path}")
            return True
        except Exception as e:
            print(f"Error occurred while running comparison analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comparison analysis script')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='Dataset name')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Log experiment start
    log_experiment_start(args.dataset, 'comparison', 'N/A')
    
    # Execute comparison
    success = run_comparison(args.dataset)
    
    # Log experiment end
    if success:
        log_experiment_end(args.dataset, 'comparison')
        print("Comparison analysis completed!")
    else:
        print("Comparison analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
