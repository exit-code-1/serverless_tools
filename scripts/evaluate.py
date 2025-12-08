# -*- coding: utf-8 -*-
"""
Unified evaluation entry point
Integrates all evaluation methods
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import glob
import re

# Import configuration and utilities
from config.main_config import DATASETS, DEFAULT_CONFIG
from utils import (
    setup_environment, log_experiment_start, log_experiment_end, Timer,
    get_output_paths, load_csv_safe, save_csv_safe
)

def calculate_qerror_stats(df: pd.DataFrame, time_col: str, mem_col: str) -> dict:
    """Calculate Q-error statistics"""
    stats = {}
    
    # Time Q-error
    if time_col in df.columns:
        time_errors = pd.to_numeric(df[time_col], errors='coerce').dropna()
        time_errors = time_errors[time_errors >= 0]
        if not time_errors.empty:
            stats['p50_q_error_time'] = np.percentile(time_errors, 50)
            stats['p90_q_error_time'] = np.percentile(time_errors, 90)
            stats['avg_q_error_time'] = time_errors.mean()
    
    # Memory Q-error
    if mem_col in df.columns:
        mem_errors = pd.to_numeric(df[mem_col], errors='coerce').dropna()
        mem_errors = mem_errors[mem_errors >= 0]
        if not mem_errors.empty:
            stats['p50_q_error_mem'] = np.percentile(mem_errors, 50)
            stats['p90_q_error_mem'] = np.percentile(mem_errors, 90)
            stats['avg_q_error_mem'] = mem_errors.mean()
    
    return stats

def evaluate_predictions(dataset: str, train_mode: str):
    """Evaluate prediction results"""
    print(f"Starting prediction result evaluation...")
    
    # Get output paths
    output_paths = get_output_paths(dataset, 'evaluation', train_mode)
    prediction_dir = output_paths['prediction_dir']
    evaluation_dir = output_paths['evaluation_dir']
    
    all_stats = []
    
    with Timer("Prediction result evaluation"):
        # Process operator-level prediction results
        op_level_pattern = os.path.join(prediction_dir, 'operator_level', 'operator_level_*_inference_results.csv')
        for file_path in glob.glob(op_level_pattern):
            filename = os.path.basename(file_path)
            print(f"Processing operator-level file: {filename}")
            
            match = re.search(r'operator_level_(.+)_inference_results.csv', filename)
            if not match:
                continue
            
            experiment_type = match.group(1)
            model_name = '(Operator-Level)'
            
            try:
                df = load_csv_safe(file_path, description=f"Operator-level file {filename}")
                if df is not None:
                    stats = calculate_qerror_stats(df, 'Execution Time Q-error', 'Memory Q-error')
                    if stats:
                        stats['model'] = model_name
                        stats['experiment_type'] = experiment_type
                        all_stats.append(stats)
                        print(f"  > Model: {model_name}, Experiment: {experiment_type}, p90 time Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
            except Exception as e:
                print(f"Error processing file: {e}")
        
        # Process query-level prediction results
        query_level_pattern = os.path.join(prediction_dir, 'query_level', 'query_level_predictions_*.csv')
        for file_path in glob.glob(query_level_pattern):
            filename = os.path.basename(file_path)
            print(f"Processing query-level file: {filename}")
            
            match = re.search(r'query_level_predictions_(.+).csv', filename)
            if not match:
                continue
            
            experiment_type = match.group(1)
            model_name = 'Query-Level (XGBoost)'
            
            try:
                df = load_csv_safe(file_path, description=f"Query-level file {filename}")
                if df is not None:
                    stats = calculate_qerror_stats(df, 'Execution Time Q-error', 'Memory Q-error')
                    if stats:
                        stats['model'] = model_name
                        stats['experiment_type'] = experiment_type
                        all_stats.append(stats)
                        print(f"  > Model: {model_name}, Experiment: {experiment_type}, p90 time Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
            except Exception as e:
                print(f"Error processing file: {e}")
        
        # Process PPM prediction results
        ppm_patterns = [
            os.path.join(prediction_dir, 'PPM', '**', 'test_predictions.csv'),
            os.path.join(prediction_dir, 'PPM', '**', 'comparison.csv')
        ]
        
        all_ppm_files = []
        for pattern in ppm_patterns:
            all_ppm_files.extend(glob.glob(pattern, recursive=True))
        
        for file_path in all_ppm_files:
            print(f"Processing PPM file: {os.path.relpath(file_path, prediction_dir)}")
            
            path_parts = file_path.split(os.sep)
            try:
                model_type = path_parts[-2]  # NN or GNN
                train_mode_from_path = path_parts[-5].split('_', 1)[1]  # exact_train
                eval_mode_folder = path_parts[-4]  # eval_exact or eval_estimated
                
                model_name = f'PPM-{model_type}'
                experiment_type = f'{train_mode_from_path}_{eval_mode_folder}'
                
                df = load_csv_safe(file_path, description=f"PPM file {filename}")
                if df is not None:
                    # Check column names
                    time_q_error_col = 'Execution Time Q-error' if 'Execution Time Q-error' in df.columns else 'q_error_time'
                    mem_q_error_col = 'Memory Q-error' if 'Memory Q-error' in df.columns else 'q_error_mem'
                    
                    stats = calculate_qerror_stats(df, time_q_error_col, mem_q_error_col)
                    if stats:
                        stats['model'] = model_name
                        stats['experiment_type'] = experiment_type
                        all_stats.append(stats)
                        print(f"  > Model: {model_name}, Experiment: {experiment_type}, p90 time Q-Error: {stats.get('p90_q_error_time', 'N/A'):.2f}")
            except (IndexError, FileNotFoundError) as e:
                print(f"Error parsing PPM path or file: {file_path}, error: {e}")
            except Exception as e:
                print(f"Error processing PPM file: {e}")
    
    # Save summary report
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        cols_order = ['model', 'experiment_type', 'p50_q_error_time', 'p90_q_error_time', 'avg_q_error_time', 
                     'p50_q_error_mem', 'p90_q_error_mem', 'avg_q_error_mem']
        
        for col in cols_order:
            if col not in summary_df:
                summary_df[col] = np.nan
        
        summary_df = summary_df[cols_order]
        summary_df.sort_values(by=['model', 'experiment_type'], inplace=True)
        
        summary_file = os.path.join(evaluation_dir, "qerror_summary_report.csv")
        if save_csv_safe(summary_df, summary_file, description="Q-error summary report"):
            print(f"\nQ-error summary report saved: {summary_file}")
            print("\nReport content preview:")
            print(summary_df.to_string())
            return True
    
    print("\nNo statistics generated.")
    return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified evaluation entry point')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=list(DATASETS.keys()),
                       help='Dataset name')
    parser.add_argument('--train_mode', type=str, default=DEFAULT_CONFIG['train_mode'],
                       help='Training mode')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Log experiment start
    log_experiment_start(args.dataset, 'evaluation', args.train_mode)
    
    # Execute evaluation
    success = evaluate_predictions(args.dataset, args.train_mode)
    
    # Log experiment end
    if success:
        log_experiment_end(args.dataset, 'evaluation')
        print("Evaluation completed!")
    else:
        print("Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
