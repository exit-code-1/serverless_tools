# -*- coding: utf-8 -*-
"""
Comparison analysis script
Integrates multi-method comparison functionality
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

# Import configuration and utilities
from config.main_config import DATASETS, DEFAULT_CONFIG, PROJECT_ROOT
from utils import (
    setup_environment, log_experiment_start, log_experiment_end, Timer,
    safe_import, check_file_exists
)


def _compute_stats(series):
    """Compute avg, p50, p90, p99 for a numeric series (dropna)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"avg": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "avg": float(s.mean()),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
    }


def print_summary_stats(comparison_csv_path: str):
    """
    Load comparison CSV and print/save summary: avg latency, avg cost, p50/p90/p99.
    """
    if not os.path.isfile(comparison_csv_path):
        print(f"Summary stats skipped: file not found {comparison_csv_path}")
        return
    try:
        df = pd.read_csv(comparison_csv_path, sep=";", encoding="utf-8")
    except Exception as e:
        print(f"Summary stats skipped: failed to read CSV: {e}")
        return

    latency_cols = [c for c in df.columns if c.endswith("_execution_time")]
    cost_cols = [c for c in df.columns if c.endswith("_cost") and "_ratio" not in c]
    # method name: strip _execution_time / _cost
    methods_lat = sorted(set(c.replace("_execution_time", "") for c in latency_cols))
    methods_cost = sorted(set(c.replace("_cost", "") for c in cost_cols))
    methods = sorted(set(methods_lat) | set(methods_cost))

    if not methods:
        print("No latency/cost columns found for summary stats.")
        return

    rows = []
    for method in methods:
        lat_col = f"{method}_execution_time" if f"{method}_execution_time" in df.columns else None
        cost_col = f"{method}_cost" if f"{method}_cost" in df.columns else None
        lat_stats = _compute_stats(df[lat_col]) if lat_col else {}
        cost_stats = _compute_stats(df[cost_col]) if cost_col else {}
        rows.append({
            "method": method,
            "latency_avg": lat_stats.get("avg", np.nan),
            "latency_p50": lat_stats.get("p50", np.nan),
            "latency_p90": lat_stats.get("p90", np.nan),
            "latency_p99": lat_stats.get("p99", np.nan),
            "cost_avg": cost_stats.get("avg", np.nan),
            "cost_p50": cost_stats.get("p50", np.nan),
            "cost_p90": cost_stats.get("p90", np.nan),
            "cost_p99": cost_stats.get("p99", np.nan),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = comparison_csv_path.replace(".csv", "_summary.csv")
    try:
        summary_df.to_csv(summary_path, sep=";", index=False, float_format="%.4f")
        print(f"Summary stats saved to: {summary_path}")
    except Exception as e:
        print(f"Could not write summary CSV: {e}")

    # Pretty print
    print("\n--- Summary: Avg latency, Avg cost, P50/P90/P99 ---")
    for _, r in summary_df.iterrows():
        print(f"  [{r['method']}]")
        print(f"    latency: avg={r['latency_avg']:.4f}, p50={r['latency_p50']:.4f}, p90={r['latency_p90']:.4f}, p99={r['latency_p99']:.4f}")
        print(f"    cost:    avg={r['cost_avg']:.4f}, p50={r['cost_p50']:.4f}, p90={r['cost_p90']:.4f}, p99={r['cost_p99']:.4f}")
    print()

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
            print_summary_stats(comparison_output_csv_path)
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
