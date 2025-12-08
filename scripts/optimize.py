# -*- coding: utf-8 -*-
"""
Optimization module
Provides unified optimization interface
"""

import os
import sys
import pandas as pd
from typing import Dict, Any

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.optimizer import run_dop_optimization, run_query_dop_optimization
from utils import (
    get_output_paths, get_model_paths, create_dataset_loader, 
    log_experiment_start, log_experiment_end, Timer
)
from config import OPTIMIZATION_ALGORITHMS
import json
import csv

def export_optimization_to_csv(json_path: str, csv_path: str) -> bool:
    """
    Convert optimization results from JSON to CSV format
    
    Args:
        json_path: Path to JSON file
        csv_path: Path to output CSV file
        
    Returns:
        bool: Success status
    """
    try:
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare CSV data
        rows = []
        for query in data.get('queries', []):
            query_id = query['query_id']
            
            # Create a mapping from plan_id to optimal_dop
            plan_to_dop = {}
            for thread_block in query.get('thread_blocks', []):
                optimal_dop = thread_block.get('optimal_dop', 1)
                for operator in thread_block.get('operators', []):
                    plan_id = operator['plan_id']
                    plan_to_dop[plan_id] = optimal_dop
            
            # Generate rows
            for thread_block in query.get('thread_blocks', []):
                optimal_dop = thread_block.get('optimal_dop', 1)
                for operator in thread_block.get('operators', []):
                    row = {
                        'query_id': query_id,
                        'plan_id': operator['plan_id'],
                        'operator_type': operator['operator_type'],
                        'width': operator['width'],
                        'dop': optimal_dop,
                        'left_child': operator['left_child'],
                        'parent_child': operator['parent_child']
                    }
                    rows.append(row)
        
        # Sort by query_id and plan_id
        rows.sort(key=lambda x: (int(x['query_id']), x['plan_id']))
        
        # Write to CSV
        if rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['query_id', 'plan_id', 'operator_type', 'width', 'dop', 'left_child', 'parent_child']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"✅ CSV results exported to: {csv_path}")
            return True
        else:
            print("⚠️  No data to export")
            return False
            
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_baseline_optimization(dataset: str, train_mode: str,
                             base_dop: int = 64, use_estimates: bool = False,
                             model_dataset: str = None) -> bool:
    """
    Generate baseline configuration - all parallel operators use base_dop
    
    Args:
        dataset: Test dataset name
        train_mode: Training mode
        base_dop: Baseline DOP (default 64)
        use_estimates: Whether to use estimates
        model_dataset: Training model dataset name
        
    Returns:
        bool: Success status
    """
    try:
        if model_dataset is None:
            model_dataset = dataset
            
        log_experiment_start(dataset, f"Baseline DOP={base_dop}", train_mode)
        
        # Use unified data loader
        loader = create_dataset_loader(dataset)
        output_paths = get_output_paths(dataset, 'dop_aware', train_mode)
        
        # Load data
        df_plans = loader.load_test_data(use_estimates)
        if df_plans is None:
            return False
        
        # Filter to base_dop data only
        df_base = df_plans[df_plans['query_dop'] == base_dop].copy()
        if df_base.empty:
            print(f"❌ No data found for base_dop={base_dop}")
            return False
        
        # Set output paths
        output_json_path = os.path.join(output_paths['optimization_dir'], f"baseline_dop{base_dop}.json")
        output_csv_path = os.path.join(output_paths['optimization_dir'], f"baseline_dop{base_dop}.csv")
        
        # Generate baseline configuration
        from core.onnx_manager import ONNXModelManager
        model_paths = get_model_paths(model_dataset, train_mode, 'pipeline')
        onnx_manager = ONNXModelManager(
            no_dop_model_dir=model_paths['non_dop_aware_dir'],
            dop_model_dir=model_paths['dop_aware_dir']
        )
        
        from optimization.tree_updater import build_query_trees
        
        with Timer(f"Baseline DOP={base_dop} generation"):
            # Build query trees
            query_trees = build_query_trees(df_base, onnx_manager, use_estimates=use_estimates)
            
            # Generate baseline configuration
            queries_data = []
            for query_id, base_nodes in query_trees.items():
                query_info = {
                    'query_id': int(query_id),
                    'thread_blocks': []
                }
                
                # Collect all nodes using DFS
                from optimization.threading_utils import collect_all_nodes_by_plan_id, assign_thread_ids_by_plan_id
                
                # Assign thread IDs to each root pipeline with different offsets
                from optimization.threading_utils import get_root_nodes
                root_nodes = get_root_nodes(base_nodes)
                current_offset = 0
                for root in root_nodes:
                    assign_thread_ids_by_plan_id(root, thread_id=current_offset)
                    nodes_in_tree = collect_all_nodes_by_plan_id([root])
                    max_tid = max(getattr(node, 'thread_id', 0) for node in nodes_in_tree)
                    current_offset = max_tid + 1
                
                all_nodes = collect_all_nodes_by_plan_id(root_nodes)
                
                # Group nodes by their optimal DOP
                thread_blocks_dict = {}
                
                # Identify root pipelines by checking which thread_ids contain root nodes
                root_pipeline_thread_ids = set()
                for node in all_nodes:
                    if node.parent_node is None:
                        root_pipeline_thread_ids.add(getattr(node, 'thread_id', 0))
                
                for node in sorted(all_nodes, key=lambda n: n.plan_id):
                    # Determine DOP based on whether node is in root pipeline and operator type
                    # Root pipeline thread blocks: always use DOP=1
                    # GATHER operators: always use DOP=1
                    # Non-parallel operators: use DOP=1
                    # Others: use base_dop
                    
                    is_root_pipeline = getattr(node, 'thread_id', 0) in root_pipeline_thread_ids
                    is_gather = 'gather' in node.operator_type.lower()
                    is_parallel = node.is_parallel
                    
                    if is_root_pipeline or is_gather or not is_parallel:
                        node_dop = 1
                    else:
                        node_dop = base_dop
                    
                    # Group by DOP for thread block structure
                    if node_dop not in thread_blocks_dict:
                        thread_blocks_dict[node_dop] = {
                            'thread_block_id': len(thread_blocks_dict),
                            'optimal_dop': int(node_dop),
                            'predicted_time': None,
                            'operators': []
                        }
                    
                    op_info = {
                        'plan_id': int(node.plan_id),
                        'operator_type': str(node.operator_type),
                        'width': int(node.width),
                        'parent_child': int(node.parent_node.plan_id) if node.parent_node else -1,
                        'left_child': int(node.child_plans[0].plan_id) if node.child_plans else -1
                    }
                    thread_blocks_dict[node_dop]['operators'].append(op_info)
                
                # Add all thread blocks to query
                query_info['thread_blocks'] = list(thread_blocks_dict.values())
                queries_data.append(query_info)
            
            # Save to JSON
            import json
            result_data = {'queries': queries_data}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"✅ Baseline configuration saved to: {output_json_path}")
            
            # Export to CSV
            export_optimization_to_csv(output_json_path, output_csv_path)
        
        log_experiment_end(dataset, f"Baseline DOP={base_dop}", output_json_path)
        return True
        
    except Exception as e:
        print(f"❌ Baseline generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_pipeline_optimization(dataset: str, train_mode: str, 
                            base_dop: int = 64, min_improvement_ratio: float = 0.2,
                            min_reduction_threshold: int = 200, use_estimates: bool = False,
                            use_moo: bool = False, use_continuous_dop: bool = True,
                            moo_population_size: int = 30, moo_generations: int = 20, 
                            interval_tolerance: float = 0.3, model_dataset: str = None) -> bool:
    """
    Run Pipeline optimization - operator-level parallelism optimization
    
    Args:
        dataset: Test dataset name
        train_mode: Training mode
        base_dop: Baseline DOP
        min_improvement_ratio: Minimum improvement ratio
        min_reduction_threshold: Minimum reduction threshold
        use_estimates: Whether to use estimates
        use_moo: Whether to use MOO (Multi-Objective Optimization) algorithm
        use_continuous_dop: Whether MOO searches DOP in continuous interval (True=interval, False=discrete candidates)
        moo_population_size: MOO population size
        moo_generations: MOO evolution generations
        interval_tolerance: Throughput matching interval tolerance
        model_dataset: Training model dataset name (if None, use dataset)
        
    Returns:
        bool: Success status
    """
    try:
        # If model_dataset is not specified, use the same dataset as test data
        if model_dataset is None:
            model_dataset = dataset
        
        log_experiment_start(dataset, f"Pipeline operator-level optimization (model:{model_dataset})", train_mode)
        
        # Use unified data loader - test data from dataset
        loader = create_dataset_loader(dataset)
        output_paths = get_output_paths(dataset, 'dop_aware', train_mode)
        
        # Load data
        df_plans_all_dops = loader.load_test_data(use_estimates)
        if df_plans_all_dops is None:
            return False
        
        # Set output path
        output_json_path = os.path.join(output_paths['optimization_dir'], "pipeline_optimization.json")
        
        # Import ONNX model manager
        from core.onnx_manager import ONNXModelManager
        
        # Initialize model manager - models from model_dataset
        model_paths = get_model_paths(model_dataset, train_mode, 'pipeline')
        onnx_manager = ONNXModelManager(
            no_dop_model_dir=model_paths['non_dop_aware_dir'],
            dop_model_dir=model_paths['dop_aware_dir']
        )
        
        # Run optimization
        with Timer("Pipeline operator-level optimization"):
            result = run_dop_optimization(
                df_plans_all_dops, onnx_manager, output_json_path,
                base_dop=base_dop, use_estimates=use_estimates,
                min_improvement_ratio=min_improvement_ratio,
                min_reduction_threshold=min_reduction_threshold,
                interval_tolerance=interval_tolerance,
                use_moo=use_moo,
                use_continuous_dop=use_continuous_dop,
                moo_population_size=moo_population_size,
                moo_generations=moo_generations
            )
        
        if result is not None:
            print("✅ Pipeline operator-level optimization completed")
            
            # Export to CSV format
            csv_path = output_json_path.replace('.json', '.csv')
            export_optimization_to_csv(output_json_path, csv_path)
            
            log_experiment_end(dataset, "Pipeline operator-level optimization", output_json_path)
            return True
        else:
            print("❌ Pipeline operator-level optimization failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline operator-level optimization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_query_level_optimization(dataset: str, algorithm: str, train_mode: str,
                               base_dop: int = 64, use_estimates: bool = False, 
                               model_dataset: str = None) -> bool:
    """
    Run query-level optimization - entire query uses uniform parallelism
    
    Args:
        dataset: Test dataset name
        algorithm: Optimization algorithm ('query_level', 'auto_dop', 'ppm')
        train_mode: Training mode
        base_dop: Baseline DOP
        use_estimates: Whether to use estimates
        model_dataset: Training model dataset name (if None, use dataset)
        
    Returns:
        bool: Success status
    """
    try:
        # If model_dataset is not specified, use the same dataset as test data
        if model_dataset is None:
            model_dataset = dataset
            
        log_experiment_start(dataset, f"Query-level optimization-{algorithm} (model:{model_dataset})", train_mode)
        
        # Use unified data loader - test data from dataset
        loader = create_dataset_loader(dataset)
        output_paths = get_output_paths(dataset, 'query_level', train_mode)
        
        # Load data
        df_plans_all_dops = loader.load_test_data(use_estimates)
        if df_plans_all_dops is None:
            return False
        
        # Set output path
        output_json_path = os.path.join(output_paths['optimization_dir'], f"query_level_optimization_{algorithm}.json")
        
        # Determine parameters based on algorithm
        optimization_kwargs = {}
        prediction_file_path = None
        
        if algorithm == 'ppm':
            # PPM: Use curve fitting model directly, no prediction file needed
            # Locate PPM ONNX models - use model_dataset
            ppm_output_paths = get_output_paths(model_dataset, 'ppm', train_mode)
            
            # PPM models are stored in: models/exact_train/PPM/{NN,GNN}/
            ppm_base_dir = os.path.join(ppm_output_paths['model_dir'], 'PPM')
            
            # Try NN first, then GNN
            ppm_model_types = ['NN', 'GNN']
            ppm_model_dir = None
            ppm_model_type = None
            
            for model_type in ppm_model_types:
                potential_dir = os.path.join(ppm_base_dir, model_type)
                exec_model = os.path.join(potential_dir, 'execution_time_model.onnx')
                if os.path.exists(exec_model):
                    ppm_model_dir = potential_dir
                    ppm_model_type = model_type
                    print(f"Found PPM-{model_type} model: {ppm_model_dir}")
                    break
            
            if ppm_model_dir is None:
                print(f"❌ Error: PPM ONNX model not found (in {model_dataset} dataset)")
                print(f"   Please train PPM model for {model_dataset} dataset first. Search paths:")
                for model_type in ppm_model_types:
                    print(f"   - {os.path.join(ppm_base_dir, model_type)}")
                return False
            
            # Get test data path
            test_paths = loader.get_file_paths('test')
            plan_csv = test_paths['plan_info']
            
            # Set kwargs for PPM optimization
            optimization_kwargs = {
                'plan_csv': plan_csv,
                'ppm_model_dir': ppm_model_dir,
                'ppm_model_type': ppm_model_type,
                'use_estimates': use_estimates
            }
            
        elif algorithm in ['query_level', 'auto_dop']:
            # Query-level and auto_dop algorithms need query-level predictions
            # Note: predictions should be from the same dataset we're testing
            query_level_output_paths = get_output_paths(dataset, 'query_level', train_mode)
            # Look for query level prediction files
            import glob
            pattern = os.path.join(query_level_output_paths['prediction_dir'], 'query_level_predictions_*.csv')
            matching_files = glob.glob(pattern)
            if matching_files:
                prediction_file_path = matching_files[0]  # Use first matching file
                print(f"Found query-level prediction file: {prediction_file_path}")
            else:
                print(f"❌ Error: Query-level prediction file not found")
                print(f"   Please run query-level inference to generate prediction file first")
                print(f"   Search pattern: {pattern}")
                return False
        
        # Import ONNX model manager
        from core.onnx_manager import ONNXModelManager
        
        # Query-level optimization also needs operator models to predict performance
        # Models from model_dataset
        model_paths = get_model_paths(model_dataset, train_mode, 'pipeline')
        onnx_manager = ONNXModelManager(
            no_dop_model_dir=model_paths['non_dop_aware_dir'],
            dop_model_dir=model_paths['dop_aware_dir']
        )
        
        # Run optimization
        with Timer(f"Query-level optimization-{algorithm}"):
            result = run_query_dop_optimization(
                prediction_file_path, df_plans_all_dops, onnx_manager, output_json_path,
                algorithm=algorithm, base_dop=base_dop, **optimization_kwargs
            )
        
        if result is not None:
            print(f"✅ Query-level optimization-{algorithm} completed")
            
            # Export to CSV format
            csv_path = output_json_path.replace('.json', '.csv')
            export_optimization_to_csv(output_json_path, csv_path)
            
            log_experiment_end(dataset, f"Query-level optimization-{algorithm}", output_json_path)
            return True
        else:
            print(f"❌ Query-level optimization-{algorithm} failed")
            return False
            
    except Exception as e:
        print(f"❌ Query-level optimization-{algorithm} error: {e}")
        return False
