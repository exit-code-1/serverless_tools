"""
MCI Model Inference Script
MCI模型推理脚本

支持PyTorch模型推理和评估
"""

import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from mci_model import MCILatencyModel, create_mci_model
from mci_data_loader import (
    load_mci_pipeline_data_exec,
    load_mci_pipeline_data_mem,
    create_mci_data_loaders_exec,
    create_mci_data_loaders_mem
)
from config import MCIConfig


def load_pytorch_model(model_path: str, config: MCIConfig, device: str = 'cpu') -> MCILatencyModel:
    """
    Load PyTorch model from saved checkpoint
    
    Args:
        model_path: Path to the saved .pth model file
        config: Model configuration
        device: Device to load model on
    
    Returns:
        Loaded MCI model
    """
    print(f"Loading PyTorch model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model architecture parameters from checkpoint
    model_config = checkpoint.get('model_config', {})
    
    # Use saved config if available, otherwise fall back to provided config
    input_dim = model_config.get('input_dim', config.model.input_dim)
    hidden_dim = model_config.get('hidden_dim', config.model.hidden_dim)
    embedding_dim = model_config.get('embedding_dim', config.model.embedding_dim)
    num_heads = model_config.get('num_heads', config.model.num_heads)
    num_layers = model_config.get('num_layers', config.model.num_layers)
    predictor_hidden_dims = model_config.get('predictor_hidden_dims', config.model.predictor_hidden_dims)
    dropout = model_config.get('dropout', config.model.dropout)
    
    # Create model with same architecture
    model = create_mci_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        predictor_hidden_dims=predictor_hidden_dims,
        dropout=dropout
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def predict_with_pytorch_model(model: MCILatencyModel, data_loader: DataLoader, device: str = 'cpu'):
    """
    Make predictions using PyTorch model
    
    Args:
        model: Loaded PyTorch model
        data_loader: Data loader for test data
        device: Device to run inference on
    
    Returns:
        Dictionary with predictions and metrics
    """
    model.eval()
    predictions = []
    targets = []
    q_errors = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # Get predictions
            pred, _ = model(
                data.x, 
                data.edge_index, 
                data.batch,
                data.dop_level.squeeze()
            )
            
            target = data.y.squeeze()
            pred = pred.squeeze()
            
            # Calculate Q-error
            q_error = torch.max(pred / (target + 1e-8), target / (pred + 1e-8)) - 1
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            q_errors.extend(q_error.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    q_errors = np.array(q_errors)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    q_error_mean = np.mean(q_errors)
    q_error_median = np.median(q_errors)
    
    return {
        'mse': mse,
        'mae': mae,
        'q_error_mean': q_error_mean,
        'q_error_median': q_error_median,
        'predictions': predictions,
        'targets': targets
    }


def load_test_data_by_query_exec(config: MCIConfig):
    """Load test data grouped by query for execution time inference"""
    print("Loading execution time test data by query...")
    
    if not config.data.test_csv or not config.data.test_info_csv:
        raise ValueError("Test CSV files not specified in config")
    
    # Load all test data for execution time
    all_test_data = load_mci_pipeline_data_exec(
        csv_file=config.data.test_csv,
        query_info_file=config.data.test_info_csv,
        target_dop_levels=config.training.dop_levels,
        use_estimates=config.training.use_estimates
    )
    
    # Group data by query_id
    query_data = {}
    for data in all_test_data:
        # Extract query_id from pipeline metadata or use a default method
        if hasattr(data, 'pipeline_metadata') and 'query_id' in data.pipeline_metadata:
            query_id = data.pipeline_metadata['query_id']
        else:
            # Fallback: use thread_id as query identifier
            query_id = f"query_{data.thread_id.item()}"
        
        if query_id not in query_data:
            query_data[query_id] = []
        query_data[query_id].append(data)
    
    print(f"Loaded {len(all_test_data)} execution time pipeline samples across {len(query_data)} queries")
    return query_data


def load_test_data_by_query_mem(config: MCIConfig):
    """Load test data grouped by query for memory inference"""
    print("Loading memory test data by query...")
    
    if not config.data.test_csv or not config.data.test_info_csv:
        raise ValueError("Test CSV files not specified in config")
    
    # Load all test data for memory usage
    all_test_data = load_mci_pipeline_data_mem(
        csv_file=config.data.test_csv,
        query_info_file=config.data.test_info_csv,
        target_dop_levels=config.training.dop_levels,
        use_estimates=config.training.use_estimates
    )
    
    # Group data by query_id
    query_data = {}
    for data in all_test_data:
        # Extract query_id from pipeline metadata or use a default method
        if hasattr(data, 'pipeline_metadata') and 'query_id' in data.pipeline_metadata:
            query_id = data.pipeline_metadata['query_id']
        else:
            # Fallback: use thread_id as query identifier
            query_id = f"query_{data.thread_id.item()}"
        
        if query_id not in query_data:
            query_data[query_id] = []
        query_data[query_id].append(data)
    
    print(f"Loaded {len(all_test_data)} memory pipeline samples across {len(query_data)} queries")
    return query_data


def run_query_level_inference_dual(exec_onnx_model_path: str, mem_onnx_model_path: str, 
                                   query_data_exec: dict, query_data_mem: dict, 
                                   output_dir: str):
    """Run query-level ONNX inference and evaluation for both execution time and memory models"""
    print(f"Running query-level ONNX inference with execution model: {exec_onnx_model_path}")
    print(f"Running query-level ONNX inference with memory model: {mem_onnx_model_path}")
    
    # Run execution time inference
    exec_results = run_query_level_inference(exec_onnx_model_path, query_data_exec, output_dir, 'execution_time')
    
    # Run memory inference  
    mem_results = run_query_level_inference(mem_onnx_model_path, query_data_mem, output_dir, 'memory_usage')
    
    # Create combined results in standard format
    combined_results = create_combined_results(exec_results, mem_results, output_dir)
    
    return combined_results


def create_combined_results(exec_results: dict, mem_results: dict, output_dir: str) -> dict:
    """Create combined results in standard format matching other methods"""
    
    # Load the detailed results from CSV files
    exec_pipeline_file = os.path.join(output_dir, "mci_execution_time_pipeline_results.csv")
    mem_pipeline_file = os.path.join(output_dir, "mci_memory_usage_pipeline_results.csv")
    
    exec_pipeline_df = pd.read_csv(exec_pipeline_file, sep=';')
    mem_pipeline_df = pd.read_csv(mem_pipeline_file, sep=';')
    
    # Merge execution time and memory results by Query ID and Pipeline ID
    combined_df = pd.merge(
        exec_pipeline_df, 
        mem_pipeline_df[['Query ID', 'Pipeline ID', 'Actual Memory Usage (MB)', 'Predicted Memory Usage (MB)', 'Memory Q-error', 'Memory Calculation Duration (s)']],
        on=['Query ID', 'Pipeline ID'],
        how='inner'
    )
    
    # Create standard format matching other methods
    standard_df = pd.DataFrame({
        'Query ID': combined_df['Query ID'],
        'Query DOP': combined_df['Query DOP'],
        'Query ID (Mapped)': range(1, len(combined_df) + 1),
        'Actual Execution Time (s)': combined_df['Actual Execution Time (s)'],
        'Predicted Execution Time (s)': combined_df['Predicted Execution Time (s)'],
        'Actual Memory Usage (MB)': combined_df['Actual Memory Usage (MB)'],
        'Predicted Memory Usage (MB)': combined_df['Predicted Memory Usage (MB)'],
        'Execution Time Q-error': combined_df['Execution Time Q-error'],
        'Memory Q-error': combined_df['Memory Q-error'],
        'Time Calculation Duration (s)': combined_df['Time Calculation Duration (s)'],
        'Memory Calculation Duration (s)': combined_df['Memory Calculation Duration (s)']
    })
    
    # Save combined results in standard format
    combined_output_file = os.path.join(output_dir, "mci_combined_inference_results.csv")
    standard_df.to_csv(combined_output_file, index=False, sep=';')
    
    print(f"Combined inference results saved to: {combined_output_file}")
    
    # Create summary statistics
    combined_summary = {
        'execution_time': exec_results,
        'memory_usage': mem_results,
        'combined_stats': {
            'num_queries': len(standard_df['Query ID'].unique()),
            'num_pipelines': len(standard_df),
            'execution_time_mean_q_error': exec_results['query_mean_q_error'],
            'execution_time_median_q_error': exec_results['query_median_q_error'],
            'memory_mean_q_error': mem_results['query_mean_q_error'],
            'memory_median_q_error': mem_results['query_median_q_error']
        }
    }
    
    # Save combined summary
    combined_summary_file = os.path.join(output_dir, "mci_combined_summary.json")
    import json
    with open(combined_summary_file, 'w') as f:
        json.dump(combined_summary, f, indent=2, default=str)
    
    return combined_summary


def run_query_level_inference(onnx_model_path: str, query_data: dict, output_dir: str, target_type: str = 'execution_time'):
    """Run query-level ONNX inference and evaluation"""
    print(f"Running query-level ONNX inference with model: {onnx_model_path}")
    
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    
    # Load ONNX model
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_model_path)
    
    query_results = []
    pipeline_results = []
    
    for query_id, pipelines in query_data.items():
        print(f"Processing query: {query_id} with {len(pipelines)} pipelines")
        
        query_predicted_value = 0.0
        query_actual_value = 0.0
        query_pipeline_count = 0
        
        for pipeline_data in pipelines:
            # Prepare inputs for ONNX
            x_np = pipeline_data.x.numpy().astype(np.float32)
            edge_index_np = pipeline_data.edge_index.numpy().astype(np.int64)
            batch_np = pipeline_data.batch.numpy().astype(np.int64)
            dop_np = pipeline_data.dop_level.numpy().astype(np.float32)
            
            ort_inputs = {
                "x": x_np,
                "edge_index": edge_index_np,
                "batch": batch_np,
                "dop_levels": dop_np
            }
            
            # Run inference
            start_time = time.time()
            outputs = session.run(["latency_predictions"], ort_inputs)
            inference_time = time.time() - start_time
            
            pred_value = outputs[0].flatten()[0]
            actual_value = pipeline_data.y.numpy().flatten()[0]
            
            # Calculate Q-error
            q_error = max(pred_value / (actual_value + 1e-8), actual_value / (pred_value + 1e-8)) - 1
            
            # Accumulate for query-level results
            query_predicted_value += pred_value
            query_actual_value += actual_value
            query_pipeline_count += 1
            
            # Store pipeline-level results
            pipeline_results.append({
                'query_id': query_id,
                'pipeline_id': f"pipeline_{pipeline_data.thread_id.item()}",
                'dop': pipeline_data.dop_level.item(),
                'actual_value': actual_value,
                'predicted_value': pred_value,
                'q_error': q_error,
                'inference_time': inference_time,
                'target_type': target_type
            })
        
        # Calculate query-level Q-error
        if query_pipeline_count > 0:
            query_q_error = max(query_predicted_value / (query_actual_value + 1e-8), 
                              query_actual_value / (query_predicted_value + 1e-8)) - 1
            
            query_results.append({
                'query_id': query_id,
                'num_pipelines': query_pipeline_count,
                'actual_value': query_actual_value,
                'predicted_value': query_predicted_value,
                'q_error': query_q_error,
                'target_type': target_type
            })
    
    # Save results in standard format
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames
    pipeline_df = pd.DataFrame(pipeline_results)
    query_df = pd.DataFrame(query_results)
    
    # Create output files with standard naming
    pipeline_output_file = os.path.join(output_dir, f"mci_{target_type}_pipeline_results.csv")
    query_output_file = os.path.join(output_dir, f"mci_{target_type}_query_results.csv")
    
    # Format columns to match standard format
    if target_type == 'execution_time':
        pipeline_df_formatted = pd.DataFrame({
            'Query ID': pipeline_df['query_id'],
            'Pipeline ID': pipeline_df['pipeline_id'],
            'Query DOP': pipeline_df['dop'],
            'Actual Execution Time (s)': pipeline_df['actual_value'] / 1000.0,  # Convert to seconds
            'Predicted Execution Time (s)': pipeline_df['predicted_value'] / 1000.0,
            'Execution Time Q-error': pipeline_df['q_error'],
            'Time Calculation Duration (s)': pipeline_df['inference_time']
        })
        
        query_df_formatted = pd.DataFrame({
            'Query ID': query_df['query_id'],
            'Num Pipelines': query_df['num_pipelines'],
            'Actual Execution Time (s)': query_df['actual_value'] / 1000.0,
            'Predicted Execution Time (s)': query_df['predicted_value'] / 1000.0,
            'Execution Time Q-error': query_df['q_error']
        })
    else:  # memory_usage
        pipeline_df_formatted = pd.DataFrame({
            'Query ID': pipeline_df['query_id'],
            'Pipeline ID': pipeline_df['pipeline_id'],
            'Query DOP': pipeline_df['dop'],
            'Actual Memory Usage (MB)': pipeline_df['actual_value'] / 1000.0,  # Convert to MB
            'Predicted Memory Usage (MB)': pipeline_df['predicted_value'] / 1000.0,
            'Memory Q-error': pipeline_df['q_error'],
            'Memory Calculation Duration (s)': pipeline_df['inference_time']
        })
        
        query_df_formatted = pd.DataFrame({
            'Query ID': query_df['query_id'],
            'Num Pipelines': query_df['num_pipelines'],
            'Actual Memory Usage (MB)': query_df['actual_value'] / 1000.0,
            'Predicted Memory Usage (MB)': query_df['predicted_value'] / 1000.0,
            'Memory Q-error': query_df['q_error']
        })
    
    # Save with semicolon separator to match other methods
    pipeline_df_formatted.to_csv(pipeline_output_file, index=False, sep=';')
    query_df_formatted.to_csv(query_output_file, index=False, sep=';')
    
    print(f"Pipeline results saved to: {pipeline_output_file}")
    print(f"Query results saved to: {query_output_file}")
    
    # Calculate summary statistics
    query_q_errors = [r['q_error'] for r in query_results]
    pipeline_q_errors = [r['q_error'] for r in pipeline_results]
    
    summary = {
        'num_queries': len(query_results),
        'num_pipelines': len(pipeline_results),
        'query_mean_q_error': np.mean(query_q_errors),
        'query_median_q_error': np.median(query_q_errors),
        'query_max_q_error': np.max(query_q_errors),
        'pipeline_mean_q_error': np.mean(pipeline_q_errors),
        'pipeline_median_q_error': np.median(pipeline_q_errors),
        'pipeline_max_q_error': np.max(pipeline_q_errors),
        'target_type': target_type
    }
    
    # Save summary
    summary_file = os.path.join(output_dir, f"mci_{target_type}_summary.csv")
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_file, index=False, sep=';')
    
    print(f"Query-level inference completed:")
    print(f"  Queries: {summary['num_queries']}")
    print(f"  Pipelines: {summary['num_pipelines']}")
    print(f"  Query mean Q-error: {summary['query_mean_q_error']:.4f}")
    print(f"  Query median Q-error: {summary['query_median_q_error']:.4f}")
    print(f"  Pipeline mean Q-error: {summary['pipeline_mean_q_error']:.4f}")
    print(f"  Results saved to: {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='MCI Model Inference for Execution Time and Memory')
    
    # Only essential arguments
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--exec_model', type=str, help='Path to execution time PyTorch model (.pth)')
    parser.add_argument('--mem_model', type=str, help='Path to memory PyTorch model (.pth)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = MCIConfig.load_config(args.config)
    
    # Create output directory
    os.makedirs(config.data.output_dir, exist_ok=True)
    
    # Determine model paths
    if args.exec_model and args.mem_model:
        # Use provided model paths
        exec_model_path = args.exec_model
        mem_model_path = args.mem_model
    else:
        # Use default model paths from config
        exec_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_exec.pth")
        mem_model_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_mem.pth")
    
    print(f"Execution time model: {exec_model_path}")
    print(f"Memory model: {mem_model_path}")
    
    # Check if models exist
    if not os.path.exists(exec_model_path):
        raise FileNotFoundError(f"Execution time model not found: {exec_model_path}")
    if not os.path.exists(mem_model_path):
        raise FileNotFoundError(f"Memory model not found: {mem_model_path}")
    
    # Load test data by query for both models
    query_data_exec = load_test_data_by_query_exec(config)
    query_data_mem = load_test_data_by_query_mem(config)
    
    # Run dual model inference
    start_time = time.time()
    results = run_query_level_inference_dual(
        exec_onnx_model_path=exec_model_path,
        mem_onnx_model_path=mem_model_path,
        query_data_exec=query_data_exec,
        query_data_mem=query_data_mem,
        output_dir=config.data.output_dir
    )
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("DUAL MODEL INFERENCE SUMMARY")
    print("="*60)
    print(f"Execution Time Model: {exec_model_path}")
    print(f"Memory Model: {mem_model_path}")
    print(f"Total inference time: {total_time:.2f}s")
    print()
    print("EXECUTION TIME RESULTS:")
    print(f"  Queries: {results['execution_time']['num_queries']}")
    print(f"  Pipelines: {results['execution_time']['num_pipelines']}")
    print(f"  Query mean Q-error: {results['execution_time']['query_mean_q_error']:.4f}")
    print(f"  Query median Q-error: {results['execution_time']['query_median_q_error']:.4f}")
    print(f"  Pipeline mean Q-error: {results['execution_time']['pipeline_mean_q_error']:.4f}")
    print()
    print("MEMORY RESULTS:")
    print(f"  Queries: {results['memory_usage']['num_queries']}")
    print(f"  Pipelines: {results['memory_usage']['num_pipelines']}")
    print(f"  Query mean Q-error: {results['memory_usage']['query_mean_q_error']:.4f}")
    print(f"  Query median Q-error: {results['memory_usage']['query_median_q_error']:.4f}")
    print(f"  Pipeline mean Q-error: {results['memory_usage']['pipeline_mean_q_error']:.4f}")
    print(f"Results saved to: {config.data.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
