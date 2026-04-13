from typing import Optional

from .predict_queries import run_inference as _run_inference_low_level

__all__ = ['run_inference', 'run_ppm_inference', 'run_query_level_inference']

def run_inference(
    dataset: str,
    train_mode: str,
    use_estimates_mode: bool = True,
    train_ratio: Optional[float] = None,
    seed: int = 42,
) -> bool:
    """
    Run operator-level inference (wrapper function).
    
    Args:
        dataset: Dataset name (e.g., 'tpch', 'tpcds')
        train_mode: Training mode (e.g., 'exact_train', 'estimated_train')
        use_estimates_mode: Whether to use estimates
        
    Returns:
        bool: True if inference completed successfully, False otherwise
    """
    import os
    from utils import create_dataset_loader, get_output_paths, get_model_paths
    
    try:
        print(f"Starting operator-level inference for dataset: {dataset}, train_mode: {train_mode}")
        
        # Use unified data loader
        loader = create_dataset_loader(dataset)
        
        # Get output paths
        output_paths = get_output_paths(dataset, 'inference', train_mode)
        
        # Get model paths - inference needs both dop_aware and non_dop_aware models
        model_paths = get_model_paths(dataset, train_mode, 'pipeline')
        
        # Get test file paths
        test_paths = loader.get_file_paths('test', train_ratio=train_ratio, seed=seed)
        
        # Define output file
        output_csv_path = os.path.join(
            output_paths['prediction_dir'], 
            f"operator_level_{train_mode}_inference_results.csv"
        )
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Execute inference using low-level function
        _run_inference_low_level(
            plan_csv_path=test_paths['plan_info'],
            query_csv_path=test_paths['query_info'],
            output_csv_path=output_csv_path,
            no_dop_model_dir=model_paths['non_dop_aware_dir'],
            dop_model_dir=model_paths['dop_aware_dir'],
            use_estimates=use_estimates_mode
        )
        
        print(f"✅ Operator-level inference completed. Results saved to: {output_csv_path}")
        return True
        
    except Exception as e:
        print(f"❌ Operator-level inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_ppm_inference(
    dataset: str,
    train_mode: str,
    ppm_type: str = 'NN',
    train_ratio: Optional[float] = None,
    seed: int = 42,
) -> bool:
    """
    Run PPM (Pipeline Performance Model) inference.
    
    Args:
        dataset: Dataset name (e.g., 'tpch', 'tpcds')
        train_mode: Training mode (e.g., 'exact_train', 'estimated_train')
        ppm_type: PPM model type ('NN' or 'GNN')
        
    Returns:
        bool: True if inference completed successfully, False otherwise
    """
    import sys
    import os
    from config.main_config import TRAIN_MODES
    from utils import create_dataset_loader, get_output_paths
    
    try:
        print(f"Starting PPM-{ppm_type} inference for dataset: {dataset}, train_mode: {train_mode}")
        
        # Get use_estimates from train_mode
        use_estimates = TRAIN_MODES[train_mode]['use_estimates']
        
        # Create dataset loader
        loader = create_dataset_loader(dataset)
        
        # Get test file paths
        test_paths = loader.get_file_paths('test', train_ratio=train_ratio, seed=seed)
        
        # Get output paths
        output_paths = get_output_paths(dataset, 'ppm', train_mode)
        
        # PPM models are stored in: models/{train_mode}/PPM/{ppm_type}/
        ppm_model_dir = os.path.join(output_paths['model_dir'], 'PPM', ppm_type)
        
        # Define model paths
        execution_onnx_path = os.path.join(ppm_model_dir, 'execution_time_model.onnx')
        memory_onnx_path = os.path.join(ppm_model_dir, 'memory_usage_model.onnx')
        
        # Check if models exist
        if not os.path.exists(execution_onnx_path):
            print(f"❌ Execution time model not found: {execution_onnx_path}")
            return False
        if not os.path.exists(memory_onnx_path):
            print(f"❌ Memory model not found: {memory_onnx_path}")
            return False
        
        # Define output file for evaluation
        output_file = os.path.join(output_paths['prediction_dir'], f'ppm_{ppm_type.lower()}_predictions_{train_mode}.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Import and call appropriate evaluation function
        if ppm_type == 'NN':
            from training.PPM.NN_train import evaluate_nn_models
            print("Running PPM-NN inference...")
            evaluate_nn_models(
                query_features_test_csv=test_paths['plan_info'],
                query_info_test_csv=test_paths['query_info'],
                execution_onnx_path=execution_onnx_path,
                memory_onnx_path=memory_onnx_path,
                output_file=output_file,
                use_estimates=use_estimates
            )
        elif ppm_type == 'GNN':
            from training.PPM.GNN_train import evaluate_gnn_models
            print("Running PPM-GNN inference...")
            evaluate_gnn_models(
                query_features_test_csv=test_paths['plan_info'],
                query_info_test_csv=test_paths['query_info'],
                execution_onnx_path=execution_onnx_path,
                memory_onnx_path=memory_onnx_path,
                output_file=output_file,
                use_estimates=use_estimates
            )
        else:
            print(f"❌ Unknown PPM type: {ppm_type}. Must be 'NN' or 'GNN'")
            return False
        
        print(f"✅ PPM-{ppm_type} inference completed. Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ PPM inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_query_level_inference(
    dataset: str,
    train_mode: str,
    train_ratio: Optional[float] = None,
    seed: int = 42,
) -> bool:
    """
    Run query-level inference using XGBoost models.
    
    Args:
        dataset: Dataset name (e.g., 'tpch', 'tpcds')
        train_mode: Training mode (e.g., 'exact_train', 'estimated_train')
        
    Returns:
        bool: True if inference completed successfully, False otherwise
    """
    import os
    from config.main_config import TRAIN_MODES
    from utils import create_dataset_loader, get_output_paths
    
    try:
        print(f"Starting query-level inference for dataset: {dataset}, train_mode: {train_mode}")
        
        # Get use_estimates from train_mode
        use_estimates = TRAIN_MODES[train_mode]['use_estimates']
        
        # Create dataset loader
        loader = create_dataset_loader(dataset)
        
        # Get test file paths
        test_paths = loader.get_file_paths('test', train_ratio=train_ratio, seed=seed)
        
        # Get output paths
        output_paths = get_output_paths(dataset, 'query_level', train_mode)
        
        # Query-level models are stored in: models/{train_mode}/query_level/
        query_level_model_dir = os.path.join(output_paths['model_dir'], 'query_level')
        
        # Define model paths
        execution_onnx_path = os.path.join(query_level_model_dir, 'execution_time_model.onnx')
        memory_onnx_path = os.path.join(query_level_model_dir, 'memory_usage_model.onnx')
        
        # Check if models exist
        if not os.path.exists(execution_onnx_path):
            print(f"❌ Execution time model not found: {execution_onnx_path}")
            return False
        if not os.path.exists(memory_onnx_path):
            print(f"❌ Memory model not found: {memory_onnx_path}")
            return False
        
        # Define output file for evaluation
        output_file = os.path.join(output_paths['prediction_dir'], f'query_level_predictions_{train_mode}.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Import and call evaluation function
        from training.query_level.train import test_onnx_xgboost
        print("Running query-level inference...")
        test_onnx_xgboost(
            execution_onnx_path=execution_onnx_path,
            memory_onnx_path=memory_onnx_path,
            feature_csv=test_paths['plan_info'],
            true_val_csv=test_paths['query_info'],
            output_file=output_file,
            use_estimates=use_estimates
        )
        
        print(f"✅ Query-level inference completed. Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Query-level inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

