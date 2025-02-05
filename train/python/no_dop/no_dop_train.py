import os
import numpy as np
import pandas as pd
import sys
import operator_train

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils



def process_and_train(csv_path_pattern, operator, train_queries, test_queries, epsilon=1e-2):
    """
    Process data, train models, and evaluate performance. Execution time and memory are handled separately,
    with distinct feature sets for each task.

    Parameters:
    - csv_path_pattern: str, glob pattern to find plan_info.csv files
    - operator: str, the operator type to filter data
    - output_prefix: str, prefix for saving ONNX models
    - train_queries: list of int, query IDs to use for training
    - test_queries: list of int, query IDs to use for testing
    - epsilon: float, a small constant to avoid division by zero in Q-error calculations

    Returns:
    - results: dict, containing models, ONNX paths, performance, and comparisons
    """
    # Prepare data for both execution time and memory prediction
    X_train, X_test, y_train, y_test = utils.prepare_data(
        csv_path_pattern=csv_path_pattern,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'actural_rows', 'instance_mem', 'width', 'predicate_cost', 'dop', 'nloops', 'query_dop'],  # General features
        target_columns=['query_id', 'execution_time', 'peak_mem'],
        train_queries=train_queries,
        test_queries=test_queries,
        epsilon=epsilon
    )

    # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = ['l_input_rows', 'r_input_rows', 'actural_rows', 'width', 'predicate_cost', 'dop']  # Example features for execution time
    features_mem = ['l_input_rows', 'r_input_rows', 'actural_rows', 'width', 'dop']  # Example features for memory prediction

    # Prepare training data based on the specific features for execution time and memory
    X_train_exec = X_train[features_exec]
    X_test_exec = X_test[features_exec]
    X_train_mem = X_train[features_mem]
    X_test_mem = X_test[features_mem]

    # Rename features for compatibility with ONNX
    X_train_exec.columns = [f"f{i}" for i in range(X_train_exec.shape[1])]
    X_test_exec.columns = [f"f{i}" for i in range(X_test_exec.shape[1])]
    X_train_mem.columns = [f"f{i}" for i in range(X_train_mem.shape[1])]
    X_test_mem.columns = [f"f{i}" for i in range(X_test_mem.shape[1])]

    # Train models for execution time and memory separately
    models_exec, training_times_exec = operator_train.train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = operator_train.train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = operator_train.predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = operator_train.predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["performance_native"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["performance_native"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }
