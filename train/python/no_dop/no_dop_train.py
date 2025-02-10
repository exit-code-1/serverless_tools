import os
import sys
import warnings
import operator_train
import time

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
warnings.simplefilter("ignore", category=FutureWarning)



import pandas as pd

def process_and_train(data, operator, train_queries, test_queries, epsilon=1e-2):
    # Prepare data for both execution time and memory prediction
    X_train, X_test, y_train, y_test = utils.prepare_data(
        data=data,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'actural_rows', 'instance_mem', 
                         'width', 'predicate_cost', 'dop', 'nloops', 'query_dop', 
                         'agg_col', 'agg_width','jointype','hash_table_size',
                         'stream_poll_time','stream_data_copy_time', 'table_names'],  # General features
        target_columns=['query_id', 'execution_time', 'peak_mem'],
        train_queries=train_queries,
        test_queries=test_queries,
        epsilon=epsilon
    )
    
    # Record the start time for training
    start_train_time = time.time()
    
    # Define a mapping of operator types to their corresponding training functions
    operator_to_train_function = {
        'CStore Scan': operator_train.train_CStoreScan,
        'CStore Index Scan': operator_train.train_CStoreIndexScan,
        'CTE Scan': operator_train.train_CTEScan,
        'Vector Materialize': operator_train.train_VectorMaterialize,
        'Vector NestLoop': operator_train.train_VectorNestLoop,
        'Vector Aggregate': operator_train.train_VectorAggregate,
        'Vector Sort': operator_train.train_VectorSort,
        'Vector Hash Aggregate': operator_train.train_VectorHashAggregate,
        'Vector Sonic Hash Aggregate': operator_train.train_VectorSonicHashAggregate,
        'Vector Hash Join': operator_train.train_VectorHashJoin,
        'Vector Sonic Hash Join': operator_train.train_VectorSonicHashJoin,
        # Add more operators and their corresponding functions here as needed
    }

    # Check if the operator exists in the mapping
    if operator in operator_to_train_function:
        # Call the corresponding training function dynamically
        results = operator_to_train_function[operator](
            X_train=X_train, 
            X_test=X_test,
            y_train=y_train, 
            y_test=y_test, 
            operator=operator
        )
    else:
        print(f"Error: No training function defined for operator '{operator}'")
        return None
    
    # Calculate the training time
    training_time = time.time() - start_train_time
    
    # Extract the performance metrics for execution time and memory
    performance_exec = results["performance_exec"]  # Directly the MAE_error
    performance_mem = results["performance_mem"]    # Directly the MAE_error
    
    # Get prediction times
    native_time_exec = results["native_time_exec"]
    onnx_time_exec = results["onnx_time_exec"]
    native_time_mem = results["native_time_mem"]
    onnx_time_mem = results["onnx_time_mem"]

    # Get predictions for execution time and memory from the results
    compare_exec = results["comparisons_exec"] # You may want to extract actual predicted values
    compare_mem = results["comparisons_mem"]   # Same for memory

    
    # Print the comparison (or log it for later review)
    print("\nExecution Time Comparison:")
    print(compare_exec.head())  # Print top 5 rows or adjust as necessary
    
    print("\nMemory Comparison:")
    print(compare_mem.head())  # Print top 5 rows or adjust as necessary
    
    # Optionally write the comparison results to CSV files
    compare_exec.to_csv(f"tmp_result/{operator}_execution_time_comparison.csv", index=False)
    compare_mem.to_csv(f"tmp_result/{operator}_memory_comparison.csv", index=False)
    
    # Print the results
    print(f"\nOperator: {operator}")
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Execution time MAE: {performance_exec['MAE_error']:.4f}")
    print(f"Execution time Q-error: {performance_exec['Q_error']:.4f}")
    print(f"Average Execution time: {performance_exec['average_actual_value']:.4f}")
    print(f"Memory MAE: {performance_mem['MAE_error']:.4f}")
    print(f"Memory Q-error: {performance_mem['Q_error']:.4f}")
    print(f"Average Memory: {performance_mem['average_actual_value']:.4f}")
    print(f"Native execution time: {native_time_exec:.4f} seconds")
    print(f"ONNX execution time: {onnx_time_exec:.4f} seconds")
    print(f"Native memory time: {native_time_mem:.4f} seconds")
    print(f"ONNX memory time: {onnx_time_mem:.4f} seconds")
    
    return results


        


def train_all_operators(data):
    train_queries = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 11, 13, 15, 17, 18, 19, 22]
    test_queries = [8, 14, 16, 20, 21]
    
    operators = [
        "CStore Scan",
        "CStore Index Scan",
        "Vector Materialize",
        "Vector Nest Loop",
        "Vector Aggregate",
        "Vector Sort",
        "Vector Hash Aggregate",
        "Vector Sonic Hash Aggregate",
        "Vector Hash Join",
        "Vector Sonic Hash Join"
    ]
    
    for operator in operators:
        print(f"\nTraining operator: {operator}")
        results = process_and_train(
            data=data,
            operator=operator,
            train_queries=train_queries,
            test_queries=test_queries
        )
