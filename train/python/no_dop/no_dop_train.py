import os
import sys
import warnings
import operator_train
import time
import pandas as pd

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
from utils.structure import operators
warnings.simplefilter("ignore", category=FutureWarning)


all_operator_results = []

def process_and_train(data, operator, train_queries, test_queries, epsilon=1e-2):
    # Prepare data for both execution time and memory prediction
    X_train, X_test, y_train, y_test = utils.prepare_data(
        data=data,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'estimate_costs', 'actural_rows', 'instance_mem', 
                         'width', 'predicate_cost', 'index_cost', 'dop', 'nloops', 'query_dop', 
                         'agg_col', 'agg_width','jointype','hash_table_size',
                         'stream_poll_time','stream_data_copy_time', 'table_names', 'up_dop', 'down_dop'],  # General features
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
        'Vector Nest Loop': operator_train.train_VectorNestLoop,
        'Vector Aggregate': operator_train.train_VectorAggregate,
        'Vector Sort': operator_train.train_VectorSort,
        'Vector Hash Aggregate': operator_train.train_VectorHashAggregate,
        'Vector Sonic Hash Aggregate': operator_train.train_VectorSonicHashAggregate,
        'Vector Merge Join': operator_train.train_VectorNestLoop,
        'Vector Hash Join': operator_train.train_VectorHashJoin,
        'Vector Sonic Hash Join': operator_train.train_VectorSonicHashJoin,
        'Vector Streaming LOCAL GATHER': operator_train.train_VectorStreaming,
        'Vector Streaming LOCAL REDISTRIBUTE': operator_train.train_VectorStreaming,
        'Vector Streaming BROADCAST': operator_train.train_VectorStreaming,
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

    # Prepare data for saving to the global list
    data_to_save = {
        'Operator': [operator],
        'Training Time (s)': [training_time],
        'Execution Time MAE': [performance_exec['MAE_error']],
        'Execution Time Q-error': [performance_exec['Q_error']],
        'Average Execution Time': [performance_exec['average_actual_value']],
        'Memory MAE': [performance_mem['MAE_error']],
        'Memory Q-error': [performance_mem['Q_error']],
        'Average Memory': [performance_mem['average_actual_value']],
        'Native Execution Time (s)': [native_time_exec],
        'ONNX Execution Time (s)': [onnx_time_exec],
        'Native Memory Time (s)': [native_time_mem],
        'ONNX Memory Time (s)': [onnx_time_mem]
    }

    # Convert to DataFrame and append to the global list
    df_to_save = pd.DataFrame(data_to_save)
    all_operator_results.append(df_to_save)

    # Optionally write the comparison results to the same CSV file for execution time and memory
    compare_exec = results["comparisons_exec"]
    compare_mem = results["comparisons_mem"]
    compare_exec['Comparison Type'] = 'Execution Time'
    compare_mem['Comparison Type'] = 'Memory'
    
    # Concatenate both comparison dataframes into one dataframe
    comparisons_combined = pd.concat([compare_exec, compare_mem], axis=0)
    comparisons_combined['Operator'] = operator  # Add operator column to identify operator in the combined file
    comparisons_combined.to_csv(f"tmp_result/{operator}_combined_comparison.csv", index=False)

    return results

def train_all_operators(data, total_queries=200, train_ratio=0.8):
    # 分割查询
    train_queries, test_queries = utils.split_queries(total_queries, train_ratio)
    
    # 保存查询分割结果
    utils.save_query_split(train_queries, test_queries, "tmp_result/query_split.csv")
    
    
    # Train each operator and collect the results
    for operator in operators:
        print(f"\nTraining operator: {operator}")
        process_and_train(
            data=data,
            operator=operator,
            train_queries=train_queries,
            test_queries=test_queries
        )
    
    # After processing all operators, combine all results into one DataFrame
    final_results_df = pd.concat(all_operator_results, ignore_index=True)

    # Save the final combined DataFrame to a single CSV file
    final_csv_file_path = "tmp_result/all_operators_performance_results.csv"
    final_results_df.to_csv(final_csv_file_path, index=False)

    print(f"All operator results have been saved to {final_csv_file_path}")
