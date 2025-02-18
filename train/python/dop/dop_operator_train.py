import os
import sys
import time
import pandas as pd
import torch

import dop_model

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
from utils.structure import dop_operators, dop_operator_features, dop_train_epochs

all_operator_results = []

def process_and_train_curve(data, operator, train_queries, test_queries, test_size=0.2, epochs=100, lr=0.001):
    """
    数据加载、处理和训练执行时间和内存预测的曲线拟合模型。

    Parameters:
    - csv_path_pattern: str, glob 模式，用于找到 plan_info.csv 文件
    - operator: str, 操作符类型，用于筛选数据
    - output_prefix: str, 用于保存模型的前缀
    - train_queries: list of int, 用于训练的查询 ID
    - test_queries: list of int, 用于测试的查询 ID
    - test_size: float, 测试集比例（仅用于内部划分）
    - epochs: int, 训练轮数
    - lr: float, 学习率

    Returns:
    - results: dict, 包含执行时间和内存模型及评估结果
    """
    # 使用 utils.prepare_data 加载并处理数据
    X_train, X_test, y_train, y_test = utils.prepare_data(
        data=data,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'estimate_costs', 'actual_rows', 'instance_mem', 
                         'width', 'predicate_cost', 'index_cost', 'dop', 'nloops', 'query_dop', 
                         'agg_col', 'agg_width','jointype','hash_table_size',
                         'stream_poll_time','stream_data_copy_time', 'table_names', 'up_dop', 'down_dop'],
        target_columns=['query_id', 'execution_time', 'peak_mem', 'dop'],
        train_queries=train_queries,
        test_queries=test_queries,
    )

    # 提取目标列
    y_train_exec = y_train['execution_time']  # 提取 execution_time 列
    y_test_exec = y_test['execution_time']    # 提取 execution_time 列
    y_train_mem = y_train['peak_mem']         # 提取 peak_mem 列
    y_test_mem = y_test['peak_mem']           # 提取 peak_mem 列
    dop_train = y_train['dop']                # 提取 dop 列
    dop_test = y_test['dop']                  # 提取 dop 列
    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = dop_operator_features[operator]['exec']  # Example features for execution time
    features_mem = dop_operator_features[operator]['mem']  # Example features for memory prediction

    # 转换为 PyTorch 张量
    X_train_exec_tensor = torch.tensor(X_train[features_exec].values, dtype=torch.float32)
    X_test_exec_tensor = torch.tensor(X_test[features_exec].values, dtype=torch.float32)
    X_train_mem_tensor = torch.tensor(X_train[features_mem].values, dtype=torch.float32)
    X_test_mem_tensor = torch.tensor(X_test[features_mem].values, dtype=torch.float32)
    y_train_exec_tensor = torch.tensor(y_train_exec.values, dtype=torch.float32)
    y_test_exec_tensor = torch.tensor(y_test_exec.values, dtype=torch.float32)
    y_train_mem_tensor = torch.tensor(y_train_mem.values, dtype=torch.float32)
    y_test_mem_tensor = torch.tensor(y_test_mem.values, dtype=torch.float32)
    dop_train_tensor = torch.tensor(dop_train.values, dtype=torch.float32)
    dop_test_tensor = torch.tensor(dop_test.values, dtype=torch.float32)

    # Record the start time for training
    start_train_time = time.time()

    # Check if the operator exists in the mapping
    if operator in dop_operators:
        # Call the corresponding training function dynamically
        results = train_one_operator(
            X_train_exec=X_train_exec_tensor, 
            X_test_exec=X_test_exec_tensor,
            X_train_mem=X_train_mem_tensor,
            X_test_mem=X_test_mem_tensor,
            y_train_exec=y_train_exec_tensor,  # 使用提取的 execution_time
            y_test_exec=y_test_exec_tensor,    # 使用提取的 execution_time
            y_train_mem=y_train_mem_tensor,   # 使用提取的 peak_mem
            y_test_mem=y_test_mem_tensor,     # 使用提取的 peak_mem
            dop_train=dop_train_tensor,       # 使用提取的 dop
            dop_test=dop_test_tensor,         # 使用提取的 dop
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


def train_one_operator(X_train_exec, X_train_mem, X_test_exec, X_test_mem, y_train_exec, y_train_mem, y_test_exec, y_test_mem, dop_train, dop_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory

    # Train models for execution time and memory separately
    models_exec, training_times_exec = dop_model.train_curve_model(X_train_exec, y_train_exec, dop_train, epochs=dop_train_epochs[operator]['exec'])
    models_mem, training_times_mem = dop_model.train_curve_model(X_train_mem, y_train_mem, dop_train, epochs=dop_train_epochs[operator]['mem'])

    # Predict and evaluate execution time models
    results_exec = dop_model.predict_and_evaluate_curve(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test_exec,
        dop_test = dop_test,
        epsilon=epsilon,
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = dop_model.predict_and_evaluate_curve(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test_mem,
        dop_test = dop_test,
        epsilon=epsilon,
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }
    
    
def train_all_operators(data, total_queries, train_ratio=0.8):
    # 分割查询
    train_queries, test_queries = utils.split_queries(total_queries, train_ratio)
    
    # 保存查询分割结果
    utils.save_query_split(train_queries, test_queries, "tmp_result/query_split.csv")
    
    
    # Train each operator and collect the results
    for operator in dop_operators:
        print(f"\nTraining operator: {operator}")
        process_and_train_curve(
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