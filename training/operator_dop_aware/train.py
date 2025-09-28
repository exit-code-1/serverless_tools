import os
import sys
import time
import pandas as pd
import torch

# import dop_model
from . import model as dop_model
from utils import helpers as utils_helpers 

# 将项目根目录添加到 sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import utils
from utils import feature_engineering as utils_feat
# from ...utils import helpers as utils_help # 如果用到 split_queries 等
# from structure import dop_operators_exec, dop_operators_mem, dop_operator_features, dop_train_epochs, operator_lists
from config.structure import dop_operators_exec, dop_operators_mem, dop_operator_features, dop_train_epochs, operator_lists
all_operator_results_exec = []
all_operator_results_mem = []

# --- 1. 修改 process_and_train_curve 函数定义，增加 use_estimates 参数 ---
def process_and_train_curve(train_data, test_data, operator, test_size=0.2, epochs=100, lr=0.001, use_estimates=False):
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
    X_train, X_test, y_train, y_test = utils_feat.prepare_data(
        train_data=train_data,
        test_data = test_data,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'actual_rows', 'instance_mem', 'estimate_costs','estimate_rows',
                         'width', 'predicate_cost', 'index_cost', 'dop', 'nloops', 'query_dop', 
                         'agg_col', 'agg_width','jointype','hash_table_size', 'disk_ratio',
                         'stream_poll_time','stream_data_copy_time', 'table_names', 'up_dop', 'down_dop'],
        target_columns=['query_id', 'execution_time', 'peak_mem', 'dop'],
        use_estimates=use_estimates # <-- 传递开关
    )
    # ==================== 新增的健壮性检查 ====================
    if X_train.empty or y_train.empty:
        print(f"警告: 算子 '{operator}' 没有有效的训练数据，将跳过此算子的训练。")
        return # 直接从函数返回，不执行后续操作
    # ==========================================================
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
    start_train_time_exec = time.time()
    results_exec = None
    results_mem = None
    if operator in dop_operators_exec:
        # Call the corresponding training function dynamically
        results_exec = train_one_operator_exec(
            X_train_exec=X_train_exec_tensor, 
            X_test_exec=X_test_exec_tensor,
            y_train_exec=y_train_exec_tensor,  # 使用提取的 execution_time
            y_test_exec=y_test_exec_tensor,    # 使用提取的 execution_time
            dop_train=dop_train_tensor,       # 使用提取的 dop
            dop_test=dop_test_tensor,         # 使用提取的 dop
            operator=operator
        )
    start_train_time_mem = time.time()    
    if operator in dop_operators_mem:
        # Call the corresponding training function dynamically
        results_mem = train_one_operator_mem(
            X_train_mem=X_train_mem_tensor,
            X_test_mem=X_test_mem_tensor,
            y_train_mem=y_train_mem_tensor,   # 使用提取的 peak_mem
            y_test_mem=y_test_mem_tensor,     # 使用提取的 peak_mem
            dop_train=dop_train_tensor,       # 使用提取的 dop
            dop_test=dop_test_tensor,         # 使用提取的 dop
            operator=operator
        )
    
       # Calculate the training time
    training_time_exec = time.time() - start_train_time_exec
    training_time_mem = time.time() - start_train_time_mem
    if results_exec is not None:
        # Extract the performance metrics for execution time and memory
        performance_exec = results_exec["performance_exec"]  # Directly the MAE_error
        native_time_exec = results_exec["native_time_exec"]
        onnx_time_exec = results_exec["onnx_time_exec"]
        compare_exec = results_exec["comparisons_exec"]
        compare_exec['Comparison Type'] = 'Execution Time'

        eval_dir = f"../output/evaluations/dop_aware/operator_comparisons/"
        os.makedirs(eval_dir, exist_ok=True) # 确保目录存在
        compare_exec.to_csv(f"{eval_dir}{operator}_combined_comparison_exec.csv", index=False)
        # compare_exec.to_csv(f"tmp_result/{operator}_combined_comparison_exec.csv", index=False)
        data_to_save_exec = {
            'Operator': [operator],
            'Training Time (s)': [training_time_exec],
            'Execution Time MAE': [performance_exec['MAE_error']],
            'Execution Time Q-error': [performance_exec['Q_error']],
            'Average Execution Time': [performance_exec['average_actual_value']],
            'Native Execution Time (s)': [native_time_exec],
            'ONNX Execution Time (s)': [onnx_time_exec],
        }
        all_operator_results_exec.append(pd.DataFrame(data_to_save_exec))
    if results_mem is not None:
        performance_mem = results_mem["performance_mem"]    # Directly the MAE_error
        native_time_mem = results_mem["native_time_mem"]
        onnx_time_mem = results_mem["onnx_time_mem"]
        compare_mem = results_mem["comparisons_mem"]
        compare_mem['Comparison Type'] = 'Memory'
        eval_dir = f"../output/evaluations/operator_non_dop_aware/operator_comparisons/"
        os.makedirs(eval_dir, exist_ok=True) # 确保目录存在
        compare_mem.to_csv(f"{eval_dir}{operator}_combined_comparison_mem.csv", index=False)
        # compare_mem.to_csv(f"tmp_result/{operator}_combined_comparison_mem.csv", index=False)
        data_to_save_mem = {
            'Operator': [operator],
            'Training Time (s)': [training_time_mem],
            'Execution Time MAE': [performance_exec['MAE_error']],
            'Execution Time Q-error': [performance_exec['Q_error']],
            'Average Execution Time': [performance_exec['average_actual_value']],
            'Memory MAE': [performance_mem['MAE_error']],
            'Memory Q-error': [performance_mem['Q_error']],
            'Average Memory': [performance_mem['average_actual_value']],
            'Native Memory Time (s)': [native_time_mem],
            'ONNX Memory Time (s)': [onnx_time_mem]
        }
        # Convert to DataFrame and append to the global list
        all_operator_results_mem.append(pd.DataFrame(data_to_save_mem))
    


def train_one_operator_exec(X_train_exec, X_test_exec, y_train_exec, y_test_exec, dop_train, dop_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory

    # ==================== 新增的防御性检查 ====================
    # 检查传入的Tensor第一维度（行数）是否为0
    if X_train_exec.shape[0] == 0:
        print(f"警告: 算子 '{operator}' (exec model) 没有有效的训练样本传入，将跳过训练。")
        return None # 返回None，上层调用需要处理这种情况
    # ==========================================================
    # Train models for execution time and memory separately
    models_exec, training_times_exec = dop_model.train_exec_curve_model(X_train_exec, y_train_exec, dop_train, epochs=dop_train_epochs[operator]['exec'])
    # Predict and evaluate execution time models
    results_exec = dop_model.predict_and_evaluate_exec_curve(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test_exec,
        dop_test = dop_test,
        epsilon=epsilon,
        operator=operator,
        suffix="exec"
    )


    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],
}
def train_one_operator_mem(X_train_mem, X_test_mem, y_train_mem, y_test_mem, dop_train, dop_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory

    # Train models for execution time and memory separately
    models_mem, training_times_mem = dop_model.train_mem_curve_model(X_train_mem, y_train_mem, dop_train, epochs=dop_train_epochs[operator]['mem'])


    # Predict and evaluate memory models
    results_mem = dop_model.predict_and_evaluate_mem_curve(
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
        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }
    
    
# --- 3. 修改 train_all_operators 函数定义，增加 use_estimates 参数 ---
def train_all_operators(train_data, test_data, total_queries, train_ratio=0.8, use_estimates=False):
    # 分割查询
    # train_queries, test_queries = utils.split_queries(total_queries, train_ratio)
    
    # --- 新增的基数传播步骤 ---
    if use_estimates:
        print("!!! 训练模拟模式：正在对训练和测试数据进行基数传播预处理... !!!")
        train_data = utils_helpers.propagate_estimates_in_dataframe(train_data)
        test_data = utils_helpers.propagate_estimates_in_dataframe(test_data)
    # --- 结束新增步骤 ---
    # Train each operator and collect the results
    for operator in operator_lists:
        # --- 新增：在循环开始时进行存在性检查 ---
        if operator not in train_data['operator_type'].unique():
            print(f"信息: 算子 '{operator}' 在提供的训练数据中不存在，将完全跳过此算子。")
            continue
        # --- 检查结束 ---
        if operator in dop_operators_exec or operator in dop_operators_mem:
            print(f"\nTraining operator: {operator}")
            process_and_train_curve(
                train_data=train_data,
                test_data=test_data,
                operator=operator,
                use_estimates=use_estimates # <-- 传递开关
            )
    
    # After processing all operators, combine all results into one DataFrame
    final_results_df_exec = pd.concat(all_operator_results_exec, ignore_index=True)

    # Save the final combined DataFrame to a single CSV file
    # final_csv_file_path_exec = "tmp_result/all_operators_performance_results_exec.csv"
    final_csv_file_path_exec = "../output/evaluations/dop_aware/all_operators_performance_exec.csv"
    final_results_df_exec.to_csv(final_csv_file_path_exec, index=False)
    # final_results_df_mem = pd.concat(all_operator_results_mem, ignore_index=True)
    final_csv_file_path_mem = "../output/evaluations/dop_aware/all_operators_performance_mem.csv"
    # final_csv_file_path_mem = "tmp_result/all_operators_performance_results_mem.csv"
    # final_results_df_mem.to_csv(final_csv_file_path_mem, index=False)


    print(f"All operator results have been saved to {final_csv_file_path_exec}")