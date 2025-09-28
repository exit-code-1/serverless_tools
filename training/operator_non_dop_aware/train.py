import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from sklearn.svm import SVR
from skl2onnx import convert_sklearn
import onnxruntime as ort
import sys

# 将项目根目录添加到 sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import utils
from utils import feature_engineering as utils_feat
from utils import helpers as utils_help # 如果用到 split_queries
# from structure import no_dop_operators_exec,no_dop_operators_mem, no_dop_operator_features, operator_lists
from config.structure import no_dop_operators_exec, no_dop_operators_mem, operator_lists
from .model import train_one_operator_exec, train_one_operator_mem
from utils import helpers as utils_helpers 

all_operator_results_exec = []
all_operator_results_mem = []

# --- 1. 修改 process_and_train 函数定义，增加 use_estimates 参数 ---
def process_and_train(train_data, test_data, operator, epsilon=1e-2, use_estimates=False):
    # Prepare data for both execution time and memory prediction
    X_train, X_test, y_train, y_test = utils_feat.prepare_data(
        train_data=train_data,
        test_data=test_data,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'estimate_costs', 'actual_rows', 'instance_mem', 'estimate_rows',
                         'width', 'predicate_cost', 'index_cost', 'dop', 'nloops', 'query_dop', 
                         'agg_col', 'agg_width','jointype','hash_table_size', 'disk_ratio',
                         'stream_poll_time','stream_data_copy_time', 'table_names', 'up_dop', 'down_dop'],  # General features
        target_columns=['query_id', 'execution_time', 'peak_mem'],
        epsilon=epsilon,
        use_estimates=use_estimates
    )
    
    # Record the start time for training
    
    results_exec = None
    results_mem = None
# Check if the operator exists in the mapping
    start_train_time_exec = time.time()
    if operator in no_dop_operators_exec:
        # Call the corresponding training function dynamically
        results_exec = train_one_operator_exec(
            X_train=X_train, 
            X_test=X_test,
            y_train=y_train,  
            y_test=y_test,    
            operator=operator
        )
    start_train_time_mem = time.time()    
    if operator in no_dop_operators_mem:
        # Call the corresponding training function dynamically
        results_mem = train_one_operator_mem(
            X_train=X_train, 
            X_test=X_test,
            y_train=y_train,  
            y_test=y_test,    
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
        eval_dir = f"../output/evaluations/operator_non_dop_aware/operator_comparisons/"
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
            'Memory MAE': [performance_mem['MAE_error']],
            'Memory Q-error': [performance_mem['Q_error']],
            'Average Memory': [performance_mem['average_actual_value']],
            'Native Memory Time (s)': [native_time_mem],
            'ONNX Memory Time (s)': [onnx_time_mem]
        }
        all_operator_results_mem.append(pd.DataFrame(data_to_save_mem))

    # Optionally write the comparison results to the same CSV file for execution time and memory

def train_all_operators(train_data, test_data, total_queries, train_ratio=0.8, use_estimates=False):
    # # 分割查询
    # train_queries, test_queries = utils.split_queries(total_queries, train_ratio)
    
    # # 保存查询分割结果
    # utils.save_query_split(train_queries, test_queries, "tmp_result/query_split.csv")
    
    # 读取包含 query_id 和 split 信息的文件
    split_info_df = pd.read_csv('/home/zhy/opengauss/tools/serverless_tools_cht/train/python/dop/tmp_result/query_split.csv')

    # 只保留前 200 行的数据
    split_info = split_info_df[['query_id', 'split']]

    # 获取原始的 split 信息的 query_id 和 split 列
    test_queries = split_info[split_info['split'] == 'test']['query_id']

    # 将扩展后的测试查询的 query_id 转换为 DataFrame
    train_queries = split_info[split_info['split'] == 'train']['query_id']
    
    # --- 新增的基数传播步骤 ---
    if use_estimates:
        print("!!! 训练模拟模式：正在对训练和测试数据进行基数传播预处理... !!!")
        train_data = utils_helpers.propagate_estimates_in_dataframe(train_data)
        test_data = utils_helpers.propagate_estimates_in_dataframe(test_data)
    # --- 结束新增步骤 ---
        
    # Train each operator and collect the results
    for operator in operator_lists:
        # --- 新增：在循环开始时进行存在性检查 ---
        # 检查当前算子是否在训练数据中实际存在
        if operator not in train_data['operator_type'].unique():
            print(f"信息: 算子 '{operator}' 在提供的训练数据中不存在，将完全跳过此算子。")
            continue # 跳到下一个算子
        # --- 检查结束 ---
        if operator in no_dop_operators_exec or operator in no_dop_operators_mem:
            print(f"\nTraining operator: {operator}")
            process_and_train(
                train_data=train_data,
                test_data=test_data,
                operator=operator,
                use_estimates=use_estimates # <-- 传递开关
            )
    
    # After processing all operators, combine all results into one DataFrame
    final_results_df_exec = pd.concat(all_operator_results_exec, ignore_index=True)

    # Save the final combined DataFrame to a single CSV file
    # final_csv_file_path_exec = "tmp_result/all_operators_performance_results_exec.csv"
    final_csv_file_path_exec = "../output/evaluations/operator_non_dop_aware/all_operators_performance_exec.csv"
    final_results_df_exec.to_csv(final_csv_file_path_exec, index=False)
    final_results_df_mem = pd.concat(all_operator_results_mem, ignore_index=True)

    # final_csv_file_path_mem = "tmp_result/all_operators_performance_results_mem.csv"
    final_csv_file_path_mem = "../output/evaluations/operator_non_dop_aware/all_operators_performance_mem.csv"
    final_results_df_mem.to_csv(final_csv_file_path_mem, index=False)


    print(f"All operator results have been saved to {final_csv_file_path_exec}")