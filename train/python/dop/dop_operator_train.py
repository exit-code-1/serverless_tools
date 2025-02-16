import os
import sys
import torch

import dop_model

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
from utils.structure import dop_operator_features

def process_and_train_curve(csv_path_pattern, operator, output_prefix, train_queries, test_queries, test_size=0.2, epochs=100, lr=0.001):
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
        csv_path_pattern=csv_path_pattern,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'estimate_costs', 'actural_rows', 'instance_mem', 
                         'width', 'predicate_cost', 'index_cost', 'dop', 'nloops', 'query_dop', 
                         'agg_col', 'agg_width','jointype','hash_table_size',
                         'stream_poll_time','stream_data_copy_time', 'table_names', 'up_dop', 'down_dop'],
        target_columns=['query_id', 'execution_time', 'peak_mem', 'dop'],
        train_queries=train_queries,
        test_queries=test_queries,
    )


    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    dop_train_tensor = torch.tensor(y_train['dop'].values, dtype=torch.float32)
    dop_test_tensor = torch.tensor(y_test['dop'].values, dtype=torch.float32)

    # 分别训练执行时间和内存预测模型
    print("Training execution time model...")
    model_exec = dop_model.train_curve_model(X_train_tensor, y_train_tensor, dop_train_tensor, epochs=epochs, lr=lr)

    print("Training memory model...")
    model_mem = dop_model.train_curve_model(X_train_tensor, y_train_tensor, dop_train_tensor, epochs=epochs, lr=lr)

    # 分别评估执行时间和内存预测模型
    print("Evaluating execution time model...")
    results_exec = dop_model.predict_and_evaluate_curve(
        model_exec, X_test_tensor, y_test_tensor, dop_test_tensor, operator, output_prefix, suffix="exec"
    )

    print("Evaluating memory model...")
    results_mem = dop_model.predict_and_evaluate_curve(
        model_mem, X_test_tensor, y_test_tensor, dop_test_tensor, operator, output_prefix, suffix="mem"
    )

    # 整理和返回结果
    results = {
        "execution_time_model": {
            "model": model_exec,
            "native_time": results_exec["native_time"],
            "onnx_time": results_exec["onnx_time"],
            "test_mse": results_exec["mse_native"],
            "predicted_params": results_exec["pred_params_native"],
        },
        "memory_model": {
            "model": model_mem,
            "native_time": results_mem["native_time"],
            "onnx_time": results_mem["onnx_time"],
            "test_mse": results_mem["mse_native"],
            "predicted_params": results_mem["pred_params_native"],
        },
    }

    print(f"\nExecution Time Test MSE: {results_exec['mse_native']:.4f}")
    print(f"Memory Test MSE: {results_mem['mse_native']:.4f}")

    return results


def train_one_operator(X_train, X_test, y_train, y_test, dop_train, dop_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = dop_operator_features[operator]['exec']  # Example features for execution time
    features_mem = dop_operator_features[operator]['mem']  # Example features for memory prediction

    # Prepare training data based on the specific features for execution time and memory
    X_train_exec = X_train[features_exec]
    X_test_exec = X_test[features_exec]
    X_train_mem = X_train[features_mem]
    X_test_mem = X_test[features_mem]

    # Train models for execution time and memory separately
    models_exec, training_times_exec = dop_model.train_curve_model(X_train_exec, y_train_exec, dop_train)
    models_mem, training_times_mem = dop_model.train_curve_model(X_train_mem, y_train_mem, dop_train)

    # Predict and evaluate execution time models
    results_exec = dop_model.predict_and_evaluate_curve(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        dop_test = dop_test,
        epsilon=epsilon,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = dop_model.predict_and_evaluate_curve(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        dop_test = dop_test,
        epsilon=epsilon,
        target_column='peak_mem',
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