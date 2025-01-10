import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import tools.serverless_tools.train.python.utils.utils as utils

# 定义网络模型
class CurveFitModel(nn.Module):
    def __init__(self, input_dim):
        super(CurveFitModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出 a, b, c
        )

    def forward(self, x):
        return self.fc(x)


# 自定义损失函数
def curve_loss(pred_params, dop, true_time, epsilon=1e-2):
    """
    计算曲线拟合损失。

    Parameters:
    - pred_params: Tensor, 预测的 a, b, c 参数
    - dop: Tensor, 并行度 dop 值
    - true_time: Tensor, 实际执行时间
    - epsilon: float, 避免 log(0)

    Returns:
    - loss: Tensor, 损失值
    """
    a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]
    pred_time = b * (dop ** a) + c
    loss = torch.mean((true_time - pred_time) ** 2)
    return loss


# 训练模型
def train_curve_model(X_train, y_train, dop_train, epochs=100, lr=0.001):
    """
    训练用于预测曲线参数的模型。

    Parameters:
    - X_train: Tensor, 特征
    - y_train: Tensor, 实际执行时间
    - dop_train: Tensor, 并行度
    - epochs: int, 训练轮数
    - lr: float, 学习率

    Returns:
    - model: CurveFitModel, 训练后的模型
    """
    input_dim = X_train.shape[1]
    model = CurveFitModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        pred_params = model(X_train)
        loss = curve_loss(pred_params, dop_train, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


# 预测与评估
def predict_and_evaluate_curve(model, X_test, y_test, dop_test):
    """
    使用模型进行预测并评估性能。

    Parameters:
    - model: CurveFitModel, 训练好的模型
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 实际执行时间
    - dop_test: Tensor, 测试并行度

    Returns:
    - pred_params: Tensor, 预测的 a, b, c 参数
    - pred_time: Tensor, 预测的执行时间
    - mse: float, 均方误差
    """
    model.eval()
    with torch.no_grad():
        pred_params = model(X_test)
        a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]
        pred_time = b * (dop_test ** a) + c
        mse = torch.mean((y_test - pred_time) ** 2).item()

    return pred_params, pred_time, mse


def prepare_and_train_curve(csv_path_pattern, operator, train_queries, test_queries, epsilon=1e-2):
    """
    数据处理、训练模型并评估性能，复用 utils.prepare_data 函数。

    Parameters:
    - csv_path_pattern: str, 数据文件路径模式
    - operator: str, 操作符类型
    - train_queries: list, 训练查询
    - test_queries: list, 测试查询
    - epsilon: float, 避免除零

    Returns:
    - models: dict, 训练后的模型 (执行时间和峰值内存)
    - performance: dict, 各模型的预测结果与评估性能
    """
    # 调用 utils.prepare_data 预处理数据
    X_train, X_test, y_train, y_test = utils.prepare_data(
        csv_path_pattern=csv_path_pattern,
        operator=operator,
        feature_columns=['l_input_rows', 'actural_rows', 'instance_mem', 'width'],
        target_columns=['query_id', 'execution_time', 'peak_mem', 'dop'],  # 包含 dop 列
        train_queries=train_queries,
        test_queries=test_queries,
        epsilon=epsilon
    )

    # 提取并行度和目标值
    dop_train = torch.tensor(y_train['dop'].values, dtype=torch.float32)
    dop_test = torch.tensor(y_test['dop'].values, dtype=torch.float32)
    y_train_exec = torch.tensor(y_train['execution_time'].values, dtype=torch.float32)
    y_test_exec = torch.tensor(y_test['execution_time'].values, dtype=torch.float32)
    y_train_mem = torch.tensor(y_train['peak_mem'].values, dtype=torch.float32)
    y_test_mem = torch.tensor(y_test['peak_mem'].values, dtype=torch.float32)

    # 转换为 Tensor 格式
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)

    # 训练曲线模型
    print("\n===== Training Execution Time Model =====")
    exec_model = train_curve_model(X_train, y_train_exec, dop_train)

    print("\n===== Training Peak Memory Model =====")
    mem_model = train_curve_model(X_train, y_train_mem, dop_train)

    # 预测与评估
    print("\n===== Evaluating Execution Time Model =====")
    exec_pred_params, exec_pred_time, exec_mse = predict_and_evaluate_curve(exec_model, X_test, y_test_exec, dop_test)

    print("\n===== Evaluating Peak Memory Model =====")
    mem_pred_params, mem_pred_time, mem_mse = predict_and_evaluate_curve(mem_model, X_test, y_test_mem, dop_test)

    # 返回结果
    return {
        "models": {"execution_time": exec_model, "peak_mem": mem_model},
        "performance": {
            "execution_time": {"predicted_params": exec_pred_params, "predicted_time": exec_pred_time, "mse": exec_mse},
            "peak_mem": {"predicted_params": mem_pred_params, "predicted_time": mem_pred_time, "mse": mem_mse},
        }
    }
