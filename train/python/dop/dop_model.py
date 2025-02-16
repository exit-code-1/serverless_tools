import os
import sys
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils

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
    relative_error = torch.abs((true_time - pred_time))
    loss = torch.mean(relative_error)
    return loss

# 训练模型
def train_curve_model(X_train, y_train, dop_train, epochs=50, lr=0.05):
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

def predict_and_evaluate_curve(
    model, X_test, y_test, dop_test, epsilon=1e-2, target_column='execution_time', operator=None, suffix=""
):
    """
    使用模型进行预测并评估性能，同时保存 ONNX 模型并比较预测时间。

    Parameters:
    - model: CurveFitModel, 训练好的 PyTorch 模型
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 实际执行时间
    - dop_test: Tensor, 测试并行度
    - operator: str, 操作符类型，用于命名保存的 ONNX 文件
    - output_prefix: str, 保存 ONNX 模型的前缀路径
    - suffix: str, ONNX 模型文件名后缀

    Returns:
    - results: dict, 包含预测性能和时间对比
    """
    model.eval()

    # 原生 PyTorch 模型预测
    start_time = time.time()
    with torch.no_grad():
        pred_params = model(X_test)
        a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]
        predictions_native = b * (dop_test ** a) + c
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator:
        onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/dop/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
        onnx_dir = os.path.dirname(onnx_path)
        os.makedirs(onnx_dir, exist_ok=True)

        # 导出 ONNX 模型
        dummy_input = torch.randn(X_test.size(0), X_test.size(1))
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
        print(f"ONNX model saved to: {onnx_path}")

    # 使用 ONNX Runtime 进行预测
    predictions_onnx = None
    onnx_time = None
    if onnx_path:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        start_time = time.time()
        predictions_onnx = session.run(None, {input_name: X_test.numpy().astype(np.float32)})[0]
        onnx_time = time.time() - start_time

     # Calculate standard mean absolute error (MAE) for native predictions
    mae_native = torch.mean(torch.abs(y_test[target_column] - predictions_native))

    # Calculate Prediction Accuracy for native model
    Q_error = torch.mean(
         torch.abs((y_test[target_column] - predictions_native) / (y_test[target_column] + epsilon))
    )

    # Create comparison DataFrame
    comparisons = pd.DataFrame({
        'query_id': y_test['query_id'].values,
        'Actual': y_test[target_column].values,
        'Predicted_Native': predictions_native,
        'Difference_Native': y_test[target_column].values - predictions_native,
    })

    # Calculate Prediction Accuracy for ONNX model
    time_accuracy_onnx = None
    if onnx_time is not None:
        time_accuracy_onnx = np.mean(
            (1 - np.abs(y_test[target_column] - predictions_native) / (y_test[target_column] + epsilon)) * 100
        )

    # Calculate the average of the actual target values
    avg_actual_value = np.mean(y_test[target_column])

    # Print model prediction times
    print(f"Native model prediction time: {native_time:.6f} seconds")
    if onnx_time is not None:
        print(f"ONNX model prediction time: {onnx_time:.6f} seconds")

    # Organize the performance metrics into one dictionary
    performance = {
        "metrics": {
            "MAE_error": mae_native,
            "Q_error": Q_error,
            "average_actual_value": avg_actual_value
        },
        "comparisons": comparisons,
        "native_time": native_time,
        "onnx_time": onnx_time,
        "onnx_accuracy": time_accuracy_onnx
    }

    return performance



