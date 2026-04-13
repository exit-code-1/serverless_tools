import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import pandas as pd
import os
import csv # <--- 新增导入
import time
import onnxruntime as ort
from training.PPM import PPM_model_NN,featurelize
from torch.utils.data import DataLoader, TensorDataset

def load_features_and_labels(feature_csv, info_csv, use_estimates=False): # <-- 增加开关
    """
    读取特征文件和查询信息文件，并匹配 query_id 和 query_dop，生成训练/测试数据。
    
    Returns:
    - X_tensor: Tensor, 特征矩阵
    - y_exec_tensor: Tensor, 执行时间标签
    - y_mem_tensor: Tensor, 内存标签
    - dop_tensor: Tensor, 并行度信息
    - query_id_list: list[int], 对应的 query_id 列表
    """
    # 读取特征数据
    # 传递开关
    query_features = featurelize.process_query_features(feature_csv, use_estimates=use_estimates)
    
    # 读取查询执行信息 (标签)
    query_info_df = pd.read_csv(info_csv, delimiter=';')
    
    X_list, y_exec_list, y_mem_list, dop_list, query_id_list = [], [], [], [], []
    
    for (query_id, query_dop), feature_vector in query_features.items():
        matching_info = query_info_df[
            (query_info_df["query_id"] == query_id) & (query_info_df["dop"] == query_dop)
        ]
        
        if not matching_info.empty:
            exec_time = matching_info["execution_time"].values[0]
            mem_usage = matching_info["query_used_mem"].values[0]
            
            X_list.append(feature_vector)
            y_exec_list.append(exec_time)
            y_mem_list.append(mem_usage)
            dop_list.append(query_dop)
            query_id_list.append(query_id)  # ✅ 加上这个

    # 转换为 Tensor
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_exec_tensor = torch.tensor(y_exec_list, dtype=torch.float32)
    y_mem_tensor = torch.tensor(y_mem_list, dtype=torch.float32)
    dop_tensor = torch.tensor(dop_list, dtype=torch.float32)

    return X_tensor, y_exec_tensor, y_mem_tensor, dop_tensor, query_id_list 

def save_results_to_csv(results, suffix):
    """
    保存训练和测试结果到 CSV 文件。

    Parameters:
    - results: dict, 评估结果
    - suffix: str, 文件后缀
    """
    result_list = []
    
    for key, value in results.items():
        dataset_type = "train" if "train" in key else "test"
        model_type = "exec" if "execution" in key else "mem"
        
        # 结果可能是 dict（包含多项指标）
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                result_list.append([dataset_type, model_type, metric, metric_value])
        else:
            result_list.append([dataset_type, model_type, "unknown_metric", value])
    
    df = pd.DataFrame(result_list, columns=["Dataset", "Model", "Metric", "Value"])
    df.to_csv(f"results_{suffix}.csv", index=False)
    print(f"结果已保存到 results_{suffix}.csv")

def train_nn_models(
    feature_csv, query_info_train_csv,
    execution_onnx_path, memory_onnx_path,
    epochs=100, lr=0.01, batch_size=32,
    use_estimates=False,
    cost_log_file=None # <-- 1. 增加 cost_log_file 参数
):
    """
    训练执行时间模型和内存模型，并导出 ONNX 文件。

    Parameters:
    - query_features_train_csv: str, 训练集特征路径
    - query_info_train_csv: str, 训练集标签路径
    - execution_onnx_path: str, 执行时间模型 ONNX 导出路径
    - memory_onnx_path: str, 内存模型 ONNX 导出路径
    """
    # 加载训练数据
    # 加载训练数据 (用5个变量接收所有返回值)
    # 传递开关
    X_train, y_exec_train, y_mem_train, dop_train, _ = load_features_and_labels(
        feature_csv, query_info_train_csv, use_estimates=use_estimates
    )
    # 初始化模型
    exec_model = PPM_model_NN.Exec_CurveFitModel(X_train.shape[1])
    mem_model = PPM_model_NN.Mem_CurveFitModel(X_train.shape[1])

    train_loader_exec = DataLoader(TensorDataset(X_train, y_exec_train, dop_train), batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader_mem = DataLoader(TensorDataset(X_train, y_mem_train, dop_train), batch_size=batch_size, shuffle=True, drop_last=True)

    exec_optimizer = optim.Adam(exec_model.parameters(), lr=lr)
    mem_optimizer = optim.Adam(mem_model.parameters(), lr=lr)
    exec_scheduler = optim.lr_scheduler.StepLR(exec_optimizer, step_size=10, gamma=0.8)
    mem_scheduler = optim.lr_scheduler.StepLR(mem_optimizer, step_size=10, gamma=0.8)

    # 训练执行时间模型
    start_time = time.time()
    for epoch in range(epochs):
        exec_model.train()
        total_loss = 0.0
        for X_batch, y_batch, dop_batch in train_loader_exec:
            exec_optimizer.zero_grad()
            pred_params = exec_model(X_batch)
            loss = PPM_model_NN.curve_exec_loss(pred_params, dop_batch, y_batch)
            if torch.any(torch.isnan(loss)):
                print(f"[Exec] NaN at epoch {epoch}.")
                break
            loss.backward()
            # Clip gradients to prevent explosion (especially important for a parameter)
            torch.nn.utils.clip_grad_norm_(exec_model.parameters(), max_norm=1.0)
            exec_optimizer.step()
            total_loss += loss.item()
        exec_scheduler.step()

    # 导出执行模型
    exec_model.eval()
    torch.onnx.export(
        exec_model,
        torch.randn(1, X_train.shape[1]),
        execution_onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )

    # 训练内存模型
    for epoch in range(epochs):
        mem_model.train()
        total_loss = 0.0
        for X_batch, y_batch, dop_batch in train_loader_mem:
            mem_optimizer.zero_grad()
            pred_params = mem_model(X_batch)
            loss = PPM_model_NN.curve_mem_loss(pred_params, dop_batch, y_batch)
            if torch.any(torch.isnan(loss)):
                print(f"[Mem] NaN at epoch {epoch}.")
                break
            loss.backward()
            # Clip gradients to prevent explosion (especially important for a parameter)
            torch.nn.utils.clip_grad_norm_(mem_model.parameters(), max_norm=1.0)
            mem_optimizer.step()
            total_loss += loss.item()
        mem_scheduler.step()
    end_time = time.time()
    duration = end_time - start_time
     # --- 2. 新增：将训练时间写入日志文件 ---
    if cost_log_file:
        with open(cost_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['model_name', 'training_time_seconds'])
            # 记录的是两个模型的总时间
            writer.writerow(['PPM-NN_total', duration]) 
        print(f"训练开销已记录到: {cost_log_file}")
    # --- 日志记录结束 ---

    # 导出内存模型
    mem_model.eval()
    torch.onnx.export(
        mem_model,
        torch.randn(1, X_train.shape[1]),
        memory_onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"training time: {duration:.2f}s")

def evaluate_nn_models(
    query_features_test_csv, query_info_test_csv,
    execution_onnx_path, memory_onnx_path,
    output_file,
    use_estimates=False # <-- 增加开关
):
    """
    使用已训练的 ONNX 模型进行评估。

    Parameters:
    - query_features_test_csv: str, 测试集特征路径
    - query_info_test_csv: str, 测试集标签路径
    - execution_onnx_path: str, 执行时间模型 ONNX 文件路径
    - memory_onnx_path: str, 内存模型 ONNX 文件路径
    """
    # 传递开关
    X_test, y_exec_test, y_mem_test, dop_test, query_ids = load_features_and_labels(
        query_features_test_csv, query_info_test_csv, use_estimates=use_estimates
    )

    PPM_model_NN.predict_and_evaluate_curve(
        onnx_time_model_path=execution_onnx_path,
        onnx_mem_model_path=memory_onnx_path,
        X_test=X_test,y_time_test=y_exec_test,y_mem_test=y_mem_test,
        dop_test=dop_test,
        output_file=output_file,
        query_ids=query_ids
    )


def main():
    """
    Main function for PPM-NN training.
    This function is called by train.py to train PPM-NN models.
    """
    import sys
    from config.main_config import TRAIN_MODES, DEFAULT_CONFIG
    from utils import create_dataset_loader, get_output_paths
    
    # Get dataset and train_mode from command line arguments or use defaults
    # If called from train.py, these will be set via environment or config
    # For now, we'll try to get them from sys.argv or use defaults
    dataset = DEFAULT_CONFIG.get('dataset', 'tpch')
    train_mode = DEFAULT_CONFIG.get('train_mode', 'exact_train')
    
    # Try to get from command line arguments if available
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    if len(sys.argv) > 2:
        train_mode = sys.argv[2]
    train_ratio = DEFAULT_CONFIG.get('train_ratio', 1.0)
    if len(sys.argv) > 3:
        train_ratio = float(sys.argv[3])
    
    print(f"Training PPM-NN models for dataset: {dataset}, train_mode: {train_mode}")
    
    # Get use_estimates from train_mode
    use_estimates = TRAIN_MODES[train_mode]['use_estimates']
    
    # Create dataset loader
    loader = create_dataset_loader(dataset)
    
    # Get file paths
    train_paths = loader.get_file_paths('train', train_ratio=train_ratio)
    test_paths = loader.get_file_paths('test', train_ratio=train_ratio)
    
    # Get output paths
    output_paths = get_output_paths(dataset, 'ppm', train_mode)
    
    # PPM models are stored in: models/{train_mode}/PPM/NN/
    ppm_model_dir = os.path.join(output_paths['model_dir'], 'PPM', 'NN')
    os.makedirs(ppm_model_dir, exist_ok=True)
    
    # Define model paths
    execution_onnx_path = os.path.join(ppm_model_dir, 'execution_time_model.onnx')
    memory_onnx_path = os.path.join(ppm_model_dir, 'memory_usage_model.onnx')
    
    # Define output file for evaluation
    output_file = os.path.join(output_paths['prediction_dir'], f'ppm_nn_predictions_{train_mode}.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Define cost log file
    cost_log_file = os.path.join(output_paths['model_dir'], 'training_cost_log.csv')
    
    # Train models
    print("Starting PPM-NN model training...")
    train_nn_models(
        feature_csv=train_paths['plan_info'],
        query_info_train_csv=train_paths['query_info'],
        execution_onnx_path=execution_onnx_path,
        memory_onnx_path=memory_onnx_path,
        epochs=100,
        lr=0.01,
        batch_size=32,
        use_estimates=use_estimates,
        cost_log_file=cost_log_file
    )
    
    # Evaluate models
    print("Starting PPM-NN model evaluation...")
    evaluate_nn_models(
        query_features_test_csv=test_paths['plan_info'],
        query_info_test_csv=test_paths['query_info'],
        execution_onnx_path=execution_onnx_path,
        memory_onnx_path=memory_onnx_path,
        output_file=output_file,
        use_estimates=use_estimates
    )
    
    print(f"PPM-NN training completed. Models saved to: {ppm_model_dir}")
    print(f"Evaluation results saved to: {output_file}")


if __name__ == "__main__":
    main()