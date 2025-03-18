import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import pandas as pd
import os
import time
import onnxruntime as ort
import PPM_model_NN
import featurelize
from torch.utils.data import DataLoader, TensorDataset

def load_features_and_labels(feature_csv, info_csv):
    """
    读取特征文件和查询信息文件，并匹配 query_id 和 query_dop，生成训练/测试数据。
    
    Parameters:
    - feature_csv: str, 特征向量的 CSV 文件路径
    - info_csv: str, 查询标签的 CSV 文件路径 (包含 execution_time, query_used_mem, dop)
    
    Returns:
    - X_tensor: Tensor, 特征矩阵
    - y_exec_tensor: Tensor, 执行时间标签
    - y_mem_tensor: Tensor, 内存标签
    - dop_tensor: Tensor, 并行度信息
    """
    # 读取特征数据
    query_features = featurelize.process_query_features(feature_csv)
    
    # 读取查询执行信息 (标签)
    query_info_df = pd.read_csv(info_csv, delimiter=';')
    
    X_list, y_exec_list, y_mem_list, dop_list = [], [], [], []
    
    for (query_id, query_dop), feature_vector in query_features.items():
        # 在标签表中找到匹配的 query_id 和 query_dop
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
    
    # 转换为 Tensor
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_exec_tensor = torch.tensor(y_exec_list, dtype=torch.float32)
    y_mem_tensor = torch.tensor(y_mem_list, dtype=torch.float32)
    dop_tensor = torch.tensor(dop_list, dtype=torch.float32)

    return X_tensor, y_exec_tensor, y_mem_tensor, dop_tensor

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

def train_and_evaluate(
    query_features_train_csv, query_info_train_csv, 
    query_features_test_csv, query_info_test_csv, 
    epochs=100, lr=0.01, batch_size=32, suffix="NN_model"
):
    """
    读取训练集和测试集数据，训练执行时间和内存模型，并评估性能。

    Parameters:
    - query_features_train_csv: str, 训练集的查询特征 CSV 文件路径
    - query_info_train_csv: str, 训练集的查询标签 CSV 文件路径
    - query_features_test_csv: str, 测试集的查询特征 CSV 文件路径
    - query_info_test_csv: str, 测试集的查询标签 CSV 文件路径
    - epochs: int, 训练轮数
    - lr: float, 学习率
    - suffix: str, 模型保存的后缀名
    """
    
    # 读取训练数据
    X_train, y_exec_train, y_mem_train, dop_train = load_features_and_labels(query_features_train_csv, query_info_train_csv)

    # 读取测试数据
    X_test, y_exec_test, y_mem_test, dop_test = load_features_and_labels(query_features_test_csv, query_info_test_csv)

    # 初始化模型
    exec_model = PPM_model_NN.Exec_CurveFitModel(X_train.shape[1])
    mem_model = PPM_model_NN.Mem_CurveFitModel(X_train.shape[1])
    
     # 创建 DataLoader 进行批量训练
    train_dataset_exec = TensorDataset(X_train, y_exec_train, dop_train)
    train_loader_exec = DataLoader(train_dataset_exec, batch_size=batch_size, shuffle=True, drop_last=True)

    train_dataset_mem = TensorDataset(X_train, y_mem_train, dop_train)
    train_loader_mem = DataLoader(train_dataset_mem, batch_size=batch_size, shuffle=True, drop_last=True)

    exec_optimizer = optim.Adam(exec_model.parameters(), lr=lr)
    mem_optimizer = optim.Adam(mem_model.parameters(), lr=lr)

    exec_scheduler = optim.lr_scheduler.StepLR(exec_optimizer, step_size=10, gamma=0.8)
    mem_scheduler = optim.lr_scheduler.StepLR(mem_optimizer, step_size=10, gamma=0.8)

    start_time = time.time()

    # 训练执行时间模型
    for epoch in range(epochs):
        exec_model.train()
        epoch_loss = 0.0  
        
        for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader_exec):
            exec_optimizer.zero_grad()
            pred_params = exec_model(X_batch)
            loss = PPM_model_NN.curve_exec_loss(pred_params, dop_batch, y_batch)

            if torch.any(torch.isnan(loss)):
                print(f"Exec Model NaN detected at epoch {epoch}, batch {batch_idx}. Resetting model.")
                break  

            loss.backward()
            exec_optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader_exec)
        exec_scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Exec Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {exec_scheduler.get_last_lr()[0]:.6f}")

    # 训练内存模型
    for epoch in range(epochs):
        mem_model.train()
        epoch_loss = 0.0  
        
        for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader_mem):
            mem_optimizer.zero_grad()
            pred_params = mem_model(X_batch)
            loss = PPM_model_NN.curve_mem_loss(pred_params, dop_batch, y_batch)

            if torch.any(torch.isnan(loss)):
                print(f"Mem Model NaN detected at epoch {epoch}, batch {batch_idx}. Resetting model.")
                break  

            loss.backward()
            mem_optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader_mem)
        mem_scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Mem Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {mem_scheduler.get_last_lr()[0]:.6f}")
    # 评估执行时间模型 (测试集)
    exec_test_results = PPM_model_NN.predict_and_evaluate_exec_curve(
        exec_model, X_test, y_exec_test, dop_test, suffix=f"exec"
    )

    # 评估内存模型 (测试集)
    mem_test_results = PPM_model_NN.predict_and_evaluate_exec_curve(
        mem_model, X_test, y_mem_test, dop_test, suffix=f"mem"
    )

    results = {
        "execution_model_test_results": exec_test_results,
        "memory_model_test_results": mem_test_results,
    }

    # 保存到 CSV
    save_results_to_csv(results, suffix)

    return results

def evaluate(
    query_features_test_csv, query_info_test_csv, 
    suffix="NN_model"
):
    """
    读取训练集和测试集数据，训练执行时间和内存模型，并评估性能。

    Parameters:
    - query_features_train_csv: str, 训练集的查询特征 CSV 文件路径
    - query_info_train_csv: str, 训练集的查询标签 CSV 文件路径
    - query_features_test_csv: str, 测试集的查询特征 CSV 文件路径
    - query_info_test_csv: str, 测试集的查询标签 CSV 文件路径
    - epochs: int, 训练轮数
    - lr: float, 学习率
    - suffix: str, 模型保存的后缀名
    """
    

    # 读取测试数据
    X_test, y_exec_test, y_mem_test, dop_test = load_features_and_labels(query_features_test_csv, query_info_test_csv)


    onnx_model_path_exec = "/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_NN/exec_NN.onnx"
    onnx_model_path_mem = "/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_NN/mem_NN.onnx"
    # 评估执行时间模型 (测试集)
    exec_test_results = PPM_model_NN.predict_and_evaluate_exec_curve(
        onnx_model_path_exec, X_test, y_exec_test, dop_test, suffix=f"exec"
    )

    # 评估内存模型 (测试集)
    mem_test_results = PPM_model_NN.predict_and_evaluate_exec_curve(
        onnx_model_path_mem, X_test, y_mem_test, dop_test, suffix=f"mem"
    )

    results = {
        "execution_model_test_results": exec_test_results,
        "memory_model_test_results": mem_test_results,
    }

    # 保存到 CSV
    save_results_to_csv(results, suffix)

    return results


# 示例调用
if __name__ == "__main__":
    results = train_and_evaluate(
    "/home/zhy/opengauss/data_file/tpch_10g_output_500/plan_info.csv", "/home/zhy/opengauss/data_file/tpch_10g_output_500/query_info.csv",
    "/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv", "/home/zhy/opengauss/data_file/tpch_10g_output_22/query_info.csv",
    epochs=200, lr=0.005, suffix="NN_model"
    )
    # results = evaluate(
    #     "/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv",
    #     "/home/zhy/opengauss/data_file/tpcds_10g_output/query_info.csv"
    # )
