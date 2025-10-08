# MCI Dual Model Training and Inference

MCI模型现在支持分别训练执行时间和内存使用两个独立的模型，这样可以更好地对比不同算法的预测效果。

## 主要变更

### 1. 数据加载器 (`mci_data_loader.py`)
- 新增 `load_mci_pipeline_data_exec()` - 加载执行时间训练数据
- 新增 `load_mci_pipeline_data_mem()` - 加载内存使用训练数据
- 新增 `create_mci_data_loaders_exec()` - 创建执行时间数据加载器
- 新增 `create_mci_data_loaders_mem()` - 创建内存数据加载器
- 内部函数 `_load_mci_pipeline_data()` 支持不同的目标类型

### 2. 训练脚本 (`mci_train.py`)
- 新增 `train_mci_execution_model()` - 训练执行时间模型
- 新增 `train_mci_memory_model()` - 训练内存模型
- 主函数现在会训练两个模型并保存为：
  - `{model_name}_exec.onnx` - 执行时间模型
  - `{model_name}_mem.onnx` - 内存模型
  - `{model_name}_combined_results.json` - 合并的训练结果

### 3. 推理脚本 (`mci_inference.py`)
- 新增 `load_test_data_by_query_exec()` - 加载执行时间测试数据
- 新增 `load_test_data_by_query_mem()` - 加载内存测试数据
- 新增 `run_query_level_inference_dual()` - 双模型推理
- 主函数现在支持双模型推理并输出对比结果

## 使用方法

### 训练双模型

```bash
python mci_train.py --config mci_config_small.json
```

训练完成后会生成：
- `mci_model_exec.onnx` - 执行时间模型
- `mci_model_mem.onnx` - 内存模型
- `mci_model_exec.pth` - 执行时间模型权重
- `mci_model_mem.pth` - 内存模型权重
- `mci_model_combined_results.json` - 合并结果

### 推理双模型

```bash
# 使用默认模型路径
python mci_inference.py --config mci_config_small.json

# 或指定特定模型路径
python mci_inference.py --config mci_config_small.json \
    --exec_model ./mci_output/mci_model_exec.onnx \
    --mem_model ./mci_output/mci_model_mem.onnx
```

### 输出结果

推理完成后会生成标准格式的文件：

**单独模型结果：**
- `mci_execution_time_pipeline_results.csv` - 执行时间pipeline级别结果
- `mci_execution_time_query_results.csv` - 执行时间query级别结果
- `mci_execution_time_summary.csv` - 执行时间汇总统计
- `mci_memory_usage_pipeline_results.csv` - 内存pipeline级别结果
- `mci_memory_usage_query_results.csv` - 内存query级别结果
- `mci_memory_usage_summary.csv` - 内存汇总统计

**合并结果（标准格式）：**
- `mci_combined_inference_results.csv` - 合并的推理结果（匹配其他方法的格式）
- `mci_combined_summary.json` - 合并的汇总统计

**标准CSV格式列：**
```csv
Query ID;Query DOP;Query ID (Mapped);Actual Execution Time (s);Predicted Execution Time (s);Actual Memory Usage (MB);Predicted Memory Usage (MB);Execution Time Q-error;Memory Q-error;Time Calculation Duration (s);Memory Calculation Duration (s)
```

## 配置说明

配置文件 `mci_config_small.json` 中的关键设置：

```json
{
  "evaluation": {
    "epsilon_time": 1e-06,    // 执行时间评估阈值
    "epsilon_mem": 0.01,      // 内存评估阈值
    "q_error_bins": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
  }
}
```

## 数据要求

确保你的数据文件包含以下列：
- `execution_time` - 执行时间（毫秒）
- `query_used_mem` - 内存使用量（KB）

## 注意事项

1. **内存数据**: 确保query_info CSV文件包含 `query_used_mem` 列
2. **模型命名**: 双模型会自动添加 `_exec` 和 `_mem` 后缀
3. **向后兼容**: 原有的单模型训练函数仍然可用
4. **性能对比**: 双模型推理可以更好地对比不同算法的执行时间和内存预测效果

## 示例输出

### 训练输出
```
============================================================
Training MCI Execution Time Model
============================================================
Loading execution time training data...
Creating MCI execution time model...
Starting execution time model training...
Exporting execution time model to ONNX: ./mci_output/mci_model_exec.onnx

============================================================
Training MCI Memory Model  
============================================================
Loading memory training data...
Creating MCI memory model...
Starting memory model training...
Exporting memory model to ONNX: ./mci_output/mci_model_mem.onnx

============================================================
Training completed successfully!
Execution time model saved to: ./mci_output/mci_model_exec.onnx
Memory model saved to: ./mci_output/mci_model_mem.onnx
Combined results saved to: ./mci_output/mci_model_combined_results.json
============================================================
```

### 推理输出
```
============================================================
DUAL MODEL INFERENCE SUMMARY
============================================================
Execution Time Model: ./mci_output/mci_model_exec.onnx
Memory Model: ./mci_output/mci_model_mem.onnx
Total inference time: 45.23s

EXECUTION TIME RESULTS:
  Queries: 100
  Pipelines: 450
  Query mean Q-error: 1.2345
  Query median Q-error: 1.1234
  Pipeline mean Q-error: 1.0987

MEMORY RESULTS:
  Queries: 100
  Pipelines: 450
  Query mean Q-error: 1.4567
  Query median Q-error: 1.3456
  Pipeline mean Q-error: 1.2345

Results saved to: ./mci_output
============================================================
```

### 标准CSV输出示例
```csv
Query ID;Query DOP;Query ID (Mapped);Actual Execution Time (s);Predicted Execution Time (s);Actual Memory Usage (MB);Predicted Memory Usage (MB);Execution Time Q-error;Memory Q-error;Time Calculation Duration (s);Memory Calculation Duration (s)
1;4;1;2.345;2.123;156.7;142.3;0.095;0.092;0.001;0.001
2;8;2;1.876;1.945;203.4;198.7;0.037;0.023;0.001;0.001
```
