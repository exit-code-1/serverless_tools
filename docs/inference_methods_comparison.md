# 推理方法比较功能

## 概述

本功能用于比较两种计算query延迟的方法的精度差异：

1. **映射算法** (`calculate_query_execution_time`): 使用复杂的线程模型和物化算子处理，基于算子预测值
2. **真实算子时间相加**: 使用真实的算子执行时间（`execution_time`列）直接相加计算query延迟

## 使用方法

### 方法1: 通过主控制脚本

在 `scripts/main.py` 中设置以下参数：

```python
# 运行控制 - 设置要运行的功能 (True/False)
RUN_COMPARE_INFERENCE_METHODS = True  # 是否运行推理方法比较

# 基础配置
DATASET = 'tpch'  # 或 'tpcds'
TRAIN_MODE = 'exact_train'  # 或 'estimated_train'
USE_ESTIMATES_MODE = False  # 是否使用估计值
```

然后运行：
```bash
python scripts/main.py
```

### 方法2: 直接运行比较脚本

```bash
python scripts/compare_inference_methods.py
```

或者修改脚本中的参数：
```python
# 在脚本末尾修改这些参数
DATASET = 'tpch'  # 'tpch' 或 'tpcds'
TRAIN_MODE = 'exact_train'  # 'exact_train' 或 'estimated_train'
USE_ESTIMATES = False  # True 或 False
```

### 方法3: 使用测试脚本

```bash
python test_compare_inference.py
```

## 输出结果

### 控制台输出

运行时会显示以下统计信息：

- **执行时间预测比较**:
  - 映射算法平均Q-error (映射算法 vs 实际query时间)
  - 真实算子相加平均Q-error (真实算子时间相加 vs 实际query时间)
  - Q-error改进值和改进比例 (映射算法相比真实算子相加的改进)

- **内存预测比较**:
  - 映射算法平均Q-error (映射算法 vs 实际query内存)
  - 真实算子相加平均Q-error (真实算子内存相加 vs 实际query内存)
  - Q-error改进值和改进比例 (映射算法相比真实算子相加的改进)

- **预测值差异统计**:
  - 执行时间平均差异和差异比例
  - 内存平均差异和差异比例

- **推理时间比较**:
  - 两种方法的平均推理时间

### 文件输出

结果会保存到以下位置：

1. **详细结果CSV文件**: `output/comparison/{dataset}/{train_mode}/inference_methods_comparison.csv`
   - 包含每个query的详细比较数据
   - 字段包括：query_id, dop, 实际值, 两种方法的预测值, Q-error, 推理时间等

2. **汇总报告**: `output/comparison/{dataset}/{train_mode}/comparison_summary.txt`
   - 包含统计汇总信息

## 结果字段说明

### CSV文件字段

| 字段名 | 说明 |
|--------|------|
| query_id | 查询ID |
| dop | 并行度 |
| actual_time | 实际执行时间(ms) |
| predicted_time_mapping | 映射算法预测时间(ms) |
| real_time_sum | 真实算子时间相加(ms) |
| actual_memory | 实际内存使用(KB) |
| predicted_memory_mapping | 映射算法预测内存(KB) |
| real_memory_sum | 真实算子内存相加(KB) |
| q_error_mapping | 映射算法Q-error |
| q_error_sum | 真实算子相加Q-error |
| q_error_mem_mapping | 映射算法内存Q-error |
| q_error_mem_sum | 真实算子相加内存Q-error |
| mapping_inference_time | 映射算法推理时间(s) |
| sum_inference_time | 真实算子相加推理时间(s) |
| time_difference | 预测时间差异(ms) |
| time_difference_ratio | 预测时间差异比例 |
| memory_difference | 预测内存差异(KB) |
| memory_difference_ratio | 预测内存差异比例 |
| q_error_improvement | Q-error改进值 |
| q_error_mem_improvement | 内存Q-error改进值 |

## 前置条件

运行此功能前需要确保：

1. 已训练好对应的模型文件
   - 非DOP模型目录: `models/{dataset}/{train_mode}/non_dop_aware`
   - DOP模型目录: `models/{dataset}/{train_mode}/dop_aware`

2. 测试数据文件存在
   - 计划信息文件: `input/{dataset}/test/plan_info.csv`
   - 查询信息文件: `input/{dataset}/test/query_info.csv`

## 注意事项

1. 该功能会处理所有测试数据中的查询，可能需要较长时间
2. 确保有足够的内存来处理大型数据集
3. 如果遇到NaN预测值，会在控制台输出警告信息
4. 结果文件使用分号(;)作为分隔符

## 示例输出

```
比较推理方法精度差异
================================================================================
数据集: tpch
训练模式: exact_train
使用估计值: False
================================================================================

📊 比较结果统计:
================================================================================
执行时间预测比较:
  映射算法平均Q-error: 0.123456 (映射算法 vs 实际query时间)
  真实算子相加平均Q-error: 0.234567 (真实算子时间相加 vs 实际query时间)
  Q-error改进: 0.111111 (映射算法相比真实算子相加的改进)
  改进比例: 47.37%

内存预测比较:
  映射算法平均Q-error: 0.056789 (映射算法 vs 实际query内存)
  真实算子相加平均Q-error: 0.067890 (真实算子内存相加 vs 实际query内存)
  Q-error改进: 0.011101 (映射算法相比真实算子相加的改进)
  改进比例: 16.35%

预测值差异统计:
  执行时间平均差异: 123.45ms
  执行时间差异比例: 5.67%
  内存平均差异: 234.56KB
  内存差异比例: 3.45%

推理时间比较:
  映射算法平均推理时间: 0.001234s
  直接相加平均推理时间: 0.000567s
  推理时间差异: 0.000667s

✅ 推理方法比较完成!
```
