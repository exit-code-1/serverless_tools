# MCI-based Fine-Grained Modeling Framework

基于MCI（Multi-Channel Integration）的细粒度建模框架，用于预测pipeline在不同DOP（Degree of Parallelism）下的延迟。

## 架构概述

```
Query Plan (Ch1) -> Plan Embedder (GTN) -> Plan Embedding -> Latency Predictor (MLP) -> Latency
```

### 主要组件

1. **Plan Embedder (GTN)**: 使用图转换器网络将查询计划嵌入为固定维度的向量
2. **Latency Predictor (MLP)**: 使用多层感知机预测不同DOP级别下的延迟

## 文件结构

```
training/MCI/
├── __init__.py              # 模块初始化
├── mci_model.py             # MCI模型架构
├── mci_data_loader.py       # 数据加载器
├── mci_train.py            # 训练脚本
├── example_usage.py        # 使用示例
└── README.md               # 说明文档
```

## 主要特性

### 1. Pipeline粒度建模
- 基于streaming算子的pipeline自动划分
- 每个pipeline独立预测其在不同DOP下的延迟
- 支持pipeline级别的特征提取和建模
- 自动处理非并行算子：包含非并行算子的pipeline只训练DOP=1

### 2. DOP特征集成
- 将pipeline DOP信息直接集成到算子特征中
- 支持多种DOP特征：直接DOP级别、对数缩放、平方根缩放等
- 自动计算DOP效率特征（如每DOP的行数、宽度等）
- 包含线程ID、算子并行性等额外特征

### 3. 图神经网络架构
- 使用GAT（Graph Attention Network）进行更好的注意力机制
- 支持残差连接和层归一化
- 全局池化（均值+最大值）生成pipeline级别表示

### 4. 多DOP级别预测
- 支持为不同DOP级别训练专门的预测头
- 灵活的预测模式：单DOP级别或全DOP级别预测

### 5. MOO优化
- 使用NSGA-II算法优化pipeline的DOP配置
- 多目标优化：最小化延迟和成本（延迟×DOP）
- 支持帕累托前沿分析和最佳解选择

## 使用方法

### 1. 配置文件方式（推荐）

```bash
# 创建配置文件
python training/MCI/mci_train.py --create_config default

# 编辑配置文件，设置数据路径
vim mci_config_default.json

# 使用配置文件训练
python training/MCI/mci_train.py --config mci_config_default.json
```

### 2. 命令行参数方式

```bash
python training/MCI/mci_train.py \
    --train_csv path/to/train_features.csv \
    --train_info_csv path/to/train_info.csv \
    --dop_levels 1 2 4 8 \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --output_dir ./mci_output
```

### 3. 编程接口使用

```python
from training.MCI import create_mci_model, mci_loss

# 创建模型
model = create_mci_model(
    input_dim=50,           # 算子特征维度
    hidden_dim=128,         # 隐藏层维度
    embedding_dim=256,      # 嵌入维度
    num_dop_levels=8,       # DOP级别数量
    dropout=0.2             # Dropout率
)

# 前向传播
predictions, embeddings = model(x, edge_index, batch, dop_levels)
```

### 4. 数据格式

#### 算子特征CSV格式
```
query_id;plan_id;operator_type;l_input_rows;r_input_rows;actual_rows;width;predicate_cost;jointype;dop;child_plan
1;1;Vector Hash Join;1000;500;800;64;0.1;1;4;2,3
1;2;CStore Index Scan;0;0;1000;32;0.2;0;4;
```

#### 查询信息CSV格式
```
query_id;dop;execution_time;query_used_mem
1;1;150.5;1024
1;4;45.2;2048
```

## 配置预设

### 可用的预设配置

1. **default**: 标准配置，CPU优化，平衡性能和训练时间
2. **small**: 小模型，CPU优化，快速训练
3. **large**: 大模型，GPU优化，高精度
4. **quick**: 快速测试配置，CPU优化，最小模型
5. **production**: 生产环境配置，CPU优化，更多训练轮数

### 创建配置文件

```bash
# 创建不同预设的配置文件
python training/MCI/mci_train.py --create_config default
python training/MCI/mci_train.py --create_config small
python training/MCI/mci_train.py --create_config large
python training/MCI/mci_train.py --create_config quick
python training/MCI/mci_train.py --create_config production
```

## 模型参数

### Plan Embedder参数
- `input_dim`: 输入特征维度
- `hidden_dim`: 隐藏层维度（默认128）
- `embedding_dim`: 嵌入维度（默认256）
- `num_heads`: 注意力头数（默认8）
- `num_layers`: GNN层数（默认3）

### Latency Predictor参数
- `embedding_dim`: 嵌入维度
- `hidden_dims`: MLP隐藏层维度列表（默认[512, 256, 128]）
- `num_dop_levels`: DOP级别数量（默认8）
- `dropout`: Dropout率（默认0.2）

## 损失函数

MCI模型使用组合损失函数：
- 基础损失：MSE/MAE/Huber损失
- 相对误差损失：提高训练稳定性

```python
loss = basic_loss + alpha * relative_loss
```

## 评估指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **Q-error**: 查询误差（max(pred/actual, actual/pred) - 1）
- **按DOP级别分组的Q-error**: 分析不同并行度下的预测准确性

## 示例代码

运行完整的使用示例：

```bash
python training/MCI/example_usage.py
```

## 与PPM模型的对比

| 特性 | PPM模型 | MCI模型 |
|------|---------|---------|
| 建模粒度 | 整个查询计划 | Pipeline级别 |
| Pipeline划分 | 无明确划分 | 基于streaming算子自动划分 |
| 架构 | GCN + 注意力 + MLP | GAT + 多层MLP |
| DOP处理 | 参数化曲线拟合 | 直接集成到特征中 |
| 预测方式 | PCC公式 | 直接预测 |
| 灵活性 | 固定公式 | 学习DOP-延迟关系 |
| 特征工程 | 基础算子特征 | 增强算子特征 + pipeline特征 |

## CPU优化说明

默认配置已针对CPU训练进行优化：

### CPU优化特性
- **较小的模型尺寸**: 减少内存使用和计算复杂度
- **更大的批次大小**: 提高CPU训练效率
- **禁用GPU相关功能**: pin_memory=False, num_workers=0
- **默认设备**: 设置为CPU而非auto

### 配置对比

| 预设 | Hidden Dim | Embedding Dim | Epochs | Batch Size | Device | 用途 |
|------|------------|---------------|--------|------------|--------|------|
| **default** | 64 | 128 | 80 | 32 | CPU | 标准使用 |
| **small** | 32 | 64 | 40 | 64 | CPU | 快速训练 |
| **large** | 256 | 512 | 200 | 8 | GPU | 高精度 |
| **quick** | 16 | 32 | 5 | 128 | CPU | 功能测试 |
| **production** | 64 | 128 | 120 | 32 | CPU | 生产环境 |

## 注意事项

1. **特征维度**: 确保输入特征维度与实际算子特征匹配
2. **DOP级别**: 根据实际使用的DOP级别设置`num_dop_levels`
3. **非并行算子**: 包含非并行算子的pipeline会自动跳过多DOP训练
4. **CPU训练**: 默认配置已优化用于CPU训练，如需GPU训练请使用large配置
5. **内存使用**: 大图可能需要调整batch_size
6. **训练时间**: CPU训练通常比GPU慢，建议使用small或quick配置进行快速验证

## 未来扩展

1. **MOO算法集成**: 添加多目标优化算法
2. **动态DOP**: 支持连续DOP值预测
3. **多任务学习**: 同时预测延迟和内存使用
4. **注意力可视化**: 分析模型关注的算子特征

## 依赖要求

- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0
- NumPy
- Pandas
- TensorBoard (用于训练可视化)
