# Serverless Query Predictor & DOP Optimizer

面向 openGauss 无服务器并行数据库的查询代价预测与并行度（DOP）优化工具。该工具从查询执行计划中提取算子级特征，通过多种机器学习模型预测执行时间与内存消耗，并基于预测结果在 Pipeline / ThreadBlock 粒度上为各执行段选择最优并行度，以实现延迟与资源成本之间的平衡。

## 项目结构

```
├── config/                  # 配置管理
│   ├── main_config.py       # 数据集、方法、优化算法、默认参数等全局配置
│   ├── structure_config.py  # 算子类型列表、特征定义、训练 epoch 配置
│   └── mci_config.py        # MCI 方法配置
├── core/                    # 核心数据结构与运行时
│   ├── plan_node.py         # 计划树节点与 ONNX 推理封装
│   ├── onnx_manager.py      # ONNX 模型加载与管理
│   ├── thread_block.py      # ThreadBlock 抽象与 DOP 选择逻辑
│   ├── pipeline_pair.py     # Pipeline 对构建
│   └── pdg_builder.py       # Pipeline 依赖图（PDG）构建
├── training/                # 模型训练
│   ├── operator_dop_aware/  # DOP 感知算子模型（PyTorch → ONNX）
│   ├── operator_non_dop_aware/ # 非 DOP 感知算子模型（XGBoost/SVR → ONNX）
│   ├── query_level/         # 查询级 XGBoost 模型（Optuna 超参搜索）
│   ├── PPM/                 # PPM 模型（GNN / NN）
│   └── MCI/                 # MCI 模型（图编码 + MLP）
├── inference/               # 推理模块
│   └── predict_queries.py   # 算子级预测与查询级时间聚合
├── optimization/            # 优化模块
│   ├── optimizer.py         # Pipeline / 查询级 DOP 优化主流程
│   ├── tree_updater.py      # 计划树 DOP 更新
│   ├── threading_utils.py   # 线程工具
│   ├── result_processor.py  # 结果输出与多策略对比
│   └── moo_dop_optimizer.py # NSGA-II 多目标优化（pymoo）
├── pdg_moo/                 # PDG-MOO 优化算法
│   ├── algorithm.py         # 分段竞争式 Top-K + 精英池迭代
│   ├── context.py           # 优化上下文
│   ├── actions.py           # 动作空间定义
│   ├── pool.py              # 精英解池
│   ├── incremental_eval.py  # 增量评估
│   └── run.py               # PDG-MOO 入口
├── evaluation/              # 评估模块
│   └── evaluate_predictions.py  # Q-error 汇总与报告
├── scripts/                 # 脚本入口
│   ├── main.py              # 总控脚本（推荐）
│   ├── train.py             # 训练入口（CLI）
│   ├── inference.py         # 推理入口（CLI）
│   ├── evaluate.py          # 评估入口（CLI）
│   ├── compare.py           # 策略对比（CLI）
│   └── ...
├── utils/                   # 工具函数
│   ├── dataset_loader.py    # 数据集加载与切分
│   ├── path_utils.py        # 路径管理
│   ├── feature_engineering.py # 特征工程
│   └── logger.py            # 日志
├── data_kunpeng/            # 实验数据（TPC-H / TPC-DS / JOB）
├── input/                   # 对比基线数据
├── output/                  # 模型、预测结果、优化结果
└── docs/                    # 补充文档
```

## 功能概述

### 1. 模型训练

支持四类预测模型，均导出为 ONNX 格式以统一推理接口：

| 模型 | 方法 | 说明 |
|------|------|------|
| DOP 感知算子模型 | PyTorch MLP → ONNX | 按算子类型分别训练，输入包含 DOP 维度 |
| 非 DOP 感知算子模型 | XGBoost / SVR → ONNX | 按算子类型分别训练，不包含 DOP 信息 |
| 查询级模型 | XGBoost + Optuna → ONNX | 以查询整体特征预测总执行时间 |
| PPM | GNN / NN → ONNX | 拟合 DOP-延迟/DOP-内存关系曲线 |
| MCI | 图编码（GAT）+ MLP | 在 Pipeline 粒度预测不同 DOP 下的延迟 |
DOP 感知算子模型 和 非 DOP 感知算子模型是我们训练的算子模型， 查询级模型 PPM MCI 是其他工作的方法

### 2. 推理

从 `plan_info.csv` 中读取测试集算子信息，加载训练好的 ONNX 模型，逐算子预测执行时间和内存消耗，再按计划树结构聚合为查询级结果。

### 3. DOP 优化

在得到算子级预测后，为查询计划中的各 Pipeline / ThreadBlock 选择最优 DOP。支持以下优化策略：

- **Pipeline 优化**：基于启发式规则与流速区间匹配，自顶向下为各 ThreadBlock 选择 DOP
- **查询级优化**：以查询级模型直接预测不同 DOP 下的总时间
- **NSGA-II 多目标优化**：同时优化延迟、资源成本（DOP × 延迟）和流速匹配度
- **PDG-MOO**：基于 Pipeline 依赖图的分段竞争式多目标优化

### 4. 评估与对比

通过 Q-error 指标评估预测精度，并支持与多种基线策略（default DOP、Auto-DOP、PPM、FGO 等）进行性能对比。

## 环境依赖

```
python >= 3.8
numpy
pandas
torch
onnxruntime
xgboost
optuna
scikit-learn
skl2onnx
onnxmltools
pymoo          # 可选，用于 NSGA-II 多目标优化
```

## 快速开始

### 数据准备

将实验数据放置到 `data_kunpeng/` 目录下，目录结构需符合 `config/main_config.py` 中 `DATASETS` 的定义。每个数据集需包含 `plan_info.csv`（算子级信息）和 `query_info.csv`（查询级信息）。

### 方式一：总控脚本（推荐）

编辑 `scripts/main.py` 中的配置区域，设置数据集、训练方法、优化算法等参数，然后运行：

```bash
python scripts/main.py
```

可通过修改以下变量控制运行流程：

```python
DATASET = 'tpcds'                    # 数据集：'tpch', 'tpcds', 'job'
TRAIN_MODE = 'estimated_train'       # 训练模式：'exact_train', 'estimated_train'
TRAIN_METHOD = 'dop_aware'           # 训练方法：'dop_aware', 'non_dop_aware', 'ppm', 'query_level', 'mci'
OPTIMIZATION_ALGORITHM = 'pipeline'  # 优化算法：'pipeline', 'query_level', 'auto_dop', 'ppm'

RUN_TRAIN = True       # 是否执行训练
RUN_INFERENCE = True   # 是否执行推理
RUN_OPTIMIZE = True    # 是否执行优化
RUN_EVALUATE = True    # 是否执行评估
RUN_COMPARE = True     # 是否执行策略对比
```

### 方式二：独立脚本

```bash
# 训练
python scripts/train.py --method dop_aware --dataset tpcds --train_mode estimated_train

# 推理
python scripts/inference.py --dataset tpcds --train_mode estimated_train --use_estimates

# 评估
python scripts/evaluate.py --dataset tpcds --train_mode estimated_train

# 策略对比
python scripts/compare.py --dataset tpcds
```

## 输出目录结构

```
output/<dataset>/
├── models/
│   └── <train_mode>/
│       ├── operator_dop_aware/        # DOP 感知模型（.onnx）
│       ├── operator_non_dop_aware/    # 非 DOP 感知模型（.onnx）
│       ├── query_level/               # 查询级模型（.onnx）
│       └── PPM/                       # PPM 模型
├── predictions/
│   ├── operator_level/                # 算子级预测结果
│   ├── query_level/                   # 查询级预测结果
│   └── PPM/
├── evaluations/
│   └── qerror_summary_report.csv      # Q-error 评估报告
└── optimization_results/
    ├── query_details_optimized.json    # 优化后的查询详情
    ├── operators_optimized.csv         # 算子级优化结果
    ├── query_max_dop_optimized.csv     # 查询最大 DOP
    └── timing_log.csv                 # 优化耗时日志
```

## 支持的数据集

| 数据集 | 训练集目录 | 测试集目录 |
|--------|-----------|-----------|
| TPC-H | `tpch_output_500` | `tpch_output_22` |
| TPC-DS | `tpcds_100g_output_train` | `tpcds_100g_output_test` |
| JOB | `job_100g_data_train` | `job_100g_data_test` |

## 支持的算子类型

系统针对 openGauss 向量化执行引擎中的主要算子进行建模，包括：CStore Scan、Vector Hash Join、Vector Sonic Hash Join、Vector Sort、Vector Hash Aggregate、Vector Streaming（LOCAL GATHER / REDISTRIBUTE / BROADCAST）等 20 余种算子。完整列表参见 `config/structure_config.py`。

## 补充文档

- [MOO Pipeline 优化说明](docs/MOO_PIPELINE_OPTIMIZATION.md)
- [推理方法对比](docs/inference_methods_comparison.md)
- [脚本使用指南](scripts/README.md)
