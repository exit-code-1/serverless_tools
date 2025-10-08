# Serverless Predictor 重构后使用说明

## 🎯 重构目标
- **整合相似功能**：将多个训练/推理/优化脚本整合到统一入口
- **减少重复代码**：提取公共配置和工具函数
- **代码结构更清晰**：通过参数控制使用不同方法

## 📁 新的脚本结构

```
scripts/
├── config.py          # 统一配置管理
├── utils.py           # 公共工具函数
├── main.py            # 主控制脚本 (推荐使用)
├── train.py           # 统一训练入口
├── inference.py       # 统一推理入口
├── optimize.py        # 统一优化入口
├── evaluate.py        # 统一评估入口
└── compare.py         # 对比分析
```

## 🚀 使用方法

### 方法1：使用主控制脚本 (推荐)

直接运行 `main.py`，通过修改文件中的变量来控制参数：

```bash
python main.py
```

在 `main.py` 文件中修改配置区域：

```python
# ==================== 配置区域 ====================
# 基础配置
DATASET = 'tpcds'  # 数据集: 'tpch' 或 'tpcds'
TRAIN_MODE = 'estimated_train'  # 训练模式: 'exact_train' 或 'estimated_train'
USE_ESTIMATES_MODE = True  # 是否使用估计值

# 训练配置
TRAIN_METHOD = 'dop_aware'  # 训练方法: 'dop_aware', 'non_dop_aware', 'ppm', 'query_level'
PPM_TYPE = 'GNN'  # PPM类型: 'GNN' 或 'NN'

# 优化配置
OPTIMIZATION_ALGORITHM = 'pipeline'  # 优化算法: 'pipeline', 'query_level', 'auto_dop', 'ppm'

# 运行控制 - 设置要运行的功能 (True/False)
RUN_TRAIN = True  # 是否运行训练
RUN_INFERENCE = True  # 是否运行推理
RUN_OPTIMIZE = True  # 是否运行优化
RUN_EVALUATE = True  # 是否运行评估
RUN_COMPARE = True  # 是否运行对比分析
# =======================================================
```

### 方法2：直接使用各个脚本

```bash
# 训练
python train.py --method dop_aware --dataset tpcds --train_mode estimated_train

# 推理
python inference.py --dataset tpcds --train_mode estimated_train --use_estimates

# 优化
python optimize.py --algorithm pipeline --dataset tpcds --train_mode estimated_train

# 评估
python evaluate.py --dataset tpcds --train_mode estimated_train

# 对比
python compare.py --dataset tpcds
```

## ⚙️ 配置说明

### 数据集配置
- `tpch`: TPC-H 数据集
- `tpcds`: TPC-DS 数据集

### 训练方法
- `dop_aware`: DOP感知算子模型
- `non_dop_aware`: 非DOP感知算子模型
- `ppm`: PPM方法 (需要指定 --ppm_type GNN/NN)
- `query_level`: 查询级别模型

### 训练模式
- `exact_train`: 精确训练
- `estimated_train`: 估计训练

### 优化算法
- `pipeline`: Pipeline优化 (你们的核心方法)
- `query_level`: 查询级别优化
- `auto_dop`: Auto-DOP方法
- `ppm`: PPM方法

## 🔧 参数说明

### 训练参数
- `--method`: 训练方法 (必需)
- `--dataset`: 数据集名称 (默认: tpcds)
- `--train_mode`: 训练模式 (默认: estimated_train)
- `--ppm_type`: PPM方法类型 (默认: GNN)
- `--total_queries`: 总查询数量 (默认: 500)
- `--train_ratio`: 训练比例 (默认: 1.0)
- `--n_trials`: XGBoost优化试验次数 (默认: 30)

### 推理参数
- `--dataset`: 数据集名称 (默认: tpcds)
- `--train_mode`: 训练模式 (默认: estimated_train)
- `--use_estimates`: 是否使用估计值 (默认: True)

### 优化参数
- `--algorithm`: 优化算法 (必需)
- `--dataset`: 数据集名称 (默认: tpcds)
- `--train_mode`: 训练模式 (默认: estimated_train)
- `--base_dop`: 基准DOP (默认: 64)
- `--min_improvement_ratio`: 最小改进比例 (默认: 0.2)
- `--min_reduction_threshold`: 最小减少阈值 (默认: 200)
- `--use_estimates`: 是否使用估计值 (默认: True)

## 📊 输出结构

```
output/
├── tpcds/
│   ├── models/
│   │   ├── estimated_train/
│   │   │   ├── operator_dop_aware/
│   │   │   ├── operator_non_dop_aware/
│   │   │   ├── query_level/
│   │   │   └── PPM/
│   ├── predictions/
│   │   ├── operator_level/
│   │   ├── query_level/
│   │   └── PPM/
│   ├── evaluations/
│   │   └── qerror_summary_report.csv
│   └── optimization_results/
│       ├── query_details_optimized.json
│       ├── operators_optimized.csv
│       ├── query_max_dop_optimized.csv
│       └── timing_log.csv
└── evaluations/
    └── optimization_comparison_report.csv
```

## 🔄 工作流程示例

### 完整实验流程
```bash
# 1. 训练所有模型
python main.py  # 修改 TRAIN_METHOD 为 'dop_aware'，设置 RUN_TRAIN = True
python main.py  # 修改 TRAIN_METHOD 为 'non_dop_aware'，设置 RUN_TRAIN = True
python main.py  # 修改 TRAIN_METHOD 为 'ppm'，设置 RUN_TRAIN = True
python main.py  # 修改 TRAIN_METHOD 为 'query_level'，设置 RUN_TRAIN = True

# 2. 运行推理
python main.py  # 设置 RUN_INFERENCE = True

# 3. 运行优化
python main.py  # 设置 RUN_OPTIMIZE = True

# 4. 运行评估
python main.py  # 设置 RUN_EVALUATE = True

# 5. 运行对比分析
python main.py  # 设置 RUN_COMPARE = True
```

## 🆚 与原始脚本的对比

### 原始方式 (11个独立脚本)
```bash
python run_dop_aware_training.py
python run_non_dop_aware_training.py
python run_PPM_train.py
python run_query_level_training.py
python run_inference.py
python run_optimization.py
python run_query_level_optimizer.py
python run_evaluation.py
python run_comparison.py
python run_create_datasplit.py
python run_consolidated_timing_analysis.py
```

### 重构后方式 (1个主控制脚本)
```bash
python main.py  # 修改文件中的变量来控制运行参数
```

## ✅ 重构优势

1. **代码复用**：消除了重复的配置和路径设置代码
2. **统一接口**：所有功能通过一个主脚本控制
3. **参数化**：通过修改文件中的变量控制使用不同方法，无需命令行参数
4. **易于维护**：配置集中管理，修改更容易
5. **错误处理**：统一的错误处理和日志记录
6. **文档完善**：每个脚本都有详细的帮助信息

## 🐛 故障排除

### 常见问题
1. **导入错误**：确保项目根目录在 Python 路径中
2. **文件不存在**：检查数据文件路径是否正确
3. **模型未找到**：确保已训练相应的模型
4. **权限问题**：确保有写入输出目录的权限

### 调试模式
```bash
# 直接运行主脚本
python main.py

# 或者直接运行各个功能脚本
python train.py --help
python optimize.py --help
```
