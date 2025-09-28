# 实验模块说明文档 (Experimentation Guide)

本文档旨在说明本项目中两个核心实验模块的设计与使用方法：**推理延迟记录**和**基数估计模拟**。

## 1. 推理延迟记录 (Inference Latency Recording)

为了评估我们各种预测模型在实际应用中的响应速度，项目中集成了对模型推理延迟的记录功能。

### 1.1 设计思路

我们关注两种类型的延迟：

-   **端到端预测延迟 (End-to-End Prediction Latency)**: 从准备输入特征到获得最终预测结果的完整耗时。
-   **纯模型推理延迟 (Pure Model Inference Latency)**: 仅指调用ONNX Runtime执行模型`run`或`predict`操作的耗时。

延迟记录被集成在各个模型的推理/评估流程中，并最终输出到相应的预测结果CSV文件中，方便后续分析。

### 1.2 实现细节与结果位置

#### a) 算子级别模型 (Operator-Level Models)

-   **实现位置**: `core/plan_node.py`
-   **逻辑**: 在 `PlanNode` 对象初始化时，其 `infer_exec_with_onnx()` 和 `infer_mem_with_onnx()` 方法内部使用 `time.time()` 记录了每次预测的耗时。这些时间被累加到 `self.pred_exec_time` 和 `self.pred_mem_time` 属性中。
-   **结果汇总**: `scripts/run_inference.py` 脚本在运行时，会累加查询中所有 `PlanNode` 对象的预测延迟。
-   **结果文件**: 最终的累积延迟时间记录在 `output/<dataset_name>/predictions/operator_level_inference_results.csv` 文件中的 `Time Calculation Duration (s)` 和 `Memory Calculation Duration (s)` 列。

#### b) 查询级别模型 (Query-Level Models - XGBoost, PPM-NN, PPM-GNN)

-   **实现位置**:
    -   XGBoost: `training/query_level/train.py` 内的 `test_onnx_xgboost` 函数。
    -   PPM-NN: `training/PPM/PPM_model_NN.py` 内的 `predict_and_evaluate_curve` 函数。
    -   PPM-GNN: `training/PPM/PPM_model_GNN.py` 内的 `predict_and_evaluate_curve` 函数。
-   **逻辑**: 在各自的评估函数中，对每次（或每批次）模型推理进行计时。
-   **结果文件**:
    -   **XGBoost模型**: 预测结果文件 `output/<dataset_name>/predictions/query_level_test_predictions.csv` 中增加了 `Time Prediction Latency (s)` 和 `Memory Prediction Latency (s)` 两列，记录了**每一次预测**的精确延迟。
    -   **PPM模型**: 统计文件 `output/<dataset_name>/predictions/PPM/<method>/qerror.csv` 的末尾，新增了**总推理时长**和**平均每查询推理延迟**的统计信息。

### 1.3 如何使用

您无需进行任何特殊配置。只需按照正常的实验流程运行各个 `run_*.py` 脚本，生成的预测结果文件中将自动包含延迟信息。

---

## 2. 基数估计模拟 (Cardinality Estimation Simulation)

该模块用于模拟真实世界中数据库优化器基数估计不准的情况，以检验我们预测模型和优化算法的鲁棒性。

### 2.1 设计思路

核心思想是在推理和优化阶段，将模型输入中的**真实行数 (`actual_rows`)** 替换为数据库提供的**估计行数 (`estimate_rows`)**。

这一模拟遵循了执行计划的依赖关系：一个算子的输入行数是其子节点的输出行数。因此，我们实现了一个**自底向上**的更新机制，确保基数估计的误差能够在整个查询计划树中正确传递和累积。

### 2.2 实现细节

1.  **模式开关**:
    -   我们在 `scripts/run_inference.py` 和 `scripts/run_optimization.py` 脚本的顶部引入了一个全局开关：
        ```python
        # 设置为 True 来运行基数估计模拟实验
        USE_ESTIMATES_MODE = True
        ```
    -   当此开关为 `True` 时，系统将进入模拟模式。

2.  **PlanNode 修改**:
    -   `core/plan_node.py` 中的 `PlanNode` 类被修改，以在初始化时同时存储真实值（如 `self.actual_rows`）和估计值（`self.estimate_rows`）。
    -   新增 `update_estimated_inputs()` 方法，用于根据子节点的估计输出 (`estimate_rows`) 来更新当前节点的估计输入 (`l_input_rows_estimate`, `r_input_rows_estimate`)。

3.  **树更新逻辑**:
    -   `optimization/tree_updater.py` 中的 `build_query_trees` 函数在建树后，如果检测到模拟模式开启，会调用一个新的 `update_tree_estimated_inputs` 函数。
    -   该函数会**自底向上**遍历整棵查询计划树，递归调用每个节点的 `update_estimated_inputs()` 方法。
    -   更新完毕后，会触发一次特征向量的重新生成和模型推理，以确保模型使用的是被“污染”过的特征。

### 2.3 如何进行实验

此实验分为两个步骤，旨在对比模型在有无基数估计误差时的表现。

**步骤一：运行基准实验 (使用真实值)**

1.  打开 `scripts/run_inference.py` 和 `scripts/run_optimization.py`。
2.  确保顶部的开关为 `USE_ESTIMATES_MODE = False`。
3.  正常运行脚本（例如 `python scripts/run_optimization.py`）。
4.  运行完成后，将生成的输出结果文件（例如 `output/tpch/optimization_results/query_details_optimized.json`）**重命名或备份**，例如命名为 `query_details_optimized_actual.json`。

**步骤二：运行模拟实验 (使用估计值)**

1.  再次打开 `scripts/run_inference.py` 和 `scripts/run_optimization.py`。
2.  将顶部的开关修改为 `USE_ESTIMATES_MODE = True`。
3.  重新运行相同的脚本。
4.  此时生成的结果（例如 `query_details_optimized.json`）是基于模拟基数估计误差得到的。

**步骤三：对比分析**

现在您可以对比分析 "actual" 版本和新生成的 "estimate" 版本的结果文件。通过比较预测的Q-error、最终的DOP决策、预测的执行时间等指标，可以量化基数估计不准对我们系统性能的影响。

> **注意**: 模型训练过程**不受** `USE_ESTIMATES_MODE` 开关的影响。模型应始终使用包含真实值的训练数据进行训练，以学习现实世界中的性能规律。模拟仅在推理和优化阶段进行。