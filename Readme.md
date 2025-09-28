项目目标: 面向 OpenGauss 的 Serverless 细粒度资源预测与查询计划优化。

核心流程:

    训练 (Training): 分别训练三种模型：

        查询级别模型 (Query Level): 预测整个查询的资源消耗（可能基于查询特征）。

        非 DOP 感知算子模型 (Operator Non-DOP Aware): 预测单个算子的资源消耗，不直接考虑 DOP 影响（或 DOP 作为普通特征）。

        DOP 感知算子模型 (Operator DOP Aware): 预测单个算子的资源消耗，并将 DOP 作为关键参数，模型能反映资源随 DOP 变化的曲线。

    推理 (Inference): 使用训练好的算子模型（主要是非 DOP 感知模型的结果聚合，但也加载了 DOP 感知模型）来预测给定查询计划的整体资源消耗，并评估预测准确性。

    优化 (Optimization): 利用训练好的模型（特别是 DOP 感知模型）和补充的真实运行数据：

        构建查询计划树并更新节点信息（预测时间、真实/插值时间）。

        将查询计划划分为线程块。

        为每个线程块选择最优的 DOP。

        输出优化后的计划详情。

    评估 (Evaluation): 对推理或优化的结果进行评估，例如计算 Q-error 的分桶统计。

-----------------------------------------------------------------------------

关键点说明:

    职责分离: 每个顶层目录（config, core, data, evaluation, inference, optimization, output, scripts, training, utils）都有明确的职责。

    核心数据结构 (core): PlanNode, ThreadBlock, ONNXModelManager 是贯穿多个模块的基础。

    配置中心 (config): structure.py 定义了大量常量和配置，是多个模块（特别是 training, core, utils）的依赖。

    训练模块 (training): 按模型类型细分，每个子模块包含模型定义 (model.py) 和训练流程 (train.py)。

    优化模块 (optimization): 内部进一步细分职责，optimizer 协调流程，tree_updater 更新数据，threading_utils 处理线程，result_processor 处理输出。

    工具函数 (utils): 存放可被多个模块复用的功能，如特征计算、数据处理等。

    入口脚本 (scripts): 是用户与系统交互的界面，负责设置环境、配置路径并调用相应模块的功能。

    输入/输出 (data, output): 清晰地分离了输入数据和所有生成的结果。

----------------------------------------------------------------------------

重构过程中的适应性修改说明 (针对 core 和 utils 模块)

目标: 在将原始代码迁移到新的模块化结构时，保持核心预测和计算逻辑不变，同时解决因模块化、环境变化或数据问题引发的错误，并提高代码健壮性。

主要修改类别:

    导入路径调整:

        修改内容: 将所有跨顶层目录（config, core, utils, training 等）的相对导入（如 from ..config import ...）修改为基于项目根目录的绝对导入（如 from config import ...）。同目录或子目录内的导入可以使用相对导入 (from . import ...) 或绝对导入。

        涉及文件: core/plan_node.py, core/onnx_manager.py, utils/feature_engineering.py, 以及所有 training 子模块下的 .py 文件。

        原因: 解决 Python 相对导入无法跨越顶层包边界的问题，确保在通过 scripts 目录下的入口脚本运行时能正确找到模块。

    配置驱动的模型调用:

        修改内容: 在 core/plan_node.py 的 infer_exec_with_onnx 和 infer_mem_with_onnx 方法开头，增加了检查逻辑。该逻辑首先判断当前算子类型 (self.operator_type) 是否存在于 config/structure.py 中定义的相应列表 (dop_operators_exec, no_dop_operators_exec, dop_operators_mem, no_dop_operators_mem) 内。

        行为变化:

            如果算子在列表中: 则继续执行后续的特征获取和 ONNX 模型调用逻辑（保持原始行为）。

            如果算子不在列表中: 则不再尝试调用 ONNXModelManager 的 infer_* 方法，而是直接为该算子的 pred_execution_time 或 pred_mem 赋予一个预定义的默认值，然后方法提前返回。

        涉及文件: core/plan_node.py (infer_exec_with_onnx, infer_mem_with_onnx 方法)。

        原因: 严格遵循 config/structure.py 中的配置，避免为明确配置为不需要模型预测的算子调用模型而导致 ValueError: No model found 错误。使代码行为与配置意图一致。

    增强的错误处理和健壮性:

        修改内容: 在多个关键位置增加了 try...except 块，用于捕获潜在的运行时错误，例如：

            utils/feature_engineering.py 中的特征计算 (calculate_index_cost, extract_predicate_cost) 和 Numpy 数组转换。

            core/plan_node.py 中的 ONNX 模型推理 (infer_exec/mem_with_onnx) 调用，捕获 ValueError (模型未找到) 和其他通用异常。

            core/plan_node.py 中的数值计算 (compute_pred_exec, interpolate_true_dop)，捕获 OverflowError, ValueError。

        行为变化: 当发生这些错误时，程序不再直接崩溃，而是会打印警告或错误信息，并尝试使用预定义的默认值继续执行。

        涉及文件: utils/feature_engineering.py, core/plan_node.py。

        原因: 提高代码在面对不完整模型、异常数据或计算问题时的容错能力。

    数据类型检查与处理:

        修改内容: 在可能出现类型问题的地方（尤其是处理从 plan_data 获取的值时）增加了 isinstance 类型检查，并在必要时进行转换（如 int(), float()）或赋予默认值。特别是在将变量用作字典键或集合元素之前，确保其为可哈希类型。在将特征列表转为 Numpy 数组时，处理了 None 值并指定了 dtype。

        涉及文件: core/plan_node.py (__init__, infer_*, compute_* 等), utils/feature_engineering.py (prepare_inference_data)。

        原因: 解决因数据加载或处理过程中意外的类型（如 list, None, NaN）导致的 TypeError 或 ValueError。

    路径参数化 (ONNXModelManager):

        修改内容: (根据我们的建议) 修改了 core/onnx_manager.py 的 __init__ 方法，使其可以接受模型目录的路径作为参数，而不是硬编码绝对路径。

        涉及文件: core/onnx_manager.py, 以及所有调用 ONNXModelManager() 的地方（主要是 scripts 下的脚本和 inference/predict_queries.py）现在需要传递这些路径参数。

        原因: 提高项目的可配置性和可移植性。

未改变的核心逻辑:

    算子特征的具体定义 (config/structure.py 中的 *_operator_features)。

    用于预测的模型算法本身（XGBoost 或 PyTorch 曲线拟合）。

    查询执行时间和内存的聚合计算方法 (calculate_query_execution_time, calculate_query_memory)。

    DOP 感知算子的多 DOP 预测和插值逻辑。