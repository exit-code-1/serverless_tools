"""
MCI-based Fine-Grained Modeling Framework for Pipeline Latency Prediction
基于MCI的细粒度建模框架，用于pipeline延迟预测

主要组件：
- mci_model.py: MCI模型架构
- mci_data_loader.py: 数据加载器
- mci_train.py: 训练脚本
"""

from .mci_model import MCILatencyModel, create_mci_model, mci_loss, export_mci_model_to_onnx, predict_and_evaluate_mci_onnx
from .mci_data_loader import (
    load_mci_pipeline_data, 
    create_multi_dop_training_data,
    create_mci_data_loaders,
    analyze_data_statistics,
    build_pipeline_from_plan_nodes,
    is_streaming_operator,
    assign_thread_ids_by_plan_id,
    has_non_parallel_operators,
    should_skip_pipeline
)
from .mci_train import MCITrainer, evaluate_model, save_model_and_results
from .mci_moo import (
    NSGA2Optimizer,
    PipelineInfo,
    MOOSolution,
    optimize_pipeline_dops,
    select_best_dop_config
)

__all__ = [
    'MCILatencyModel',
    'create_mci_model', 
    'mci_loss',
    'export_mci_model_to_onnx',
    'predict_and_evaluate_mci_onnx',
    'load_mci_pipeline_data',
    'create_multi_dop_training_data',
    'create_mci_data_loaders',
    'analyze_data_statistics',
    'build_pipeline_from_plan_nodes',
    'is_streaming_operator',
    'assign_thread_ids_by_plan_id',
    'has_non_parallel_operators',
    'should_skip_pipeline',
    'MCITrainer',
    'evaluate_model',
    'save_model_and_results',
    'NSGA2Optimizer',
    'PipelineInfo',
    'MOOSolution',
    'optimize_pipeline_dops',
    'select_best_dop_config'
]
