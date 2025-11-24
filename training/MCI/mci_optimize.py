"""
MCI MOO Optimization Entry Point
MCI多目标优化入口脚本

使用NSGA-II算法优化pipeline的DOP配置
"""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any

from mci_moo import optimize_single_pipeline_dop, select_best_dop_config, MOOSolution, PipelineInfo
from mci_data_loader import load_mci_pipeline_data_exec
from config import MCIConfig


def load_pipeline_info_for_moo(config: MCIConfig, pytorch_model_path: str) -> List[PipelineInfo]:
    """
    加载pipeline信息用于MOO优化
    
    Args:
        config: MCI配置
        pytorch_model_path: PyTorch模型路径
        
    Returns:
        Pipeline信息列表
    """
    # 从config/main_config.py获取DOP候选集
    from config.main_config import DOP_SETS as dop_sets
    
    # 加载pipeline数据 (使用执行时间数据)
    # 使用 target_dop_levels=None 加载所有DOP的数据，然后我们会为每个pipeline选择一个基准DOP
    pipeline_data = load_mci_pipeline_data_exec(
        csv_file=config.data.test_csv,
        query_info_file=config.data.test_info_csv,
        target_dop_levels=None,  # 加载所有DOP的数据
        use_estimates=config.training.use_estimates
    )
    
    # 去重：对于同一个pipeline（相同query_id和thread_id），只保留一份
    seen_pipelines = set()
    unique_pipeline_data = []
    for data in pipeline_data:
        pipeline_key = (data.pipeline_metadata['query_id'], data.pipeline_metadata['pipeline_id'])
        if pipeline_key not in seen_pipelines:
            seen_pipelines.add(pipeline_key)
            unique_pipeline_data.append(data)
    
    pipeline_data = unique_pipeline_data
    print(f"Loaded {len(pipeline_data)} unique pipelines from data")
    
    pipeline_infos = []
    skipped_non_parallel = 0
    skipped_root = 0
    
    for data in pipeline_data:
        # 检查是否有非并行算子
        has_non_parallel = data.pipeline_metadata.get('has_non_parallel', False)
        # 检查是否是根pipeline（最上层）
        is_root_pipeline = data.pipeline_metadata.get('is_root_pipeline', False)
        
        # 确定候选DOP范围
        if has_non_parallel:
            # 非并行pipeline直接跳过，不加入优化列表
            # 稍后会在结果中直接设置DOP=1
            skipped_non_parallel += 1
            continue
        elif is_root_pipeline:
            # 根pipeline必须DOP=1（最终结果需要汇聚到单线程）
            # 跳过优化，稍后直接设置DOP=1
            skipped_root += 1
            continue
        else:
            # 可并行pipeline：DOP在 [8, 96] 连续区间内优化
            # candidate_dops 用于设置边界
            candidate_dops = [8, 96]  # [min_dop, max_dop]
        
        # 准备特征数据
        # For single graph data, batch is None, so we create it manually
        # All nodes belong to batch 0 for a single graph
        if data.batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
        else:
            batch = data.batch
        
        features = {
            'x': data.x.numpy(),
            'edge_index': data.edge_index.numpy(),
            'batch': batch.numpy()
        }
        
        pipeline_info = PipelineInfo(
            pipeline_id=data.pipeline_metadata['pipeline_id'],
            query_id=data.pipeline_metadata['query_id'],
            candidate_dops=candidate_dops,
            current_dop=data.dop_level.item(),
            latency_model=None,  # 将在optimize_pipeline_dops中设置
            features=features
        )
        
        pipeline_infos.append(pipeline_info)
    
    print(f"Loaded {len(pipeline_infos)} pipelines for optimization")
    print(f"Skipped {skipped_non_parallel} non-parallel pipelines (will use DOP=1)")
    
    return pipeline_infos


def generate_operators_optimized_csv(config: MCIConfig, 
                                    all_results: Dict[str, Any],
                                    output_dir: str):
    """
    Generate operators_optimized.csv file with optimized DOP for each operator
    
    Args:
        config: MCI configuration
        all_results: MOO optimization results containing DOP configurations
        output_dir: Output directory
    """
    from utils.data_utils import load_csv_safe, build_query_plan
    from mci_data_loader import build_pipeline_from_plan_nodes
    
    # Read original plan_info.csv
    plan_info_file = config.data.test_csv
    print(f"Reading plan_info from: {plan_info_file}")
    
    # Read the CSV file with semicolon separator
    df = load_csv_safe(plan_info_file, description="Plan info")
    if df is None:
        print("Error: Could not load plan_info.csv")
        return
    
    # Build pipeline_id -> optimized_dop mapping
    pipeline_dop_mapping = {}
    for query_id, results in all_results.items():
        dop_config = results['combined_dop_config']
        for pipeline_id, dop in dop_config.items():
            pipeline_dop_mapping[pipeline_id] = dop
    
    print(f"Found {len(pipeline_dop_mapping)} optimized pipeline DOPs")
    
    # Build mapping from (query_id, plan_id) to optimized_dop and parent-child info
    # We need to reconstruct pipelines to map operators to pipelines
    operator_dop_mapping = {}
    operator_parent_mapping = {}  # (query_id, plan_id) -> parent_plan_id
    operator_left_child_mapping = {}  # (query_id, plan_id) -> left_child_plan_id
    
    # Use base_dop (typically 64) as baseline for optimization
    # Only process query plans that were executed with base_dop
    from config.main_config import DEFAULT_DOP
    base_dop = DEFAULT_DOP  # Usually 64
    
    print(f"Selecting query plans with base_dop={base_dop}...")
    
    # Filter dataframe to only include rows with base_dop
    df_filtered = df[df['query_dop'] == base_dop].copy()
    print(f"Filtered from {len(df)} to {len(df_filtered)} operators (using base_dop={base_dop} only)")
    
    if len(df_filtered) == 0:
        print(f"Warning: No data found with query_dop={base_dop}. Available query_dops: {sorted(df['query_dop'].unique())}")
        print("Using first available query_dop for each query instead...")
        # Fallback: use first query_dop for each query
        df_filtered = df.drop_duplicates(subset=['query_id', 'plan_id']).copy()
    
    # Group by query_id and query_dop to rebuild pipelines
    grouped = df_filtered.groupby(['query_id', 'query_dop'])
    
    for (query_id, query_dop), group in grouped:
        # Build plan tree
        nodes, root_nodes = build_query_plan(group, use_estimates=config.training.use_estimates)
        
        if not nodes:
            continue
        
        # Build pipelines
        pipelines = build_pipeline_from_plan_nodes(nodes, root_nodes, None)
        
        # Map operators to their pipeline's optimized DOP
        for pipeline in pipelines:
            pipeline_id = f"pipeline_{pipeline['thread_id']}"
            
            # Check if this pipeline has non-parallel operators or is root pipeline
            has_non_parallel = pipeline.get('has_non_parallel', False)
            is_root_pipeline = pipeline.get('is_root_pipeline', False)
            
            if has_non_parallel or is_root_pipeline:
                # Non-parallel or root pipeline: force DOP=1
                optimized_dop = 1
            else:
                # Use optimized DOP from MOO, or original DOP if not optimized
                optimized_dop = pipeline_dop_mapping.get(pipeline_id, pipeline['dop'])
            
            # Assign optimized DOP to all operators in this pipeline
            for node in pipeline['nodes']:
                operator_key = (query_id, node.plan_id)
                operator_dop_mapping[operator_key] = optimized_dop
                
                # Get parent-child info from node structure (already built in add_child)
                if node.parent_node is not None:
                    operator_parent_mapping[operator_key] = node.parent_node.plan_id
                else:
                    operator_parent_mapping[operator_key] = -1
                
                # Get left child (first child)
                if len(node.child_plans) > 0:
                    operator_left_child_mapping[operator_key] = node.child_plans[0].plan_id
                else:
                    operator_left_child_mapping[operator_key] = -1
    
    print(f"Mapped {len(operator_dop_mapping)} operators to optimized DOPs")
    
    # Count statistics for different types of operators
    num_root_or_non_parallel = 0
    num_optimized = 0
    num_unchanged = 0
    
    for key in operator_dop_mapping.keys():
        query_id, plan_id = key
        optimized_dop = operator_dop_mapping[key]
        # Find original DOP
        matching_rows = df[(df['query_id'] == query_id) & (df['plan_id'] == plan_id)]
        if not matching_rows.empty:
            original_dop = matching_rows.iloc[0]['dop']
            if optimized_dop == 1 and original_dop != 1:
                num_root_or_non_parallel += 1
            elif optimized_dop != original_dop:
                num_optimized += 1
            else:
                num_unchanged += 1
    
    print(f"  Root/non-parallel pipelines (DOP forced to 1): {num_root_or_non_parallel} operators")
    print(f"  Optimized by MOO: {num_optimized} operators")
    print(f"  Unchanged: {num_unchanged} operators")
    
    # Apply optimized DOPs and parent-child relationships to dataframe
    df['optimized_dop'] = df.apply(
        lambda row: operator_dop_mapping.get((row['query_id'], row['plan_id']), row['dop']),
        axis=1
    )
    
    df['parent_child'] = df.apply(
        lambda row: operator_parent_mapping.get((row['query_id'], row['plan_id']), -1),
        axis=1
    )
    
    df['left_child'] = df.apply(
        lambda row: operator_left_child_mapping.get((row['query_id'], row['plan_id']), -1),
        axis=1
    )
    
    # Select and rename columns to match output format
    output_df = pd.DataFrame({
        'query_id': df['query_id'],
        'plan_id': df['plan_id'],
        'operator_type': df['operator_type'],
        'width': df['width'],
        'dop': df['optimized_dop'].astype(int),
        'left_child': df['left_child'].astype(int),
        'parent_child': df['parent_child'].astype(int)
    })
    
    # Save to CSV
    output_file = os.path.join(output_dir, "operators_optimized.csv")
    output_df.to_csv(output_file, index=False)
    
    print(f"\noperators_optimized.csv saved to: {output_file}")
    print(f"  Total operators: {len(output_df)}")


def run_moo_optimization(config: MCIConfig, 
                        pytorch_model_path: str,
                        output_dir: str,
                        population_size: int = 20,
                        generations: int = 15,
                        weight_latency: float = 0.7,
                        weight_cost: float = 0.3) -> Dict[str, Any]:
    """
    运行MOO优化
    
    Args:
        config: MCI配置
        pytorch_model_path: PyTorch模型路径
        output_dir: 输出目录
        population_size: 种群大小
        generations: 进化代数
        weight_latency: 延迟权重
        weight_cost: 成本权重
        
    Returns:
        优化结果
    """
    print("Loading pipeline information for MOO optimization...")
    pipeline_infos = load_pipeline_info_for_moo(config, pytorch_model_path)
    
    print(f"Loaded {len(pipeline_infos)} pipelines for optimization")
    
    # 🚀 优化1：预先加载模型，避免每个pipeline重复加载
    print("Loading PyTorch model (only once)...")
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    
    from mci_model import create_mci_model
    shared_model = create_mci_model(
        input_dim=model_config.get('input_dim'),
        hidden_dim=model_config.get('hidden_dim'),
        embedding_dim=model_config.get('embedding_dim'),
        num_heads=model_config.get('num_heads'),
        num_layers=model_config.get('num_layers'),
        predictor_hidden_dims=model_config.get('predictor_hidden_dims'),
        dropout=model_config.get('dropout')
    )
    shared_model.load_state_dict(checkpoint['model_state_dict'])
    shared_model.eval()
    
    # 为所有pipeline设置共享模型
    for pipeline_info in pipeline_infos:
        pipeline_info.latency_model = shared_model
    
    print("Model loaded and shared across all pipelines")
    
    # 按query分组
    query_pipelines = {}
    for pipeline_info in pipeline_infos:
        query_id = pipeline_info.query_id
        if query_id not in query_pipelines:
            query_pipelines[query_id] = []
        query_pipelines[query_id].append(pipeline_info)
    
    print(f"Found {len(query_pipelines)} queries")
    
    # 为每个query运行优化
    all_results = {}
    
    for query_id, pipelines in query_pipelines.items():
        print(f"\nOptimizing query: {query_id} with {len(pipelines)} pipelines")
        
        start_time = time.time()
        
        # 为每个pipeline独立运行MOO优化（对比基线方法）
        # 这样可以证明stage级别的优化不如考虑整个query的方法
        pipeline_results = []
        total_query_latency = 0.0
        total_query_cost = 0.0
        combined_dop_config = {}
        
        for pipeline in pipelines:
            print(f"  Optimizing pipeline: {pipeline.pipeline_id}")
            
            # 运行单个pipeline的NSGA-II优化
            # DOP在连续区间 [min_dop, max_dop] 内搜索 (如 [8, 96])
            # 模型已预先加载，无需重复加载
            pareto_front = optimize_single_pipeline_dop(
                pipeline=pipeline,
                population_size=population_size,
                generations=generations
            )
            
            # 选择最佳解
            best_solution = select_best_dop_config(
                pareto_front=pareto_front,
                weight_latency=weight_latency,
                weight_cost=weight_cost
            )
            
            # 累加到query总延迟和成本
            total_query_latency += best_solution.latency
            total_query_cost += best_solution.cost
            
            # 合并DOP配置
            combined_dop_config.update(best_solution.dop_config)
            
            # 保存单个pipeline的结果
            pipeline_result = {
                'pipeline_id': pipeline.pipeline_id,
                'optimized_dop': best_solution.dop_config.get(pipeline.pipeline_id, pipeline.current_dop),
                'latency': best_solution.latency,
                'cost': best_solution.cost,
                'pareto_front_size': len(pareto_front)
            }
            pipeline_results.append(pipeline_result)
            
            print(f"    Pipeline {pipeline.pipeline_id}: DOP={best_solution.dop_config}, latency={best_solution.latency:.2f}, cost={best_solution.cost:.2f}")
        
        optimization_time = time.time() - start_time
        
        print(f"  Query optimization completed:")
        print(f"    Total latency (sum of pipelines): {total_query_latency:.2f}")
        print(f"    Total cost (sum of pipelines): {total_query_cost:.2f}")
        print(f"    Combined DOP config: {combined_dop_config}")
        print(f"    Optimization time: {optimization_time:.2f}s")
        
        # 保存query级别结果
        query_results = {
            'query_id': query_id,
            'num_pipelines': len(pipelines),
            'total_latency': total_query_latency,
            'total_cost': total_query_cost,
            'combined_dop_config': combined_dop_config,
            'optimization_time': optimization_time,
            'pipeline_results': pipeline_results
        }
        
        all_results[query_id] = query_results
    
    # 保存所有结果
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {convert_to_json_serializable(k): convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # 保存详细结果
    results_file = os.path.join(output_dir, "moo_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable(all_results), f, indent=2)
    
    # 保存摘要
    summary_data = []
    for query_id, results in all_results.items():
        summary_data.append({
            'query_id': query_id,
            'num_pipelines': results['num_pipelines'],
            'total_latency': results['total_latency'],
            'total_cost': results['total_cost'],
            'optimization_time': results['optimization_time']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "moo_optimization_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # 计算总体统计
    total_queries = len(all_results)
    
    if total_queries == 0:
        print("Warning: No queries were optimized. Please check your data and configuration.")
        return {
            'total_queries': 0,
            'total_pipelines': 0,
            'avg_total_latency': 0,
            'avg_total_cost': 0,
            'total_optimization_time': 0,
            'avg_optimization_time_per_query': 0
        }
    
    total_pipelines = sum(r['num_pipelines'] for r in all_results.values())
    avg_latency = np.mean([r['total_latency'] for r in all_results.values()])
    avg_cost = np.mean([r['total_cost'] for r in all_results.values()])
    total_optimization_time = sum(r['optimization_time'] for r in all_results.values())
    
    summary_stats = {
        'total_queries': total_queries,
        'total_pipelines': total_pipelines,
        'avg_total_latency': avg_latency,
        'avg_total_cost': avg_cost,
        'total_optimization_time': total_optimization_time,
        'avg_optimization_time_per_query': total_optimization_time / total_queries
    }
    
    # 保存统计摘要
    stats_file = os.path.join(output_dir, "moo_optimization_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nMOO optimization completed!")
    print(f"  Total queries: {total_queries}")
    print(f"  Total pipelines: {total_pipelines}")
    print(f"  Average total latency: {avg_latency:.2f}")
    print(f"  Average total cost: {avg_cost:.2f}")
    print(f"  Total optimization time: {total_optimization_time:.2f}s")
    print(f"  Results saved to: {output_dir}")
    
    # Generate operators_optimized.csv
    print("\nGenerating operators_optimized.csv...")
    generate_operators_optimized_csv(
        config=config,
        all_results=all_results,
        output_dir=output_dir
    )
    
    return summary_stats


def main():
    parser = argparse.ArgumentParser(description='MCI MOO Optimization')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MCI MOO OPTIMIZATION")
    print("="*60)
    print(f"Config file: {args.config}")
    print("="*60)
    
    # 加载配置
    config = MCIConfig.load_config(args.config)
    
    # 运行优化
    start_time = time.time()
    results = run_moo_optimization(
        config=config,
        pytorch_model_path=config.data.model_name + '.pth',
        output_dir=config.data.output_dir,
        population_size=20,
        generations=15,
        weight_latency=0.7,
        weight_cost=0.3
    )
    total_time = time.time() - start_time
    
    # 打印最终摘要
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Total queries: {results['total_queries']}")
    print(f"Total pipelines: {results['total_pipelines']}")
    print(f"Average total latency: {results['avg_total_latency']:.2f}")
    print(f"Average total cost: {results['avg_total_cost']:.2f}")
    print(f"Total optimization time: {results['total_optimization_time']:.2f}s")
    print(f"Average time per query: {results['avg_optimization_time_per_query']:.2f}s")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Results saved to: {config.data.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
