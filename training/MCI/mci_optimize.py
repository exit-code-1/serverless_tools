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
from typing import List, Dict, Any

from mci_moo import optimize_single_pipeline_dop, select_best_dop_config, PipelineInfo
from mci_data_loader import load_mci_pipeline_data
from mci_model import load_pytorch_model
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
    # 从config/structure.py获取DOP候选集
    from config.structure_config import dop_sets
    
    # 加载pipeline数据
    pipeline_data = load_mci_pipeline_data(
        csv_file=config.data.test_csv,
        query_info_file=config.data.test_info_csv,
        target_dop_levels=[1],  # 只需要基础数据，DOP会在优化中确定
        use_estimates=config.training.use_estimates
    )
    
    pipeline_infos = []
    
    for data in pipeline_data:
        # 检查是否有非并行算子
        has_non_parallel = data.pipeline_metadata.get('has_non_parallel', False)
        
        # 确定候选DOP
        if has_non_parallel:
            candidate_dops = [1]  # 非并行算子只能使用DOP=1
        else:
            candidate_dops = list(dop_sets)
        
        # 准备特征数据
        features = {
            'x': data.x.numpy(),
            'edge_index': data.edge_index.numpy(),
            'batch': data.batch.numpy()
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
    
    return pipeline_infos


def run_moo_optimization(config: MCIConfig, 
                        pytorch_model_path: str,
                        output_dir: str,
                        population_size: int = 50,
                        generations: int = 100,
                        weight_latency: float = 0.7,
                        weight_cost: float = 0.3) -> Dict[str, Any]:
    """
    运行MOO优化
    
    Args:
        config: MCI配置
        onnx_model_path: ONNX模型路径
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
        
        # 为每个pipeline独立运行优化
        pipeline_results = []
        total_query_latency = 0.0
        total_query_cost = 0.0
        combined_dop_config = {}
        
        for pipeline in pipelines:
            print(f"  Optimizing pipeline: {pipeline.pipeline_id}")
            
            # 运行单个pipeline的NSGA-II优化
            pareto_front = optimize_single_pipeline_dop(
                pipeline=pipeline,
                pytorch_model_path=pytorch_model_path,
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
                'pareto_front_size': len(pareto_front),
                'best_solution': {
                    'dop_config': best_solution.dop_config,
                    'latency': best_solution.latency,
                    'cost': best_solution.cost
                },
                'pareto_front': [
                    {
                        'dop_config': sol.dop_config,
                        'latency': sol.latency,
                        'cost': sol.cost
                    }
                    for sol in pareto_front
                ]
            }
            pipeline_results.append(pipeline_result)
            
            print(f"    Best solution: latency={best_solution.latency:.2f}, cost={best_solution.cost:.2f}")
            print(f"    DOP config: {best_solution.dop_config}")
        
        optimization_time = time.time() - start_time
        
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
        
        print(f"  Query total: latency={total_query_latency:.2f}, cost={total_query_cost:.2f}")
        print(f"  Combined DOP config: {combined_dop_config}")
        print(f"  Optimization time: {optimization_time:.2f}s")
    
    # 保存所有结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    results_file = os.path.join(output_dir, "moo_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
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
        population_size=50,
        generations=100,
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
