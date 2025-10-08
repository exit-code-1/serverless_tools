"""
MCI Multi-Objective Optimization (MOO) using NSGA-II
MCI多目标优化算法，使用pymoo库的NSGA-II优化pipeline的DOP选择

目标：
1. 最小化延迟 (latency)
2. 最小化成本 (cost = latency * DOP)
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import onnxruntime as ort

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not available. Please install with: pip install pymoo")


@dataclass
class PipelineInfo:
    """Pipeline信息"""
    pipeline_id: str
    query_id: str
    candidate_dops: List[int]
    current_dop: int
    latency_model: Any  # ONNX model for latency prediction
    features: Dict[str, Any]  # Pipeline features for prediction


@dataclass
class MOOSolution:
    """MOO解"""
    dop_config: Dict[str, int]  # pipeline_id -> DOP
    latency: float
    cost: float


class PipelineDOPProblem(Problem):
    """Pipeline DOP优化问题定义"""
    
    def __init__(self, pipelines: List[PipelineInfo]):
        self.pipelines = pipelines
        self.pipeline_ids = [p.pipeline_id for p in pipelines]
        
        # 定义变量边界：每个pipeline的DOP选择
        n_vars = len(pipelines)
        xl = np.zeros(n_vars)  # 最小DOP索引
        xu = np.array([len(p.candidate_dops) - 1 for p in pipelines])  # 最大DOP索引
        
        # 2个目标：延迟和成本
        super().__init__(n_var=n_vars, n_obj=2, xl=xl, xu=xu, vtype=int)
    
    def _evaluate(self, x, out, *args, **kwargs):
        """评估解的目标函数值"""
        n_solutions = x.shape[0]
        f = np.zeros((n_solutions, 2))
        
        for i in range(n_solutions):
            total_latency = 0.0
            total_cost = 0.0
            
            for j, pipeline in enumerate(self.pipelines):
                # 获取DOP索引并转换为实际DOP值
                dop_idx = int(x[i, j])
                dop = pipeline.candidate_dops[dop_idx]
                
                # 预测延迟
                latency = self._predict_latency(pipeline, dop)
                
                # 计算成本
                cost = latency * dop
                
                total_latency += latency
                total_cost += cost
            
            f[i, 0] = total_latency  # 目标1：最小化延迟
            f[i, 1] = total_cost     # 目标2：最小化成本
        
        out["F"] = f
    
    def _predict_latency(self, pipeline: PipelineInfo, dop: int) -> float:
        """预测pipeline在指定DOP下的延迟"""
        try:
            # 准备ONNX输入
            ort_inputs = {
                "x": pipeline.features['x'].astype(np.float32),
                "edge_index": pipeline.features['edge_index'].astype(np.int64),
                "batch": pipeline.features['batch'].astype(np.int64),
                "dop_levels": np.array([dop], dtype=np.float32)
            }
            
            # 运行推理
            outputs = pipeline.latency_model.run(["latency_predictions"], ort_inputs)
            latency = outputs[0].flatten()[0]
            
            return max(latency, 0.1)  # 确保延迟为正数
            
        except Exception as e:
            print(f"Error predicting latency for pipeline {pipeline.pipeline_id}: {e}")
            return 100.0  # 返回一个较大的默认值


class NSGA2Optimizer:
    """NSGA-II多目标优化器（基于pymoo库）"""
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        
    def optimize(self, pipelines: List[PipelineInfo]) -> List[MOOSolution]:
        """
        使用NSGA-II优化pipeline的DOP配置
        
        Args:
            pipelines: Pipeline信息列表
            
        Returns:
            帕累托最优解集
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo library is required. Install with: pip install pymoo")
        
        # 创建优化问题
        problem = PipelineDOPProblem(pipelines)
        
        # 配置NSGA-II算法
        algorithm = NSGA2(
            pop_size=self.population_size,
            n_offsprings=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.8, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # 设置终止条件
        termination = get_termination("n_gen", self.generations)
        
        # 运行优化
        print(f"Starting NSGA-II optimization with {self.population_size} individuals for {self.generations} generations...")
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=True
        )
        
        # 转换结果为MOOSolution格式
        pareto_front = []
        for i in range(len(res.F)):
            # 将DOP索引转换为实际DOP值
            dop_config = {}
            for j, pipeline in enumerate(pipelines):
                dop_idx = int(res.X[i, j])
                dop = pipeline.candidate_dops[dop_idx]
                dop_config[pipeline.pipeline_id] = dop
            
            solution = MOOSolution(
                dop_config=dop_config,
                latency=res.F[i, 0],
                cost=res.F[i, 1]
            )
            pareto_front.append(solution)
        
        print(f"Found {len(pareto_front)} Pareto optimal solutions")
        return pareto_front


def optimize_single_pipeline_dop(pipeline: PipelineInfo, 
                                onnx_model_path: str,
                                population_size: int = 50,
                                generations: int = 100) -> List[MOOSolution]:
    """
    优化单个pipeline的DOP配置
    
    Args:
        pipeline: 单个Pipeline信息
        onnx_model_path: ONNX模型路径
        population_size: 种群大小
        generations: 进化代数
        
    Returns:
        帕累托最优解集
    """
    # 加载ONNX模型
    session = ort.InferenceSession(onnx_model_path)
    
    # 为pipeline设置模型
    pipeline.latency_model = session
    
    # 创建优化器
    optimizer = NSGA2Optimizer(
        population_size=population_size,
        generations=generations
    )
    
    # 运行优化（单个pipeline）
    pareto_front = optimizer.optimize([pipeline])
    
    return pareto_front


def optimize_pipeline_dops(pipelines: List[PipelineInfo], 
                          onnx_model_path: str,
                          population_size: int = 50,
                          generations: int = 100) -> List[MOOSolution]:
    """
    优化pipeline的DOP配置（全局优化版本，保留用于兼容性）
    
    Args:
        pipelines: Pipeline信息列表
        onnx_model_path: ONNX模型路径
        population_size: 种群大小
        generations: 进化代数
        
    Returns:
        帕累托最优解集
    """
    # 加载ONNX模型
    session = ort.InferenceSession(onnx_model_path)
    
    # 为每个pipeline设置模型
    for pipeline in pipelines:
        pipeline.latency_model = session
    
    # 创建优化器
    optimizer = NSGA2Optimizer(
        population_size=population_size,
        generations=generations
    )
    
    # 运行优化
    pareto_front = optimizer.optimize(pipelines)
    
    return pareto_front


def select_best_dop_config(pareto_front: List[MOOSolution], 
                          weight_latency: float = 0.7,
                          weight_cost: float = 0.3) -> MOOSolution:
    """
    从帕累托前沿中选择最佳解
    
    Args:
        pareto_front: 帕累托前沿
        weight_latency: 延迟权重
        weight_cost: 成本权重
        
    Returns:
        最佳解
    """
    if not pareto_front:
        raise ValueError("Empty Pareto front")
    
    # 归一化目标值
    latencies = [sol.latency for sol in pareto_front]
    costs = [sol.cost for sol in pareto_front]
    
    min_latency, max_latency = min(latencies), max(latencies)
    min_cost, max_cost = min(costs), max(costs)
    
    # 计算加权得分
    best_solution = None
    best_score = float('inf')
    
    for solution in pareto_front:
        # 归一化
        norm_latency = (solution.latency - min_latency) / (max_latency - min_latency + 1e-8)
        norm_cost = (solution.cost - min_cost) / (max_cost - min_cost + 1e-8)
        
        # 加权得分（越小越好）
        score = weight_latency * norm_latency + weight_cost * norm_cost
        
        if score < best_score:
            best_score = score
            best_solution = solution
    
    return best_solution
