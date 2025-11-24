"""
MCI Multi-Objective Optimization (MOO) using NSGA-II
MCI multi-objective optimization algorithm, using pymoo library's NSGA-II to optimize pipeline DOP selection

Objectives:
1. Minimize latency
2. Minimize cost (cost = latency * DOP)
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import torch

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
    """Pipeline information"""
    pipeline_id: str
    query_id: str
    candidate_dops: List[int]
    current_dop: int
    latency_model: Any  # PyTorch model for latency prediction
    features: Dict[str, Any]  # Pipeline features for prediction


@dataclass
class MOOSolution:
    """MOO solution"""
    dop_config: Dict[str, int]  # pipeline_id -> DOP
    latency: float
    cost: float


def round_to_even(value: float) -> int:
    """Round a float value to the nearest even integer"""
    rounded = int(round(value))
    # If odd, round to nearest even
    if rounded % 2 == 1:
        # Check which even neighbor is closer
        if value - (rounded - 1) < (rounded + 1) - value:
            rounded = rounded - 1
        else:
            rounded = rounded + 1
    return rounded


class PipelineDOPProblem(Problem):
    """Pipeline DOP optimization problem definition"""
    
    def __init__(self, pipelines: List[PipelineInfo]):
        self.pipelines = pipelines
        self.pipeline_ids = [p.pipeline_id for p in pipelines]
        
        # Define variable boundaries: DOP as continuous variable in [8, 96]
        # For pipelines that can't be parallelized, min_dop would be 1
        n_vars = len(pipelines)
        xl = np.array([min(p.candidate_dops) for p in pipelines])  # Min DOP for each pipeline
        xu = np.array([max(p.candidate_dops) for p in pipelines])  # Max DOP for each pipeline
        
        # 2 objectives: latency and cost
        # Use float type for continuous DOP optimization
        super().__init__(n_var=n_vars, n_obj=2, xl=xl, xu=xu, vtype=float)
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective function values for solutions"""
        n_solutions = x.shape[0]
        f = np.zeros((n_solutions, 2))
        
        for i in range(n_solutions):
            total_latency = 0.0
            total_cost = 0.0
            
            for j, pipeline in enumerate(self.pipelines):
                # DOP is now a continuous value, round to nearest even integer
                dop = round_to_even(x[i, j])
                
                # Clamp to valid range
                dop = max(min(pipeline.candidate_dops), min(dop, max(pipeline.candidate_dops)))
                
                # Ensure DOP is even (after clamping, might need to adjust)
                if dop % 2 == 1 and dop > min(pipeline.candidate_dops):
                    dop -= 1  # Round down to even
                
                # Predict latency
                latency = self._predict_latency(pipeline, dop)
                
                # Calculate cost
                cost = latency * dop
                
                total_latency += latency
                total_cost += cost
            
            f[i, 0] = total_latency  # Objective 1: minimize latency
            f[i, 1] = total_cost     # Objective 2: minimize cost
        
        out["F"] = f
    
    def _predict_latency(self, pipeline: PipelineInfo, dop: int) -> float:
        """Predict pipeline latency at specified DOP"""
        try:
            # Prepare PyTorch inputs
            x = torch.from_numpy(pipeline.features['x']).float()
            edge_index = torch.from_numpy(pipeline.features['edge_index']).long()
            batch = torch.from_numpy(pipeline.features['batch']).long()
            dop_level = torch.tensor([dop], dtype=torch.float32)
            
            # Move to same device as model
            device = next(pipeline.latency_model.parameters()).device
            x = x.to(device)
            edge_index = edge_index.to(device)
            batch = batch.to(device)
            dop_level = dop_level.to(device)
            
            # Run inference
            pipeline.latency_model.eval()
            with torch.no_grad():
                predictions, _ = pipeline.latency_model(x, edge_index, batch, dop_level)
                latency = predictions.item()
            
            return max(latency, 0.1)  # Ensure positive latency
            
        except Exception as e:
            print(f"Error predicting latency for pipeline {pipeline.pipeline_id}: {e}")
            return 100.0  # Return a large default value


class NSGA2Optimizer:
    """NSGA-II multi-objective optimizer (based on pymoo library)"""
    
    def __init__(self, 
                 population_size: int = 20,
                 generations: int = 15):
        self.population_size = population_size
        self.generations = generations
        
    def optimize(self, pipelines: List[PipelineInfo]) -> List[MOOSolution]:
        """
        Optimize pipeline DOP configuration using NSGA-II
        
        Args:
            pipelines: List of pipeline information
            
        Returns:
            Pareto optimal solution set
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo library is required. Install with: pip install pymoo")
        
        # Create optimization problem
        problem = PipelineDOPProblem(pipelines)
        
        # Configure NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=self.population_size,
            n_offsprings=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.8, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Set termination criterion
        termination = get_termination("n_gen", self.generations)
        
        # Run optimization
        print(f"Starting NSGA-II optimization with {self.population_size} individuals for {self.generations} generations...")
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=True
        )
        
        # Convert results to MOOSolution format
        pareto_front = []
        for i in range(len(res.F)):
            # DOP is continuous, round to nearest even integer
            dop_config = {}
            for j, pipeline in enumerate(pipelines):
                dop_value = res.X[i, j]
                # Round to nearest even integer
                dop = round_to_even(dop_value)
                # Clamp to valid range
                dop = max(min(pipeline.candidate_dops), min(dop, max(pipeline.candidate_dops)))
                # Ensure DOP is even after clamping
                if dop % 2 == 1 and dop > min(pipeline.candidate_dops):
                    dop -= 1  # Round down to even
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
                                pytorch_model_path: str = None,
                                population_size: int = 20,
                                generations: int = 15) -> List[MOOSolution]:
    """
    Optimize DOP configuration for a single pipeline
    
    Args:
        pipeline: Single pipeline information (with latency_model already set)
        pytorch_model_path: PyTorch model path (deprecated, model should be pre-loaded in pipeline)
        population_size: Population size
        generations: Number of generations
        
    Returns:
        Pareto optimal solution set
    """
    # Model should already be loaded and set in pipeline.latency_model
    if pipeline.latency_model is None:
        raise ValueError(f"Model not loaded for pipeline {pipeline.pipeline_id}. Please load model first.")
    
    # Create optimizer
    optimizer = NSGA2Optimizer(
        population_size=population_size,
        generations=generations
    )
    
    # Run optimization (single pipeline)
    pareto_front = optimizer.optimize([pipeline])
    
    return pareto_front


def optimize_pipeline_dops(pipelines: List[PipelineInfo], 
                          pytorch_model_path: str,
                          population_size: int = 20,
                          generations: int = 15) -> List[MOOSolution]:
    """
    Optimize DOP configuration for pipelines (global optimization, kept for compatibility)
    
    Args:
        pipelines: List of pipeline information
        pytorch_model_path: PyTorch model path
        population_size: Population size
        generations: Number of generations
        
    Returns:
        Pareto optimal solution set
    """
    # Import model loading function
    from mci_model import MCILatencyModel
    from config import MCIConfig
    
    # Load PyTorch model
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    
    # Create model with saved configuration
    from mci_model import create_mci_model
    model = create_mci_model(
        input_dim=model_config.get('input_dim'),
        hidden_dim=model_config.get('hidden_dim'),
        embedding_dim=model_config.get('embedding_dim'),
        num_heads=model_config.get('num_heads'),
        num_layers=model_config.get('num_layers'),
        predictor_hidden_dims=model_config.get('predictor_hidden_dims'),
        dropout=model_config.get('dropout')
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Set model for each pipeline
    for pipeline in pipelines:
        pipeline.latency_model = model
    
    # Create optimizer
    optimizer = NSGA2Optimizer(
        population_size=population_size,
        generations=generations
    )
    
    # Run optimization
    pareto_front = optimizer.optimize(pipelines)
    
    return pareto_front


def select_best_dop_config(pareto_front: List[MOOSolution], 
                          weight_latency: float = 0.7,
                          weight_cost: float = 0.3) -> MOOSolution:
    """
    Select best solution from Pareto front
    
    Args:
        pareto_front: Pareto front
        weight_latency: Latency weight
        weight_cost: Cost weight
        
    Returns:
        Best solution
    """
    if not pareto_front:
        raise ValueError("Empty Pareto front")
    
    # Normalize objective values
    latencies = [sol.latency for sol in pareto_front]
    costs = [sol.cost for sol in pareto_front]
    
    min_latency, max_latency = min(latencies), max(latencies)
    min_cost, max_cost = min(costs), max(costs)
    
    # Calculate weighted score
    best_solution = None
    best_score = float('inf')
    
    for solution in pareto_front:
        # Normalize
        norm_latency = (solution.latency - min_latency) / (max_latency - min_latency + 1e-8)
        norm_cost = (solution.cost - min_cost) / (max_cost - min_cost + 1e-8)
        
        # Weighted score (lower is better)
        score = weight_latency * norm_latency + weight_cost * norm_cost
        
        if score < best_score:
            best_score = score
            best_solution = solution
    
    return best_solution
