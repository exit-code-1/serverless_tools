"""
MCI Data Loader for Pipeline Latency Prediction
MCI数据加载器，用于pipeline延迟预测

功能：
1. 基于现有的pipeline划分逻辑（streaming算子作为分界点）
2. 将DOP特征集成到算子特征中
3. 构建pipeline的图结构
4. 为不同DOP级别准备训练数据
"""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Dict
import sys
import os
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.PPM import infos, featurelize
from utils.feature_engineering import extract_predicate_cost
from utils.data_utils import build_query_plan, update_tree_estimated_inputs_recursive
from config.structure_config import jointype_encoding, operator_encoding, parallel_op, no_dop_operators_exec
from core.plan_node import PlanNode
from core.thread_block import ThreadBlock


def is_streaming_operator(operator_type: str) -> bool:
    """
    Check if an operator is a streaming operator
    
    Args:
        operator_type: Operator type string
    
    Returns:
        True if it's a streaming operator, False otherwise
    """
    return 'streaming' in operator_type.lower()


def has_non_parallel_operators(pipeline_nodes: List[PlanNode]) -> bool:
    """
    Check if a pipeline contains operators that don't support parallelization
    
    Args:
        pipeline_nodes: List of PlanNode instances in the pipeline
    
    Returns:
        True if pipeline contains non-parallel operators, False otherwise
    """
    for node in pipeline_nodes:
        # Check if the operator is in the non-parallel list
        if node.operator_type in no_dop_operators_exec:
            return True
    return False


def should_skip_pipeline(pipeline_nodes: List[PlanNode], target_dop_levels: Optional[List[int]] = None) -> bool:
    """
    Determine if a pipeline should be skipped based on parallelization support
    
    Args:
        pipeline_nodes: List of PlanNode instances in the pipeline
        target_dop_levels: List of target DOP levels for training
    
    Returns:
        True if pipeline should be skipped, False otherwise
    """
    # If pipeline contains non-parallel operators
    if has_non_parallel_operators(pipeline_nodes):
        # Only include if we're targeting DOP=1 specifically
        if target_dop_levels is None or 1 in target_dop_levels:
            # Check if all target DOP levels are 1
            if target_dop_levels is None or all(dop == 1 for dop in target_dop_levels):
                return False  # Don't skip if only training DOP=1
            else:
                return True   # Skip if training multiple DOP levels
        else:
            return True  # Skip if not targeting DOP=1
    
    return False  # Don't skip if all operators support parallelization


def assign_thread_ids_by_plan_id(node, thread_id=0, max_thread_id=0, visited=None):
    """
    Recursively assign thread IDs to plan tree based on streaming operators
    基于streaming算子递归为计划树分配线程ID
    
    Args:
        node: PlanNode instance
        thread_id: Current thread ID
        max_thread_id: Maximum thread ID used so far
        visited: Set of visited node IDs to prevent infinite recursion
    
    Returns:
        Maximum thread ID used
    """
    if visited is None:
        visited = set()
    
    # 防止循环引用
    if node.plan_id in visited:
        print(f"Warning: Detected circular reference in plan tree at node {node.plan_id}")
        return max_thread_id
    
    visited.add(node.plan_id)
    
    node.thread_id = thread_id
    max_thread_id = max(max_thread_id, thread_id)

    if is_streaming_operator(node.operator_type):
        new_thread_id = max_thread_id + 1
    else:
        new_thread_id = thread_id

    for child in node.child_plans:
        max_thread_id = assign_thread_ids_by_plan_id(child, new_thread_id, max_thread_id, visited)

    return max_thread_id


def collect_nodes_by_thread_id(nodes: List[PlanNode]) -> Dict[int, List[PlanNode]]:
    """
    Collect nodes grouped by thread ID
    
    Args:
        nodes: List of PlanNode instances
    
    Returns:
        Dictionary mapping thread_id to list of nodes
    """
    thread_groups = defaultdict(list)
    for node in nodes:
        thread_groups[node.thread_id].append(node)
    return dict(thread_groups)


def extract_enhanced_operator_features(plan_node: PlanNode, pipeline_dop: int) -> np.ndarray:
    """
    Extract enhanced operator features including DOP information
    
    Args:
        plan_node: PlanNode instance
        pipeline_dop: DOP level for the entire pipeline
    
    Returns:
        Enhanced feature vector including DOP features
    """
    # 生成GNN特征向量（包括序列化的算子类型）
    if plan_node.GNN_feature is None:
        plan_node.generate_gnn_feature('exec', operator_encoding, jointype_encoding)
    
    # 获取基础特征
    base_features = plan_node.GNN_feature
    
    # Add DOP-specific features
    dop_features = np.array([
        pipeline_dop,                              # Pipeline DOP level
        np.log2(pipeline_dop + 1),                 # Log-scaled DOP
        np.sqrt(pipeline_dop),                     # Square root scaled DOP
        1 if pipeline_dop > 1 else 0,             # Is parallel flag
        plan_node.actual_rows / max(pipeline_dop, 1),  # Rows per DOP
        plan_node.width / max(pipeline_dop, 1),        # Width per DOP
        plan_node.thread_id,                       # Thread ID within pipeline
        len(plan_node.child_plans),                # Number of children
        1 if plan_node.is_parallel else 0,        # Is parallel operator
        1 if plan_node.materialized else 0,       # Is materialized operator
    ])
    
    # Combine base and DOP features
    enhanced_features = np.concatenate([base_features, dop_features])
    
    return enhanced_features


def build_pipeline_from_plan_nodes(nodes: Dict[int, PlanNode], root_nodes: List[PlanNode], 
                                 target_dop_levels: Optional[List[int]] = None) -> List[Dict]:
    """
    Build pipelines from plan nodes using streaming operators as boundaries
    
    Args:
        nodes: Dictionary mapping plan_id to PlanNode
        root_nodes: List of root nodes
        target_dop_levels: List of target DOP levels for training
    
    Returns:
        List of pipeline dictionaries, each containing nodes and metadata
    """
    pipelines = []
    
    # Assign thread IDs based on streaming operators
    current_offset = 0
    root_thread_ids = set()  # Track which thread_ids contain root nodes
    
    for root in root_nodes:
        root_tid = current_offset
        assign_thread_ids_by_plan_id(root, thread_id=current_offset)
        # Mark the root node's thread_id
        root_thread_ids.add(root.thread_id)
        nodes_in_tree = list(nodes.values())  # Get all nodes in this tree
        max_tid = max(getattr(node, 'thread_id', 0) for node in nodes_in_tree)
        current_offset = max_tid + 1
    
    # Group nodes by thread ID to form pipelines
    thread_groups = collect_nodes_by_thread_id(list(nodes.values()))
    
    for thread_id, pipeline_nodes in thread_groups.items():
        if not pipeline_nodes:
            continue
            
        # Sort nodes by plan_id to maintain order
        pipeline_nodes.sort(key=lambda x: x.plan_id)
        
        # Get pipeline DOP from the first node (or use the maximum DOP in pipeline)
        pipeline_dop = pipeline_nodes[0].dop
        
        # If pipeline has non-parallel operators, force DOP to 1
        # This is for the model's DOP feature - the pipeline actually runs serially
        # Note: query_dop (stored in metadata) still keeps the original query DOP
        if has_non_parallel_operators(pipeline_nodes):
            pipeline_dop = 1
        
        # Calculate pipeline execution time and memory usage (sum of all nodes in pipeline)
        pipeline_execution_time = sum(node.execution_time for node in pipeline_nodes)
        pipeline_memory_usage = sum(node.peak_mem for node in pipeline_nodes)
        
        # Check if this pipeline contains root node(s)
        is_root_pipeline = thread_id in root_thread_ids
        
        pipeline = {
            'thread_id': thread_id,
            'nodes': pipeline_nodes,
            'dop': pipeline_dop,
            'execution_time': pipeline_execution_time,
            'memory_usage': pipeline_memory_usage,
            'num_nodes': len(pipeline_nodes),
            'has_streaming': any(is_streaming_operator(node.operator_type) for node in pipeline_nodes),
            'has_non_parallel': has_non_parallel_operators(pipeline_nodes),
            'is_root_pipeline': is_root_pipeline  # Mark if this pipeline contains root node
        }
        
        pipelines.append(pipeline)
    
    return pipelines


def load_mci_pipeline_data_exec(csv_file: str, query_info_file: str, 
                               target_dop_levels: Optional[List[int]] = None,
                               use_estimates: bool = False) -> List[Data]:
    """
    Load pipeline data for MCI execution time model training using streaming operator division
    
    Args:
        csv_file: Path to operator features CSV file
        query_info_file: Path to query execution info CSV file
        target_dop_levels: List of DOP levels to generate data for. If None, use all available DOPs
        use_estimates: Whether to use estimated values instead of actual values
    
    Returns:
        List of PyG Data objects for pipeline execution time training
        Each pipeline is a prediction unit with execution time as sum of all operators
    """
    return _load_mci_pipeline_data(csv_file, query_info_file, target_dop_levels, use_estimates, target_type='execution_time')


def load_mci_pipeline_data_mem(csv_file: str, query_info_file: str, 
                              target_dop_levels: Optional[List[int]] = None,
                              use_estimates: bool = False) -> List[Data]:
    """
    Load pipeline data for MCI memory model training using streaming operator division
    
    Args:
        csv_file: Path to operator features CSV file
        query_info_file: Path to query execution info CSV file
        target_dop_levels: List of DOP levels to generate data for. If None, use all available DOPs
        use_estimates: Whether to use estimated values instead of actual values
    
    Returns:
        List of PyG Data objects for pipeline memory training
        Each pipeline is a prediction unit with memory usage as sum of all operators
    """
    return _load_mci_pipeline_data(csv_file, query_info_file, target_dop_levels, use_estimates, target_type='memory_usage')


def _load_mci_pipeline_data(csv_file: str, query_info_file: str, 
                           target_dop_levels: Optional[List[int]] = None,
                           use_estimates: bool = False, 
                           target_type: str = 'execution_time') -> List[Data]:
    # Read operator features
    from utils import load_csv_safe
    df = load_csv_safe(csv_file, description="算子特征")
    if df is None:
        return []
    
    # Filter operators
    df = df[df['operator_type'].isin(operator_encoding)]
    
    # Add predicate cost
    df['predicate_cost'] = df['filter'].apply(
        lambda x: extract_predicate_cost(x) if pd.notnull(x) and x != '' else 0
    )
    
    # Encode categorical features - 暂时注释掉，在特征提取时再编码
    # df['operator_type'] = df['operator_type'].map(operator_encoding).astype(int)
    # df['jointype'] = df['jointype'].map(jointype_encoding).astype(int)
    
    # Read query execution info
    from utils import load_csv_safe
    query_info_df = load_csv_safe(query_info_file, description="查询信息")
    if query_info_df is None:
        return []
    
    # Use estimates if requested
    if use_estimates:
        print("MCI模拟模式：正在使用 'estimate_rows' 作为特征...")
        df['actual_rows'] = df['estimate_rows']
    
    # Group by query_id and dop to build individual pipeline graphs
    grouped = df.groupby(['query_id', 'query_dop'])
    
    data_list = []
    
    for (query_id, query_dop), group in grouped:
        # Filter target DOP levels if specified
        if target_dop_levels is not None and query_dop not in target_dop_levels:
            continue
            
        # Build pipeline plan tree using utils build_query_plan
        nodes, root_nodes = build_query_plan(group, use_estimates=use_estimates)
        
        if not nodes:
            continue
        
        # Build pipelines from the plan using streaming operator division
        pipelines = build_pipeline_from_plan_nodes(nodes, root_nodes, target_dop_levels)
        
        # Process each pipeline
        for pipeline in pipelines:
            pipeline_nodes = pipeline['nodes']
            pipeline_dop = pipeline['dop']
            
            # Select target value based on target_type
            if target_type == 'execution_time':
                pipeline_target_value = pipeline['execution_time']
            elif target_type == 'memory_usage':
                pipeline_target_value = pipeline['memory_usage']
            else:
                raise ValueError(f"Unknown target_type: {target_type}")
            
            # Extract enhanced node features
            node_features_list = []
            plan_ids = []
            
            for node in pipeline_nodes:
                enhanced_features = extract_enhanced_operator_features(node, pipeline_dop)
                node_features_list.append(enhanced_features)
                plan_ids.append(node.plan_id)
            
            if not node_features_list:
                continue
                
            # Create node feature tensor
            node_features = torch.tensor(np.array(node_features_list), dtype=torch.float32)
            
            # Build edge index for this pipeline (only internal edges)
            edge_list = []
            plan_id_to_index = {pid: idx for idx, pid in enumerate(plan_ids)}
            
            for node in pipeline_nodes:
                idx = plan_id_to_index[node.plan_id]
                for child in node.child_plans:
                    if child in pipeline_nodes:  # Only internal edges within pipeline
                        child_idx = plan_id_to_index.get(child.plan_id)
                        if child_idx is not None:
                            edge_list.append([child_idx, idx])  # Child -> Parent
            
            # Create edge index tensor
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Create PyG Data object for this pipeline
            data = Data(
                x=node_features,
                edge_index=edge_index,
                y=torch.tensor([[pipeline_target_value]], dtype=torch.float32),  # Pipeline target value (execution time or memory usage)
                dop_level=torch.tensor([pipeline_dop], dtype=torch.long),
                thread_id=torch.tensor([pipeline['thread_id']], dtype=torch.long),
                num_nodes=node_features.size(0),
                pipeline_metadata={
                    'query_id': query_id,  # Add query_id for query-level inference
                    'query_dop': query_dop,  # Add query's original DOP (not pipeline DOP)
                    'has_streaming': pipeline['has_streaming'],
                    'has_non_parallel': pipeline['has_non_parallel'],
                    'is_root_pipeline': pipeline.get('is_root_pipeline', False),  # Mark root pipeline
                    'num_operators': pipeline['num_nodes'],
                    'pipeline_id': f"pipeline_{pipeline['thread_id']}"  # Use thread_id as pipeline identifier
                }
            )
            
            data_list.append(data)
    
    return data_list


def create_multi_dop_training_data(csv_file: str, query_info_file: str,
                                 dop_levels: List[int] = [8, 16, 32, 64, 96],
                                 use_estimates: bool = False) -> List[Data]:
    """
    Create training data for multiple DOP levels using pipeline division
    Each pipeline is a prediction unit with execution time as sum of all operators
    DOP is used as input feature (ch2) rather than separate models
    
    Args:
        csv_file: Path to operator features CSV file
        query_info_file: Path to query execution info CSV file
        dop_levels: List of DOP levels to generate training data for
        use_estimates: Whether to use estimated values
    
    Returns:
        List of PyG Data objects for multi-DOP training
        Each data object represents a pipeline with its execution time
    """
    all_data = []
    
    for dop_level in dop_levels:
        print(f"Loading pipeline data for DOP level {dop_level}...")
        dop_data = _load_mci_pipeline_data(
            csv_file=csv_file,
            query_info_file=query_info_file,
            target_dop_levels=[dop_level],
            use_estimates=use_estimates,
            target_type='execution_time'  # Default to execution time for backward compatibility
        )
        
        # Note: dop_level is already set in build_pipeline_from_plan_nodes based on actual pipeline DOP
        # No need to override it with the target dop_level
        
        all_data.extend(dop_data)
        print(f"Loaded {len(dop_data)} pipeline samples for DOP level {dop_level}")
    
    return all_data


def create_mci_data_loaders_exec(csv_file: str, query_info_file: str, 
                                 target_dop_levels: Optional[List[int]] = None,
                                 batch_size: int = 32, 
                                 num_workers: int = 4,
                                 use_estimates: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for MCI execution time model training
    
    Args:
        csv_file: Path to operator features CSV file
        query_info_file: Path to query execution info CSV file
        target_dop_levels: List of DOP levels to generate data for
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        use_estimates: Whether to use estimated values instead of actual values
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    return create_mci_data_loaders_from_csv(csv_file, query_info_file, target_dop_levels, 
                                           batch_size, num_workers, use_estimates, 'execution_time')


def create_mci_data_loaders_mem(csv_file: str, query_info_file: str, 
                                target_dop_levels: Optional[List[int]] = None,
                                batch_size: int = 32, 
                                num_workers: int = 4,
                                use_estimates: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for MCI memory model training
    
    Args:
        csv_file: Path to operator features CSV file
        query_info_file: Path to query execution info CSV file
        target_dop_levels: List of DOP levels to generate data for
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        use_estimates: Whether to use estimated values instead of actual values
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    return create_mci_data_loaders_from_csv(csv_file, query_info_file, target_dop_levels, 
                                           batch_size, num_workers, use_estimates, 'memory_usage')


def create_mci_data_loaders_from_csv(csv_file: str, query_info_file: str, 
                                    target_dop_levels: Optional[List[int]] = None,
                                    batch_size: int = 32, 
                                    num_workers: int = 4,
                                    use_estimates: bool = False,
                                    target_type: str = 'execution_time') -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders from CSV files for MCI model training
    
    Args:
        csv_file: Path to operator features CSV file
        query_info_file: Path to query execution info CSV file
        target_dop_levels: List of DOP levels to generate data for
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        use_estimates: Whether to use estimated values instead of actual values
        target_type: Type of target data ('execution_time' or 'memory_usage')
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load data based on target type
    if target_type == 'execution_time':
        all_data = load_mci_pipeline_data_exec(csv_file, query_info_file, target_dop_levels, use_estimates)
    elif target_type == 'memory_usage':
        all_data = load_mci_pipeline_data_mem(csv_file, query_info_file, target_dop_levels, use_estimates)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    # Split data into train/validation (simple split for now)
    train_size = int(0.8 * len(all_data))
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]
    
    return create_mci_data_loaders(train_data, val_data, batch_size, num_workers)


def create_mci_data_loaders(train_data: List[Data], val_data: List[Data], 
                          batch_size: int = 16, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for MCI model training
    
    Args:
        train_data: List of training Data objects
        val_data: List of validation Data objects
        batch_size: Batch size for training
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def analyze_data_statistics(data_list: List[Data], target_type: str = 'execution_time') -> dict:
    """
    Analyze statistics of the loaded pipeline data
    
    Args:
        data_list: List of Data objects
        target_type: Type of target data ('execution_time' or 'memory_usage')
    
    Returns:
        Dictionary containing data statistics
    """
    stats = {
        'total_samples': len(data_list),
        'dop_levels': set(),
        'num_nodes_distribution': [],
        'num_edges_distribution': [],
        'feature_dim': 0,
        f'{target_type}_stats': [],
        'thread_ids': set(),
        'has_streaming_count': 0,
        'has_non_parallel_count': 0,
        'num_operators_distribution': []
    }
    
    for data in data_list:
        # DOP levels
        if hasattr(data, 'dop_level'):
            stats['dop_levels'].add(data.dop_level.item())
        
        # Thread IDs
        if hasattr(data, 'thread_id'):
            stats['thread_ids'].add(data.thread_id.item())
        
        # Node and edge counts
        stats['num_nodes_distribution'].append(data.num_nodes)
        stats['num_edges_distribution'].append(data.edge_index.size(1))
        
        # Feature dimension
        if stats['feature_dim'] == 0:
            stats['feature_dim'] = data.x.size(1)
        
        # Target value (execution time or memory usage)
        stats[f'{target_type}_stats'].append(data.y.item())
        
        # Pipeline metadata
        if hasattr(data, 'pipeline_metadata'):
            metadata = data.pipeline_metadata
            if metadata.get('has_streaming', False):
                stats['has_streaming_count'] += 1
            if metadata.get('has_non_parallel', False):
                stats['has_non_parallel_count'] += 1
            stats['num_operators_distribution'].append(metadata.get('num_operators', 0))
    
    # Convert sets to sorted lists
    stats['dop_levels'] = sorted(list(stats['dop_levels']))
    stats['thread_ids'] = sorted(list(stats['thread_ids']))
    
    # Calculate statistics
    stats['num_nodes_mean'] = np.mean(stats['num_nodes_distribution'])
    stats['num_nodes_std'] = np.std(stats['num_nodes_distribution'])
    stats['num_edges_mean'] = np.mean(stats['num_edges_distribution'])
    stats['num_edges_std'] = np.std(stats['num_edges_distribution'])
    stats['num_operators_mean'] = np.mean(stats['num_operators_distribution'])
    stats['num_operators_std'] = np.std(stats['num_operators_distribution'])
    stats[f'{target_type}_mean'] = np.mean(stats[f'{target_type}_stats'])
    stats[f'{target_type}_std'] = np.std(stats[f'{target_type}_stats'])
    stats['streaming_pipeline_ratio'] = stats['has_streaming_count'] / len(data_list) if data_list else 0
    stats['non_parallel_pipeline_ratio'] = stats['has_non_parallel_count'] / len(data_list) if data_list else 0
    
    return stats


if __name__ == "__main__":
    # Test the data loader
    print("Testing MCI Data Loader...")
    
    # Example usage (replace with actual file paths)
    # csv_file = "path/to/operator_features.csv"
    # query_info_file = "path/to/query_info.csv"
    
    # Load data for specific DOP levels
    # dop_levels = [1, 2, 4, 8]
    # data = create_multi_dop_training_data(csv_file, query_info_file, dop_levels)
    
    # Analyze statistics
    # stats = analyze_data_statistics(data)
    # print(f"Data Statistics: {stats}")
    
    print("✅ MCI Data Loader test completed!")
