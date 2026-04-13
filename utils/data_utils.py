# -*- coding: utf-8 -*-
"""
数据处理工具模块
包含数据加载、保存、查询分割等功能
"""

import os
import random
import pandas as pd
import numpy as np
from collections import deque
from typing import Optional

def load_csv_safe(file_path: str, delimiter: str = ';', description: str = "CSV文件") -> Optional[pd.DataFrame]:
    """安全加载CSV文件"""
    try:
        if not os.path.exists(file_path):
            print(f"错误: {description}未找到: {file_path}")
            return None
        
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
        print(f"成功加载{description}: {file_path} ({len(df)} 行)")
        return df
    except Exception as e:
        print(f"加载{description}时出错: {e}")
        return None

def save_csv_safe(df: pd.DataFrame, file_path: str, delimiter: str = ';', description: str = "CSV文件") -> bool:
    """安全保存CSV文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, sep=delimiter, index=False)
        print(f"成功保存{description}: {file_path}")
        return True
    except Exception as e:
        print(f"保存{description}时出错: {e}")
        return False

def check_file_exists(file_path: str, description: str = "文件") -> bool:
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        print(f"错误: {description}未找到: {file_path}")
        return False
    return True

def split_queries(total_queries, train_ratio=0.8):
    """分割查询为训练集和测试集"""
    # 生成1到total_queries的列表
    queries = list(range(1, total_queries + 1))
    
    # 随机打乱查询顺序
    random.shuffle(queries)
    
    # 计算训练集和测试集的分割点
    split_point = int(len(queries) * train_ratio)
    
    # 分割查询
    train_queries = queries[:split_point]
    test_queries = queries[split_point:]
    
    return train_queries, test_queries

def save_query_split(train_queries, test_queries, file_path):
    """保存查询分割结果"""
    # 创建一个DataFrame来存储训练和测试查询
    query_split_df = pd.DataFrame({
        'query_id': train_queries + test_queries,
        'split': ['train'] * len(train_queries) + ['test'] * len(test_queries)
    })
    
    # 按照query_id排序
    query_split_df = query_split_df.sort_values(by='query_id').reset_index(drop=True)
    
    # 保存到CSV文件
    query_split_df.to_csv(file_path, index=False)
    print(f"Query split has been saved to {file_path}")

def propagate_estimates_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    对包含所有查询计划的DataFrame进行预处理，以模拟基数估计的自底向上传播。
    此函数会直接修改l_input_rows和r_input_rows列。

    Args:
        df (pd.DataFrame): 原始的 plan_info DataFrame。

    Returns:
        pd.DataFrame: 修改了输入行数的 DataFrame。
    """
    print("开始在DataFrame中预处理以传播基数估计...")
    
    # 创建要修改的列的副本，以避免直接在原始数据上操作
    df['l_input_rows_est'] = df['l_input_rows']
    df['r_input_rows_est'] = df['r_input_rows']
    
    # 按每个查询计划进行分组处理
    groups = df.groupby(['query_id', 'query_dop'])
    processed_groups = []

    for key, group in groups:
        group = group.copy() # 操作副本
        
        # 建立plan_id到行索引的映射
        id_to_idx = {row['plan_id']: index for index, row in group.iterrows()}
        # 建立父子关系图
        parent_map = {} # {child_id: parent_id}
        children_map = {row['plan_id']: [] for _, row in group.iterrows()}
        
        for idx, row in group.iterrows():
            child_ids_str = row['child_plan']
            if pd.notna(child_ids_str) and str(child_ids_str).strip():
                try:
                    child_ids = [int(cid) for cid in str(child_ids_str).split(',')]
                    children_map[row['plan_id']] = child_ids
                    for child_id in child_ids:
                        if child_id in id_to_idx:
                            parent_map[child_id] = row['plan_id']
                except ValueError:
                    continue

        # 找到叶子节点 (没有子节点的节点)
        leaf_nodes = [pid for pid, children in children_map.items() if not children]
        
        # 使用队列进行自底向上的拓扑处理 (Kahn's algorithm a-like)
        queue = deque(leaf_nodes)
        processed_nodes = set()
        
        while queue:
            node_id = queue.popleft()
            if node_id in processed_nodes:
                continue
            
            processed_nodes.add(node_id)
            
            # 如果这个节点是某个节点的子节点，我们需要更新其父节点
            if node_id in parent_map:
                parent_id = parent_map[node_id]
                parent_idx = id_to_idx.get(parent_id)
                if parent_idx is None: continue
                
                # 获取父节点的子节点列表
                parent_children = children_map.get(parent_id, [])
                
                # 获取子节点的估计输出
                child_idx = id_to_idx.get(node_id)
                child_estimate_rows = group.loc[child_idx, 'estimate_rows']

                # 根据是左子节点还是右子节点来更新父节点的输入
                if len(parent_children) > 0 and parent_children[0] == node_id:
                    # 这是左子节点
                    group.loc[parent_idx, 'l_input_rows_est'] = child_estimate_rows
                elif len(parent_children) > 1 and parent_children[1] == node_id:
                    # 这是右子节点
                    group.loc[parent_idx, 'r_input_rows_est'] = child_estimate_rows

                # 检查父节点的所有子节点是否都已处理完毕
                all_children_processed = all(cid in processed_nodes for cid in parent_children)
                if all_children_processed and parent_id not in processed_nodes:
                    queue.append(parent_id)
        
        processed_groups.append(group)

    if not processed_groups:
        print("警告: 未处理任何查询组，返回原始DataFrame。")
        return df
        
    # 合并所有处理过的组
    result_df = pd.concat(processed_groups)
    
    # 用更新后的估计值覆盖原始的输入行数
    result_df['l_input_rows'] = result_df['l_input_rows_est']
    result_df['r_input_rows'] = result_df['r_input_rows_est']
    # 删除临时列
    result_df.drop(columns=['l_input_rows_est', 'r_input_rows_est'], inplace=True)
    
    print("DataFrame基数传播预处理完成。")
    return result_df


def build_query_plan(group_df, use_estimates=False, onnx_manager=None):
    """
    根据group_df构建查询计划树，使用core PlanNode类
    对于 Hash Join 和 Aggregate 算子，拆分成 Build 和 Probe 两个节点
    
    Args:
        group_df: DataFrame containing plan data grouped by query_id and dop
        use_estimates: Whether to use estimated values instead of actual values
        onnx_manager: ONNXModelManager instance, if None will not use ONNX inference
    
    Returns:
        tuple: (nodes_dict, root_nodes_list)
            - nodes_dict: Dictionary mapping plan_id to PlanNode instances
            - root_nodes_list: List of root PlanNode instances
    """
    from core.plan_node import PlanNode
    
    nodes = {}
    root_nodes = []
    nodes_to_split = {}  # Store original nodes that need to be split
    
    # Helper function to check if operator should be split
    def should_split_operator(operator_type):
        """Check if operator should be split into build and probe"""
        op_lower = operator_type.lower()
        return ('hash' in op_lower and 'join' in op_lower) or \
               ('aggregate' in op_lower and ('hash' in op_lower or 'sonic' in op_lower))
    
    # First pass: create all PlanNode instances and identify nodes to split
    for _, row in group_df.iterrows():
        plan_data = row.to_dict()
        operator_type = plan_data.get('operator_type', '')
        
        if should_split_operator(operator_type):
            # Store original node data for splitting later
            nodes_to_split[row['plan_id']] = (row, plan_data)
        else:
            # Create normal node
            node = PlanNode(plan_data, onnx_manager, use_estimates=use_estimates)
            nodes[row['plan_id']] = node
    
    # Second pass: split join/agg nodes into build and probe
    for original_plan_id, (row, plan_data) in nodes_to_split.items():
        operator_type = plan_data.get('operator_type', '')
        build_time = plan_data.get('build_time', 0.0)
        hash_time = plan_data.get('hash_time', 0.0)
        
        # If hash_time is 0, calculate it from execution_time - build_time
        if hash_time == 0.0:
            execution_time = plan_data.get('execution_time', 0.0)
            hash_time = max(0.0, execution_time - build_time)
        
        # Create Build node
        build_plan_data = plan_data.copy()
        build_plan_data['plan_id'] = original_plan_id * 10000 + 1  # Use a large offset to avoid conflicts
        build_plan_data['operator_type'] = operator_type + ' Build'
        build_plan_data['execution_time'] = build_time
        build_plan_data['build_time'] = build_time
        build_plan_data['hash_time'] = 0.0
        
        build_node = PlanNode(build_plan_data, onnx_manager, use_estimates=use_estimates)
        # Override materialized to True for build node
        build_node.materialized = True
        build_node.execution_time = build_time
        build_node.pred_execution_time = build_time
        nodes[build_node.plan_id] = build_node
        
        # Create Probe node
        # Probe execution time = total execution time - build time
        # hash_time includes subtree time, so we should use execution_time - build_time instead
        execution_time = plan_data.get('execution_time', 0.0)
        probe_execution_time = max(0.0, execution_time - build_time)
        
        probe_plan_data = plan_data.copy()
        probe_plan_data['plan_id'] = original_plan_id * 10000 + 2  # Use a large offset to avoid conflicts
        probe_plan_data['operator_type'] = operator_type + ' Probe'
        probe_plan_data['execution_time'] = probe_execution_time
        probe_plan_data['build_time'] = 0.0
        probe_plan_data['hash_time'] = hash_time  # Keep hash_time for reference, but don't use it for execution_time
        
        probe_node = PlanNode(probe_plan_data, onnx_manager, use_estimates=use_estimates)
        # Override materialized to False for probe node
        probe_node.materialized = False
        probe_node.execution_time = probe_execution_time
        probe_node.pred_execution_time = probe_execution_time
        nodes[probe_node.plan_id] = probe_node
        
        # Set parent-child relationship: Probe -> Build
        # Build is a child of Probe (Build will be connected to right subtree later)
        build_node.parent_node = probe_node
        probe_node.child_plans.append(build_node)
    
    # Third pass: associate parent-child relationships
    for _, row in group_df.iterrows():
        original_plan_id = row['plan_id']
        
        # If this node was split, handle relationships for build/probe nodes
        if original_plan_id in nodes_to_split:
            build_node_id = original_plan_id * 10000 + 1
            probe_node_id = original_plan_id * 10000 + 2
            
            if build_node_id not in nodes or probe_node_id not in nodes:
                continue
            
            build_node = nodes[build_node_id]
            probe_node = nodes[probe_node_id]
            
            # Handle children: 
            # - Right subtree (build side) -> Build node
            # - Left subtree (probe side) -> Probe node
            if pd.notna(row['child_plan']):
                child_ids = [int(pid) for pid in str(row['child_plan']).split(',')]
                if len(child_ids) == 2:
                    # Hash Join: left child (probe side) -> Probe, right child (build side) -> Build
                    left_child_id = child_ids[0]
                    right_child_id = child_ids[1]
                    
                    # Connect left subtree to Probe node
                    if left_child_id in nodes:
                        nodes[left_child_id].parent_node = probe_node
                        probe_node.child_plans.append(nodes[left_child_id])
                    
                    # Connect right subtree to Build node
                    if right_child_id in nodes:
                        nodes[right_child_id].parent_node = build_node
                        build_node.child_plans.append(nodes[right_child_id])
                elif len(child_ids) == 1:
                    # Aggregate: single child -> Build node
                    child_id = child_ids[0]
                    if child_id in nodes:
                        nodes[child_id].parent_node = build_node
                        build_node.child_plans.append(nodes[child_id])
                else:
                    # Multiple children or no children - handle normally
                    for cid in child_ids:
                        if cid in nodes:
                            if cid == original_plan_id:
                                print(f"Warning: Self-reference detected for plan_id {original_plan_id}")
                                continue
                            # Default: connect to build node
                            nodes[cid].parent_node = build_node
                            build_node.child_plans.append(nodes[cid])
            
            # Handle parent: original node's parent becomes probe node's parent
            # We need to find the parent by checking which nodes have original_plan_id as child
            for _, parent_row in group_df.iterrows():
                if pd.notna(parent_row['child_plan']):
                    child_ids = [int(pid) for pid in str(parent_row['child_plan']).split(',')]
                    if original_plan_id in child_ids:
                        parent_node_id = parent_row['plan_id']
                        if parent_node_id in nodes_to_split:
                            # Parent was also split, connect to its probe node
                            parent_probe_id = parent_node_id * 10000 + 2
                            if parent_probe_id in nodes:
                                probe_node.parent_node = nodes[parent_probe_id]
                                nodes[parent_probe_id].child_plans.append(probe_node)
                        elif parent_node_id in nodes:
                            probe_node.parent_node = nodes[parent_node_id]
                            nodes[parent_node_id].child_plans.append(probe_node)
                        break
        else:
            # Normal node: handle relationships normally
            node = nodes.get(original_plan_id)
            if node is None:
                continue
                
            if pd.notna(row['child_plan']):
                child_ids = [int(pid) for pid in str(row['child_plan']).split(',')]
                for cid in child_ids:
                    # Check if child was split
                    if cid in nodes_to_split:
                        # Child was split, connect to its probe node (not build node)
                        # Because probe is the parent of build, and we want probe to be the child of this node
                        child_probe_id = cid * 10000 + 2
                        if child_probe_id in nodes:
                            nodes[child_probe_id].parent_node = node
                            node.child_plans.append(nodes[child_probe_id])
                    elif cid in nodes:
                        if cid == original_plan_id:
                            print(f"Warning: Self-reference detected for plan_id {original_plan_id}")
                            continue
                        nodes[cid].parent_node = node
                        node.child_plans.append(nodes[cid])
    
    # Identify root nodes
    for node in nodes.values():
        if node.parent_node is None:
            root_nodes.append(node)
    
    return nodes, root_nodes


def update_tree_estimated_inputs_recursive(node):
    """
    递归更新树中所有节点的估计输入值
    
    Args:
        node: PlanNode instance to start the recursive update from
    """
    for child in node.child_plans:
        update_tree_estimated_inputs_recursive(child)
    node.update_estimated_inputs()
