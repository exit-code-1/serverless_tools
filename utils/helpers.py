import random
import pandas as pd
import numpy as np # <-- 确保导入
from collections import deque # <-- 确保导入

def split_queries(total_queries, train_ratio=0.8):
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

# ==============================================================================
# 新增函数：用于在 DataFrame 中传播基数估计
# ==============================================================================
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