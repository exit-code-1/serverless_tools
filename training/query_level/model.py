import os
import sys
import numpy as np
# sys.path.append(os.path.abspath("/home/zhy/opengauss/tools/serverless_tools/train/python"))

# 定义所有可能的算子类型及其特征，并添加 weight
operator_features = {
    'Row Adapter': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Limit': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'CStore Index Scan': ['l_input_rows', 'actual_rows', 'width', 'predicate_cost', 'weight', 'is_parallel'],
    'CTE Scan': ['l_input_rows', 'actual_rows', 'width', 'predicate_cost', 'weight', 'is_parallel'],
    'Vector Nest Loop': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'query_dop', 'weight', 'is_parallel'],
    'Vector Merge Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'weight', 'is_parallel'],
    'Aggregate': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'weight', 'is_parallel'],
    'Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'weight', 'is_parallel'], # <-- 已删除 'hash_table_size'
    # 'Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size', 'weight', 'is_parallel'],
    'Hash': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector WindowAgg': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Appendr': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Index Only Scan': ['l_input_rows', 'actual_rows', 'width', 'weight', 'predicate_cost', 'is_parallel'],
    # 'Vector Hash Aggregate': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio', 'weight', 'is_parallel'],
    'Vector Hash Aggregate': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio', 'weight', 'is_parallel'], # <-- 已删除 'hash_table_size'
    'Vector Aggregate': ['l_input_rows', 'actual_rows', 'width', 'agg_width', 'weight', 'is_parallel'],
    'Vector Sort': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Materialize': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    # 'Vector Sonic Hash Aggregate': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio', 'weight', 'is_parallel'],
    # 'Vector Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size', 'weight', 'is_parallel'],
    # 'Vector Sonic Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', "jointype",'predicate_cost','hash_table_size','weight', 'is_parallel'],
    'Vector Sonic Hash Aggregate': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width',  'disk_ratio', 'weight', 'is_parallel'], # <-- 已删除 'hash_table_size'
    'Vector Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'weight', 'is_parallel'], # <-- 已删除 'hash_table_size'
    'Vector Sonic Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', "jointype",'predicate_cost','weight', 'is_parallel'], # <-- 已删除 'hash_table_size'
    'Vector Streaming LOCAL GATHER': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Streaming BROADCAST': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Streaming REDISTRIBUTE': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector SetOp': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Append': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
}

ALL_OPERATORS = sorted(operator_features.keys())  # 确保固定顺序

# 计算每个算子的最大特征维度
feature_dim_per_operator = {op: len(features) for op, features in operator_features.items()}
total_feature_dim = sum(feature_dim_per_operator.values())  # 最终特征向量长度
all_features = sorted(set(f for feats in operator_features.values() for f in feats))
feature_index = {feat: i for i, feat in enumerate(all_features)}

class PlanNode:
    def __init__(self, plan_id, operator, features, use_estimates=False): # <-- 1. 增加 use_estimates 开关
        self.plan_id = plan_id
        self.operator = operator
        self.features = features
        self.children = []
        self.weight = None
        self.parent = None  # 添加 parent 属性
        self.depth = None  # 初始化 depth 属性
        self.use_estimates = use_estimates # 存储模式
    
        # 新增：存储估计值，以便传播
        self.estimate_rows = self.features.get('estimate_rows', self.features.get('actual_rows', 0))
    
    def update_estimated_inputs(self):
        """根据子节点的估计输出，更新本节点的特征字典。"""
        if not self.use_estimates or not self.children:
            return

        # 这个模型比较特殊，特征是直接从原始row来的，所以我们直接修改features字典
        if len(self.children) > 0:
            self.features['l_input_rows'] = self.children[0].estimate_rows
        
        if len(self.children) > 1:
            self.features['r_input_rows'] = self.children[1].estimate_rows
    def compute_depth(self):
        """计算当前节点的深度（叶子节点的深度最高，根节点的深度最低）"""
        if not self.children:
            self.depth = 1  # 叶子节点深度是1
        else:
            self.depth = 1 + max(child.compute_depth() for child in self.children)
        return self.depth  # 确保 depth 被存储

    def compute_weight(self, max_depth):
        """根据最大深度计算 weigh,weight = actual_rows * width * depth"""
        if self.depth is None:
            raise ValueError("compute_depth() must be called before compute_weight()")

        # --- 2. 根据模式选择行数 ---
        # 如果是模拟模式，使用估计行数计算权重，否则使用实际行数
        rows_for_weight = self.estimate_rows if self.use_estimates else self.features.get('actual_rows', 1)
        
        width = self.features.get('width', 1)
        depth = max_depth - self.depth + 1

        self.weight = rows_for_weight * width * depth
        self.features['weight'] = self.weight
        return self.weight
    def extract_feature_vector(self):
        """根据 `operator` 提取对应的特征向量，并填充到固定位置"""
        feature_dict = {op: [0] * len(operator_features[op]) for op in ALL_OPERATORS}
        
        if self.operator in operator_features:
            feature_names = operator_features[self.operator]
            
            # --- 3. 根据模式选择特征值 ---
            # 创建一个临时特征字典用于提取
            features_to_extract = self.features.copy()
            if self.use_estimates:
                features_to_extract['actual_rows'] = self.estimate_rows
                # l_input_rows 和 r_input_rows 已在 update_estimated_inputs 中被修改
            
            feature_dict[self.operator] = [features_to_extract.get(feat, 0) for feat in feature_names]

        return np.concatenate(list(feature_dict.values()))