import os
import sys
import numpy as np
sys.path.append(os.path.abspath("/home/zhy/opengauss/tools/serverless_tools/train/python"))
# 定义所有可能的算子类型及其特征，并添加 weight
operator_features = {
    'Row Adapter': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Limit': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'CStore Index Scan': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'predicate_cost', 'weight', 'is_parallel'],
    'CTE Scan': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'predicate_cost', 'weight', 'is_parallel'],
    'Vector Nest Loop': ['estimate_costs', 'l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'query_dop', 'weight', 'is_parallel'],
    'Vector Merge Join': ['estimate_costs', 'l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'weight', 'is_parallel'],
    'Aggregate': ['estimate_costs', 'l_input_rows', 'width', 'agg_col', 'agg_width', 'weight', 'is_parallel'],
    'Hash Join': ['estimate_costs', 'l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size', 'weight', 'is_parallel'],
    'Hash': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector WindowAgg': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Appendr': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Index Only Scan': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'predicate_cost', 'is_parallel'],
    'Vector Hash Aggregate': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio', 'weight', 'is_parallel'],
    'Vector Aggregate': ['l_input_rows', 'actual_rows', 'width', 'agg_width', 'weight', 'is_parallel'],
    'Vector Sort': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Materialize': ['l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Sonic Hash Aggregate': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio', 'weight', 'is_parallel'],
    'Vector Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size', 'weight', 'is_parallel'],
    'Vector Sonic Hash Join': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', "jointype",'predicate_cost','hash_table_size','weight', 'is_parallel'],
    'Vector Streaming LOCAL GATHER': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Streaming BROADCAST': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Streaming REDISTRIBUTE': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector SetOp': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
    'Vector Append': ['estimate_costs', 'l_input_rows', 'actual_rows', 'width', 'weight', 'is_parallel'],
}

ALL_OPERATORS = sorted(operator_features.keys())  # 确保固定顺序

# 计算每个算子的最大特征维度
feature_dim_per_operator = {op: len(features) for op, features in operator_features.items()}
total_feature_dim = sum(feature_dim_per_operator.values())  # 最终特征向量长度
all_features = sorted(set(f for feats in operator_features.values() for f in feats))
feature_index = {feat: i for i, feat in enumerate(all_features)}

class PlanNode:
    def __init__(self, plan_id, operator, features):
        self.plan_id = plan_id
        self.operator = operator
        self.features = features
        self.children = []
        self.weight = None
        self.parent = None  # 添加 parent 属性
        self.depth = None  # 初始化 depth 属性

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

        actual_rows = self.features.get('actual_rows', 1)
        width = self.features.get('width', 1)
        depth = max_depth - self.depth + 1  # 让叶子深度最高，根深度最低

        self.weight = actual_rows * width * depth
        self.features['weight'] = self.weight  # 存入 features
        return self.weight
    def extract_feature_vector(self):
        """根据 `operator` 提取对应的特征向量，并填充到固定位置"""
        feature_dict = {op: [0] * len(operator_features[op]) for op in ALL_OPERATORS}  # 先全填 0
        
        if self.operator in operator_features:
            feature_names = operator_features[self.operator]
            feature_dict[self.operator] = [self.features.get(feat, 0) for feat in feature_names]

        return np.concatenate(list(feature_dict.values()))