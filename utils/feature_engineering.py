# 文件路径: utils/feature_engineering.py
import random
import re
import numpy as np
import pandas as pd
# import glob # 未在此文件中直接使用
# from sklearn.calibration import LabelEncoder # 未在此文件中直接使用
# from sklearn.model_selection import train_test_split # 未在此文件中直接使用

# --- 导入 config (确保是绝对导入) ---
from config.structure import (
    column_type_cost_dict, table_structure, jointype_encoding,
    dop_operator_features, no_dop_operator_features,
    dop_operators_exec, dop_operators_mem,
    no_dop_operators_exec, no_dop_operators_mem
)
# --- 结束导入 ---

# 根据表名和列类型创建特征词典 (保持不变)
def create_feature_dict(table_structure, column_type_cost_dict):
    feature_dict = []
    for table_name, columns in table_structure.items():
        for column_name, column_type in columns:
            column_cost = 0 # 初始化
            try: # 增加错误处理
                if 'CHAR' in column_type or 'VARCHAR' in column_type:
                    # 提取列的长度（数字部分）
                    length_match = re.search(r'\((\d+)\)', column_type)
                    if length_match:
                         length = int(length_match.group(1))
                         base_type = column_type.split('(')[0]
                         if base_type in column_type_cost_dict and callable(column_type_cost_dict[base_type]):
                             column_cost = column_type_cost_dict[base_type](length)
                elif 'DECIMAL' in column_type:
                    # 提取 DECIMAL 的精度和标度
                    decimal_match = re.search(r'\((\d+),(\d+)\)', column_type)
                    if decimal_match:
                        precision, scale = map(int, decimal_match.groups())
                        if 'DECIMAL' in column_type_cost_dict and callable(column_type_cost_dict['DECIMAL']):
                             column_cost = column_type_cost_dict['DECIMAL'](precision, scale)
                else:
                    # 对于其他类型，直接获取对应的开销
                    column_cost = column_type_cost_dict.get(column_type, 0)
            except Exception as e:
                 print(f"警告: 计算列 '{column_name}' ({column_type}) 开销时出错: {e}")
                 column_cost = 0 # 出错时设为0

            feature_dict.append({
                'table_name': table_name,
                'column_name': column_name,
                'column_type': column_type,
                'column_cost': column_cost
            })
    return feature_dict

def extract_predicate_cost(predicate):
    if not predicate or pd.isna(predicate):
        return 0
    total_cost = 0
    
    predicate = re.sub(r"NUMTODSINTERVAL\((\d+),\s*'([A-Z]+)'\)", lambda m: f"INTERVAL({m.group(1)})", predicate)
    predicate_parts = re.split(r'\s+AND\s+|\s+OR\s+', predicate, flags=re.IGNORECASE)

    for part in predicate_parts:
        # 找出列名
        column_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|<|>|<=|>=|LIKE|IN|ANY)?', part)
        if not column_match:
            continue
        column_name = column_match.group(1)

        # 推测字段类型
        if re.search(r"TO_DATE|INTERVAL|'?\d{4}-\d{2}-\d{2}'?", part):  # 日期
            column_type = 'DATE'
        elif 'LIKE' in part or re.search(r"'[^']*'", part):  # LIKE 或字符串常量
            column_type = 'VARCHAR(20)'
        elif re.search(r'\d+\.\d+', part):  # 小数
            column_type = 'DECIMAL(10,2)'
        elif re.search(r'\d+', part):  # 整数
            column_type = 'INT'
        else:
            column_type = 'VARCHAR(20)'

        # 解析基础类型和参数
        if 'DECIMAL' in column_type:
            precision, scale = map(int, column_type.split('(')[1].split(')')[0].split(','))
            column_type_base = 'DECIMAL'
        elif 'CHAR' in column_type or 'VARCHAR' in column_type:
            length = int(column_type.split('(')[1].split(')')[0])
            column_type_base = 'VARCHAR' if 'VARCHAR' in column_type else 'CHAR'
        else:
            column_type_base = column_type

        # 计算谓词成本
        operator_cost = 0
        if 'ANY' in part or 'IN' in part:
            num_elements = len(re.findall(r'\d+', part))
            if column_type_base == 'DECIMAL':
                operator_cost = num_elements * column_type_cost_dict[column_type_base](precision, scale)
            elif column_type_base in ['CHAR', 'VARCHAR']:
                operator_cost = num_elements * column_type_cost_dict[column_type_base](length)
            else:
                operator_cost = num_elements * column_type_cost_dict[column_type_base]
        elif 'LIKE' in part:
            if column_type_base in ['CHAR', 'VARCHAR']:
                operator_cost = 2 * column_type_cost_dict[column_type_base](length)
        elif '=' in part or '<' in part or '>' in part:
            if column_type_base == 'DECIMAL':
                operator_cost = column_type_cost_dict[column_type_base](precision, scale)
            elif column_type_base in ['CHAR', 'VARCHAR']:
                operator_cost = column_type_cost_dict[column_type_base](length)
            else:
                operator_cost = column_type_cost_dict[column_type_base]
        elif 'INTERVAL' in part:
            operator_cost = 4
        else:
            if column_type_base == 'DECIMAL':
                operator_cost = column_type_cost_dict[column_type_base](precision, scale)
            elif column_type_base in ['CHAR', 'VARCHAR']:
                operator_cost = column_type_cost_dict[column_type_base](length)
            else:
                operator_cost = column_type_cost_dict[column_type_base]

        total_cost += operator_cost

    return total_cost

# 提取索引名称 (保持不变)
def extract_index_name(index_name):
    if pd.isnull(index_name) or index_name == '':
        return None
    match = re.search(r'\(([a-zA-Z_][a-zA-Z0-9_]*)', str(index_name)) # 转 str 避免类型错误
    if match:
        return match.group(1)
    return None

# 计算索引成本 (保持不变, 增加健壮性)
def calculate_index_cost(index_name, table_structure, column_type_cost_dict):
    """
    根据 index_name 计算开销。
    """
    column_name = extract_index_name(index_name)
    if not column_name:
        return 0
    
    # 遍历表结构，找到列名对应的列类型
    for table_name, columns in table_structure.items():
        for col_name, col_type in columns:
            if col_name.lower() == column_name.lower():
                # 计算开销
                if 'CHAR' in col_type or 'VARCHAR' in col_type:
                    length = int(col_type.split('(')[1].split(')')[0])
                    return column_type_cost_dict[col_type.split('(')[0]](length)
                elif 'DECIMAL' in col_type:
                    precision, scale = map(int, col_type.split('(')[1].split(')')[0].split(','))
                    return column_type_cost_dict['DECIMAL'](precision, scale)
                else:
                    return column_type_cost_dict.get(col_type, 0)
    return 0

# 实例归一化 (保持不变)
def instance_normalize(data, epsilon=1e-5):
    if not isinstance(data, pd.DataFrame) or data.empty: # 增加检查
         return data # 或者返回空 DataFrame
    num_instances, num_channels = data.shape
    normalized_data = data.copy()
    for i in range(num_instances):
        instance = data.iloc[i].astype(float) # 确保是浮点数
        mean = np.nanmean(instance) # 使用 nanmean 忽略 NaN
        variance = np.nanvar(instance) # 使用 nanvar 忽略 NaN
        # 处理方差为0或非常小的情况
        if variance < epsilon:
            normalized_data.iloc[i] = 0.0 # 或者 (instance - mean)
        else:
             normalized_data.iloc[i] = abs(instance - mean) / np.sqrt(variance + epsilon)
    return normalized_data

def prepare_data(train_data, test_data, operator, feature_columns, target_columns, epsilon=1e-5, use_estimates=False): # <-- 1. 增加 use_estimates 开关
    # Filter by operator type
    if 'is_executed' in train_data.columns:
        operator_data = train_data[(train_data['operator_type'] == operator) & (train_data['is_executed'] == True) & (train_data['execution_time'] > 0)].copy()
    else:
        operator_data = train_data[(train_data['operator_type'] == operator) & (train_data['execution_time'] > 0)].copy()
    
    # --- 2. 新增：如果开启模拟模式，替换训练数据中的行数 ---
    if use_estimates:
        # 在模拟模式下，我们只替换输出行数。
        # 输入行数 (l_input_rows, r_input_rows) 已经在之前的
        # propagate_estimates_in_dataframe 函数中被正确地修改了。
        if 'estimate_rows' in operator_data.columns:
            print(f"训练模拟模式: 为算子 '{operator}' 使用 'estimate_rows' 作为 'actual_rows' 特征。")
            operator_data['actual_rows'] = operator_data['estimate_rows']
        else:
            print(f"警告: 开启了训练模拟模式，但找不到 'estimate_rows' 列。")
    # Filter out rows where query_dop == 1
    # operator_data = operator_data[operator_data['query_dop'] != 1]
    
    # 添加 index_cost 列
    operator_data['index_cost'] = operator_data['index_names'].apply(
        lambda x: calculate_index_cost(x, table_structure, column_type_cost_dict)
    )
    # 对 filter 列进行谓词开销计算
    operator_data['predicate_cost'] = operator_data['filter'].apply(
        lambda x: extract_predicate_cost(x) if pd.notnull(x) and x != '' else 0
    )


    # # Assign train/test split based on query_id
    # operator_data.loc[:, 'set'] = operator_data['query_id'].apply(
    #     lambda qid: 'train' if qid  in train_queries else 'test' if qid  in test_queries else 'exclude'
    # )

    # 使用提前编码的字典将 jointype 和 table_names 转换为标签
    operator_data['jointype'] = operator_data['jointype'].map(jointype_encoding)
    
    if 'is_executed' in test_data.columns:
        operator_test = test_data[(test_data['operator_type'] == operator) & (test_data['is_executed'] == True) & (test_data['execution_time'] > 0)].copy()
    else:
        operator_test = test_data[(test_data['operator_type'] == operator) & (test_data['execution_time'] > 0)].copy()
    operator_test['index_cost'] = operator_test['index_names'].apply(
        lambda x: calculate_index_cost(x, table_structure, column_type_cost_dict)
    )
    # 对 filter 列进行谓词开销计算
    operator_test['predicate_cost'] = operator_test['filter'].apply(
        lambda x: extract_predicate_cost(x) if pd.notnull(x) and x != '' else 0
    )
    
    operator_test['jointype'] = operator_test['jointype'].map(jointype_encoding)

    # Split into train and test data
    train_data = operator_data
    test_data = operator_test

    # 选取特征和目标
    # 移除分类类型的列，不参与归一化
    categorical_columns = ['jointype', 'table_names']  # 你可以根据需要调整
    numerical_columns = [col for col in feature_columns if col not in categorical_columns]

    # 对数值列进行归一化
    # X_train = instance_normalize(train_data[numerical_columns], epsilon)
    # X_test = instance_normalize(test_data[numerical_columns], epsilon)
    
    X_train = train_data[numerical_columns].copy()
    X_test = test_data[numerical_columns].copy()
    
    # 将分类列保留并添加到归一化后的数据中
    X_train[categorical_columns] = train_data[categorical_columns]
    X_test[categorical_columns] = test_data[categorical_columns]

    # 目标列
    y_train = train_data[target_columns]
    y_test = test_data[target_columns]

    # --- 新增的修正代码 ---
    # 遍历所有特征列，强制转换为数值类型，并填充 NaN
    for col in feature_columns:
        if col in X_train.columns:
            # errors='coerce' 会将无法转换的值变为 NaN
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            # 使用 0 或 -1 等一个合适的默认值填充 NaN
            X_train[col].fillna(0, inplace=True) 

        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            X_test[col].fillna(0, inplace=True)
    # --- 修正代码结束 ---

    return X_train, X_test, y_train, y_test

def prepare_inference_data(data, operator):
    """准备用于推理的特征数据 (修正字典索引方式)"""
    plan_id = data.get('plan_id', 'N/A')
    # print(f"--- DEBUG (prepare_inference_data): Operator = {operator}, Plan ID = {plan_id} ---")

    data_copy = data.copy() # data 是一个字典

    try:
        # --- 特征计算 (保持不变) ---
        jointype_val = data_copy.get('jointype')
        if isinstance(jointype_val, list): jointype_val = None # 简单处理列表
        data_copy['jointype'] = jointype_encoding.get(jointype_val, -1)

        index_names_val = data_copy.get('index_names')
        if isinstance(index_names_val, list): index_names_val = str(index_names_val)
        data_copy['index_cost'] = calculate_index_cost(index_names_val, table_structure, column_type_cost_dict)

        filter_val = data_copy.get('filter')
        if isinstance(filter_val, list): filter_val = " AND ".join(map(str, filter_val))
        data_copy['predicate_cost'] = extract_predicate_cost(filter_val)
        # --- 结束特征计算 ---

    except Exception as e:
         print(f"  ERROR during feature calculation in prepare_inference_data (Plan ID: {plan_id}): {e}")
         return None, None

    features_exec = None
    features_mem = None
    data_exec = None
    data_mem = None

    # 获取特征列表 (保持不变)
    op_features = None
    if operator in dop_operator_features:
        op_features = dop_operator_features.get(operator)
    elif operator in no_dop_operator_features:
        op_features = no_dop_operator_features.get(operator)
    if op_features:
        features_exec = op_features.get('exec')
        features_mem = op_features.get('mem')
    # else: (警告逻辑保持不变)


    # --- 修改特征数据提取方式 ---
    if features_exec:
        data_exec_list = []
        for feat in features_exec: # 遍历特征名列表
            value = data_copy.get(feat) # 用特征名作为键从字典获取值
            # (处理 list 和 NaN 值)
            if isinstance(value, list): value = None # 或其他处理
            if pd.isna(value): value = 0
            data_exec_list.append(value)
        try:
            # 将收集到的值列表转换为 Numpy Array
            data_exec = np.array([float(v) if v is not None else 0.0 for v in data_exec_list], dtype=np.float32)
        except (ValueError, TypeError) as e:
            print(f"ERROR converting exec features: {e}")
            data_exec = None

    if features_mem:
        data_mem_list = []
        for feat in features_mem: # 遍历特征名列表
            value = data_copy.get(feat) # 用特征名作为键从字典获取值
            # (处理 list 和 NaN 值)
            if isinstance(value, list): value = None
            if pd.isna(value): value = 0
            data_mem_list.append(value)
        try:
            # 将收集到的值列表转换为 Numpy Array
            data_mem = np.array([float(v) if v is not None else 0.0 for v in data_mem_list], dtype=np.float32)
        except (ValueError, TypeError) as e:
             print(f"ERROR converting mem features: {e}")
             data_mem = None
    # --- 结束修改 ---

    return data_exec, data_mem # 返回两个 Numpy 数组或 None

# def prepare_inference_data(data, operator):
#     # 使用提前编码的字典将 jointype 和 table_names 转换为标签
#     data['jointype'] = jointype_encoding.get(data['jointype'], -1)
#     data['index_cost'] = calculate_index_cost(data['index_names'], table_structure, column_type_cost_dict)
#     data['predicate_cost'] = extract_predicate_cost(data['filter'])
#     features_exec = None
#     features_mem = None
#     data_exec = None
#     data_mem = None
#     if operator in dop_operators_exec:
#         features_exec = dop_operator_features[operator]['exec'] 
#     elif operator in no_dop_operators_exec:
#         features_exec = no_dop_operator_features[operator]['exec'] 
#     if operator in dop_operators_mem:
#         features_mem = dop_operator_features[operator]['mem']  
#     elif operator in no_dop_operators_mem:
#         features_mem = no_dop_operator_features[operator]['mem'] 
    
#     # 目标列
#     if features_exec is not None:
#         data_exec = data[features_exec]
#     if features_mem is not None:
#         data_mem = data[features_mem]

#     return data_exec, data_mem

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