from math import sqrt
import random
import re
import numpy as np
import pandas as pd
import glob
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split


import pandas as pd

# 1. 定义列类型到开销的映射
column_type_cost_dict = {
    'INT': 4,              # INT 类型开销 4 字节
    'BIGINT': 8,           # BIGINT 类型开销 8 字节
    'CHAR': lambda s: 4 + sqrt(s),   # CHAR(n) 类型开销为 n 字节
    'VARCHAR': lambda s: 4 + sqrt(s), # VARCHAR(n) 类型开销为 n 字节
    'DECIMAL': lambda p, s: 4 + 4 + sqrt((p + s) // 2),  # DECIMAL(p, s) 假设开销为 (p + s) / 2 字节
    'DATE': 4              # DATE 类型开销 4 字节
}

# 2. 预定义的表结构及列类型
table_structure = {
    'REGION': [
        ('R_REGIONKEY', 'INT'),
        ('R_NAME', 'CHAR(25)'),
        ('R_COMMENT', 'VARCHAR(152)')
    ],
    'NATION': [
        ('N_NATIONKEY', 'INT'),
        ('N_NAME', 'CHAR(25)'),
        ('N_REGIONKEY', 'INT'),
        ('N_COMMENT', 'VARCHAR(152)')
    ],
    'SUPPLIER': [
        ('S_SUPPKEY', 'BIGINT'),
        ('S_NAME', 'CHAR(25)'),
        ('S_ADDRESS', 'VARCHAR(40)'),
        ('S_NATIONKEY', 'INT'),
        ('S_PHONE', 'CHAR(15)'),
        ('S_ACCTBAL', 'DECIMAL(15,2)'),
        ('S_COMMENT', 'VARCHAR(101)')
    ],
    'CUSTOMER': [
        ('C_CUSTKEY', 'BIGINT'),
        ('C_NAME', 'VARCHAR(25)'),
        ('C_ADDRESS', 'VARCHAR(40)'),
        ('C_NATIONKEY', 'INT'),
        ('C_PHONE', 'CHAR(15)'),
        ('C_ACCTBAL', 'DECIMAL(15,2)'),
        ('C_MKTSEGMENT', 'CHAR(10)'),
        ('C_COMMENT', 'VARCHAR(117)')
    ],
    'PART': [
        ('P_PARTKEY', 'BIGINT'),
        ('P_NAME', 'VARCHAR(100)'),
        ('P_MFGR', 'CHAR(100)'),
        ('P_BRAND', 'CHAR(20)'),
        ('P_TYPE', 'VARCHAR(100)'),
        ('P_SIZE', 'BIGINT'),
        ('P_CONTAINER', 'CHAR(10)'),
        ('P_RETAILPRICE', 'DECIMAL(15,2)'),
        ('P_COMMENT', 'VARCHAR(23)')
    ],
    'PARTSUPP': [
        ('PS_PARTKEY', 'BIGINT'),
        ('PS_SUPPKEY', 'BIGINT'),
        ('PS_AVAILQTY', 'BIGINT'),
        ('PS_SUPPLYCOST', 'DECIMAL(15,2)'),
        ('PS_COMMENT', 'VARCHAR(199)')
    ],
    'ORDERS': [
        ('O_ORDERKEY', 'BIGINT'),
        ('O_CUSTKEY', 'BIGINT'),
        ('O_ORDERSTATUS', 'CHAR(1)'),
        ('O_TOTALPRICE', 'DECIMAL(15,2)'),
        ('O_ORDERDATE', 'DATE'),
        ('O_ORDERPRIORITY', 'CHAR(15)'),
        ('O_CLERK', 'CHAR(15)'),
        ('O_SHIPPRIORITY', 'BIGINT'),
        ('O_COMMENT', 'VARCHAR(79)')
    ],
    'LINEITEM': [
        ('L_ORDERKEY', 'BIGINT'),
        ('L_PARTKEY', 'BIGINT'),
        ('L_SUPPKEY', 'BIGINT'),
        ('L_LINENUMBER', 'BIGINT'),
        ('L_QUANTITY', 'DECIMAL(15,2)'),
        ('L_EXTENDEDPRICE', 'DECIMAL(15,2)'),
        ('L_DISCOUNT', 'DECIMAL(15,2)'),
        ('L_TAX', 'DECIMAL(15,2)'),
        ('L_RETURNFLAG', 'CHAR(1)'),
        ('L_LINESTATUS', 'CHAR(1)'),
        ('L_SHIPDATE', 'DATE'),
        ('L_COMMITDATE', 'DATE'),
        ('L_RECEIPTDATE', 'DATE'),
        ('L_SHIPINSTRUCT', 'CHAR(25)'),
        ('L_SHIPMODE', 'CHAR(10)'),
        ('L_COMMENT', 'VARCHAR(44)')
    ]
}

# 假设已知的 jointype 和 table_names 类型
jointypes = ['none', 'Inner', 'Right', 'Left', 'Full', 'Semi', 'Anti', 'Right Semi', 'Right Anti', 'Left Anti Full', 'Right Anti Full']
table_names = ['none', 'none,region', 'none,nation', 'none,supplier', 'none,customer', 'none,part', 'none,partsupp', 'none,orders', 'none,lineitem']

# 创建编码字典
jointype_encoding = {jointype: idx for idx, jointype in enumerate(jointypes)}
table_names_encoding = {table_name: idx for idx, table_name in enumerate(table_names)}

# 2. 根据表名和列类型创建特征词典
def create_feature_dict(table_structure, column_type_cost_dict):
    feature_dict = []
    
    # 遍历所有表和列类型
    for table_name, columns in table_structure.items():
        for column_name, column_type in columns:
            # 获取列的类型和对应的开销
            if 'CHAR' in column_type or 'VARCHAR' in column_type:
                # 提取列的长度（数字部分）
                length = int(column_type.split('(')[1].split(')')[0])
                column_cost = column_type_cost_dict[column_type.split('(')[0]](length)
            elif 'DECIMAL' in column_type:
                # 提取 DECIMAL 的精度和标度
                precision, scale = map(int, column_type.split('(')[1].split(')')[0].split(','))
                column_cost = column_type_cost_dict['DECIMAL'](precision, scale)
            else:
                # 对于其他类型，直接获取对应的开销
                column_cost = column_type_cost_dict.get(column_type, 0)
            
            feature_dict.append({
                'table_name': table_name,
                'column_name': column_name,
                'column_type': column_type,
                'column_cost': column_cost
            })
    
    return feature_dict


def extract_predicate_cost(predicate):
    total_cost = 0
    
    # 处理日期偏移量，例如 NUMTODSINTERVAL
    predicate = re.sub(r"NUMTODSINTERVAL\((\d+),\s*'([A-Z]+)'\)", lambda m: f"INTERVAL({m.group(1)})", predicate)

    # 拆分谓词 (AND 和 OR 连接的部分)
    predicate_parts = re.split(r'\sAND\s|\sOR\s', predicate)
    
    for part in predicate_parts:
        # 提取列名和操作符
        column_name_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)', part)
        if not column_name_match:
            continue
        column_name = column_name_match.group(1)
        
        # 判断列类型，这里可以根据实际的表结构映射
        if 'acctbal' in column_name:
            column_type = 'DECIMAL(15,2)'
        elif 'shipdate' in column_name or 'orderdate' in column_name:
            column_type = 'DATE'
        else:
            column_type = 'CHAR(15)'  # 默认类型

        # 处理不同类型的谓词
        operator_cost = 0
        if 'ANY' in part:
            # 处理 = ANY 操作符
            num_elements = len(re.findall(r'\d+', part))  # 统计 ANY 中的元素个数
            column_type_base = column_type.split('(')[0]
            if 'DECIMAL' in column_type:
                precision, scale = map(int, column_type.split('(')[1].split(')')[0].split(','))
                operator_cost = num_elements * column_type_cost_dict[column_type_base](precision, scale)
            else:
                length = int(column_type.split('(')[1].split(')')[0]) if '(' in column_type else 0
                operator_cost = num_elements * column_type_cost_dict[column_type_base](length) if callable(column_type_cost_dict[column_type_base]) else column_type_cost_dict[column_type_base]
        elif 'LIKE' in part:
            # 处理 LIKE 操作符
            column_type_base = column_type.split('(')[0]
            if 'DECIMAL' in column_type:
                precision, scale = map(int, column_type.split('(')[1].split(')')[0].split(','))
                operator_cost = column_type_cost_dict[column_type_base](precision, scale) * 2
            else:
                length = int(column_type.split('(')[1].split(')')[0]) if '(' in column_type else 0
                operator_cost = column_type_cost_dict[column_type_base](length) * 2 if callable(column_type_cost_dict[column_type_base]) else column_type_cost_dict[column_type_base] * 2
        elif '=' in part or '<' in part or '>' in part:
            # 普通的比较操作符
            column_type_base = column_type.split('(')[0]
            if 'DECIMAL' in column_type:
                precision, scale = map(int, column_type.split('(')[1].split(')')[0].split(','))
                operator_cost = column_type_cost_dict[column_type_base](precision, scale)
            else:
                length = int(column_type.split('(')[1].split(')')[0]) if '(' in column_type else 0
                operator_cost = column_type_cost_dict[column_type_base](length) if callable(column_type_cost_dict[column_type_base]) else column_type_cost_dict[column_type_base]
        elif 'IN' in part:
            # 处理 IN 操作符，类似于 ANY
            num_elements = len(re.findall(r'\d+', part))  # 统计 IN 中的元素个数
            column_type_base = column_type.split('(')[0]
            if 'DECIMAL' in column_type:
                precision, scale = map(int, column_type.split('(')[1].split(')')[0].split(','))
                operator_cost = num_elements * column_type_cost_dict[column_type_base](precision, scale)
            else:
                length = int(column_type.split('(')[1].split(')')[0]) if '(' in column_type else 0
                operator_cost = num_elements * column_type_cost_dict[column_type_base](length) if callable(column_type_cost_dict[column_type_base]) else column_type_cost_dict[column_type_base]
        elif 'INTERVAL' in part:
            # 处理日期间隔
            operator_cost = 4  # 假设日期间隔的开销为 4 字节
        else:
            column_type_base = column_type.split('(')[0]
            if 'DECIMAL' in column_type:
                precision, scale = map(int, column_type.split('(')[1].split(')')[0].split(','))
                operator_cost = column_type_cost_dict[column_type_base](precision, scale)
            else:
                length = int(column_type.split('(')[1].split(')')[0]) if '(' in column_type else 0
                operator_cost = column_type_cost_dict[column_type_base](length) if callable(column_type_cost_dict[column_type_base]) else column_type_cost_dict[column_type_base]
        
        total_cost += operator_cost
    
    return total_cost

def extract_index_name(index_name):
    """
    从 index_name 中提取列名。
    例如：(s_suppkey = $1) -> s_suppkey
    """
    if pd.isnull(index_name) or index_name == '':
        return None
    # 匹配左边的列名
    match = re.search(r'\(([a-zA-Z_][a-zA-Z0-9_]*)', index_name)
    if match:
        return match.group(1)
    return None

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



# 定义实例归一化函数
def instance_normalize(data, epsilon=1e-5):
    """
    Perform instance normalization on the data.
    Parameters:
    - data: pd.DataFrame of shape (N, C)
    - epsilon: a small constant to avoid division by zero
    Returns:
    - normalized_data: pd.DataFrame of shape (N, C)
    """
    num_instances, num_channels = data.shape
    normalized_data = data.copy()
    for i in range(num_instances):
        instance = data.iloc[i]
        mean = np.mean(instance)
        variance = np.var(instance)
        normalized_data.iloc[i] = abs(instance - mean) / np.sqrt(variance + epsilon)
    return normalized_data


def prepare_data(data, operator, feature_columns, target_columns, train_queries, test_queries, epsilon=1e-5):
    # Filter by operator type
    operator_data = data[data['operator_type'] == operator].copy()  # Add .copy() to avoid chained assignment warnings
    
    # 添加 index_cost 列
    operator_data['index_cost'] = operator_data['index_names'].apply(
        lambda x: calculate_index_cost(x, table_structure, column_type_cost_dict)
    )
    # 对 filter 列进行谓词开销计算
    operator_data['predicate_cost'] = operator_data['filter'].apply(
        lambda x: extract_predicate_cost(x) if pd.notnull(x) and x != '' else 0
    )

    # Assign train/test split based on query_id
    operator_data.loc[:, 'set'] = operator_data['query_id'].apply(
        lambda qid: 'train' if ((qid - 1) % 200 + 1 in train_queries) else ('test' if ((qid - 1) % 200 + 1 in test_queries) else 'exclude')
    )

    # 使用提前编码的字典将 jointype 和 table_names 转换为标签
    operator_data['jointype'] = operator_data['jointype'].map(jointype_encoding).astype(int)
    operator_data['table_names'] = operator_data['table_names'].map(table_names_encoding).astype(int)

    # Split into train and test data
    train_data = operator_data[operator_data['set'] == 'train']
    test_data = operator_data[operator_data['set'] == 'test']

    # 选取特征和目标
    # 移除分类类型的列，不参与归一化
    categorical_columns = ['jointype', 'table_names']  # 你可以根据需要调整
    numerical_columns = [col for col in feature_columns if col not in categorical_columns]

    # 对数值列进行归一化
    X_train = instance_normalize(train_data[numerical_columns], epsilon)
    X_test = instance_normalize(test_data[numerical_columns], epsilon)
    
    # 将分类列保留并添加到归一化后的数据中
    X_train[categorical_columns] = train_data[categorical_columns]
    X_test[categorical_columns] = test_data[categorical_columns]

    # 目标列
    y_train = train_data[target_columns]
    y_test = test_data[target_columns]

    return X_train, X_test, y_train, y_test


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