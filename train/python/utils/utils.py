import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


import pandas as pd

# 1. 定义列类型到开销的映射
column_type_cost_dict = {
    'INT': 4,              # INT 类型开销 4 字节
    'BIGINT': 8,           # BIGINT 类型开销 8 字节
    'CHAR': lambda n: n,   # CHAR(n) 类型开销为 n 字节
    'VARCHAR': lambda n: n, # VARCHAR(n) 类型开销为 n 字节
    'DECIMAL': lambda p, s: (p + s) // 2,  # DECIMAL(p, s) 假设开销为 (p + s) / 2 字节
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

# 3. 根据表名和列类型创建特征词典
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


def prepare_data(csv_path_pattern, operator, feature_columns, target_columns, train_queries, test_queries, epsilon=1e-5):
    # Load and merge data
    csv_files = glob.glob(csv_path_pattern, recursive=True)
    data_list = [pd.read_csv(file, delimiter=';', encoding='utf-8') for file in csv_files]
    data = pd.concat(data_list, ignore_index=True)

    # Filter by operator type
    operator_data = data[data['operator_type'] == operator].copy()  # Add .copy() to avoid chained assignment warnings

    # Assign train/test split based on query_id
    operator_data.loc[:, 'set'] = operator_data['query_id'].apply(
        lambda qid: 'train' if (qid % 22 in train_queries) else ('test' if (qid % 22 in test_queries) else 'exclude')
    )

    train_data = operator_data[operator_data['set'] == 'train']
    test_data = operator_data[operator_data['set'] == 'test']

    # Select features and targets
    X_train = instance_normalize(train_data[feature_columns], epsilon)
    X_test = instance_normalize(test_data[feature_columns], epsilon)
    # X_test = test_data[feature_columns]
    # X_train = train_data[feature_columns]
    y_train = train_data[target_columns]
    y_test = test_data[target_columns]

    return X_train, X_test, y_train, y_test


def prepare_index_data(csv_path_pattern, operator, feature_columns, target_columns, train_queries, test_queries, epsilon=1e-5, column_type_cost_dict=None, table_structure=None):
    """
    Process the data and add additional features based on table name and index type.
    
    Parameters:
    - csv_path_pattern: str, glob pattern to find CSV files
    - operator: str, the operator type to filter data
    - feature_columns: list of str, feature columns to use for training
    - target_columns: list of str, target columns to predict
    - train_queries: list of int, query IDs to use for training
    - test_queries: list of int, query IDs to use for testing
    - epsilon: float, a small constant to avoid division by zero in normalization
    - column_type_cost_dict: dict, mapping of column types to storage costs
    - table_structure: dict, table structure defining column types

    Returns:
    - X_train: DataFrame, training features
    - X_test: DataFrame, testing features
    - y_train: DataFrame, training targets
    - y_test: DataFrame, testing targets
    """
    # Load and merge data
    csv_files = glob.glob(csv_path_pattern, recursive=True)
    data_list = [pd.read_csv(file, delimiter=';', encoding='utf-8') for file in csv_files]
    data = pd.concat(data_list, ignore_index=True)

    # Filter by operator type
    operator_data = data[data['operator_type'] == operator].copy()  # Add .copy() to avoid chained assignment warnings

    # Assign train/test split based on query_id
    operator_data.loc[:, 'set'] = operator_data['query_id'].apply(
        lambda qid: 'train' if (qid % 22 in train_queries) else ('test' if (qid % 22 in test_queries) else 'exclude')
    )

    train_data = operator_data[operator_data['set'] == 'train']
    test_data = operator_data[operator_data['set'] == 'test']

    # Add features based on table name and column type cost
    def get_column_type_cost(table_name, column_name):
        """Retrieve the storage cost of the column based on table structure and column type."""
        table_name = table_name.lower()  # Convert table name to lowercase
        column_name = column_name.lower()  # Convert column name to lowercase
        
        for table, columns in table_structure.items():
            if table.lower() == table_name:  # Match table name without case sensitivity
                for col_name, col_type in columns:
                    if col_name.lower() == column_name:  # Match column name without case sensitivity
                        if 'CHAR' in col_type or 'VARCHAR' in col_type:
                            length = int(col_type.split('(')[1].split(')')[0])
                            return column_type_cost_dict[col_type.split('(')[0]](length)
                        elif 'DECIMAL' in col_type:
                            precision, scale = map(int, col_type.split('(')[1].split(')')[0].split(','))
                            return column_type_cost_dict['DECIMAL'](precision, scale)
                        else:
                            return column_type_cost_dict.get(col_type, 0)
        return 0  # Default if no match found

    # Create new features for table name and index type cost
    train_data['table_name'] = train_data['table_name_column']  # Replace with actual column name containing table names
    train_data['index_type_cost'] = train_data.apply(lambda row: get_column_type_cost(row['table_name'], row['index_column_name']), axis=1)

    test_data['table_name'] = test_data['table_name_column']  # Replace with actual column name containing table names
    test_data['index_type_cost'] = test_data.apply(lambda row: get_column_type_cost(row['table_name'], row['index_column_name']), axis=1)

    # Select features and targets
    # X_train = instance_normalize(train_data[feature_columns], epsilon)
    X_train = train_data[feature_columns + ['index_type_cost']]  # Add the new feature
    y_train = train_data[target_columns]
    
    # X_test = instance_normalize(test_data[feature_columns], epsilon)
    X_test = test_data[feature_columns + ['index_type_cost']]  # Add the new feature
    y_test = test_data[target_columns]

    return X_train, X_test, y_train, y_test