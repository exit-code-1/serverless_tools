from math import sqrt

default_dop = 64
thread_cost = 6
thread_mem = 8194
dop_sets = {1,8,16,32,64,96}

# --- 新增：特征开关配置 ---
# 设置为 False 来运行移除 hash_table_size 特征的实验
USE_HASH_TABLE_SIZE_FEATURE = False
# --- 配置结束 ---

# 1. 定义列类型到开销的映射
column_type_cost_dict = {
    'INT': 4,              # INT 类型开销 4 字节
    'BIGINT': 8,           # BIGINT 类型开销 8 字节
    'CHAR': lambda s: 4 + s * 2,   # CHAR(n) 类型开销为 n 字节
    'VARCHAR': lambda s: 4 + s * 2, # VARCHAR(n) 类型开销为 n 字节
    'DECIMAL': lambda p, s: 4 + 4 + (p + s),  # DECIMAL(p, s) 假设开销为 (p + s) / 2 字节
    'DATE': 8              # DATE 类型开销 8 字节
}

tpcds_non_parallel = {1,2,3,9,23,24,30,32,33,42,54,56,57,59,60,63,64,77,80,81,92}

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

# OPERATORS_WITHOUT_FEATURES = {
#     'Row Adapter', 'Vector Limit', 'Vector Sort Aggregate',
#     'Vector Subquery Scan', 'Sort', 'Nested Loop',
#     # 根据你的实际情况添加或删除
# }

# 假设已知的 jointype 和 table_names 类型
jointypes = ['none', 'Inner', 'Right', 'Left', 'Full', 'Semi', 'Anti', 'Right Semi', 'Right Anti', 'Left Anti Full', 'Right Anti Full']
table_names = ['none', 'region', 'nation', 'supplier', 'customer', 'part', 'partsupp', 'orders', 'lineitem']
operator_type =['CStore Index Scan',
        'Vector Nest Loop',
        'Vector Merge Join',
        'Aggregate',
        'Hash',
        'Vector WindowAgg',
        'Append',
        'Index Only Scan',
        'Hash Join',
        'CStore Scan',
        'Vector Materialize',
        'Vector Aggregate',
        'Vector Sort',
        'Vector Hash Aggregate',
        'Vector Sonic Hash Aggregate',
        'Vector Hash Join',
        'Vector Sonic Hash Join',
        'Vector Streaming LOCAL GATHER',
        'Vector Streaming LOCAL REDISTRIBUTE', 
        'Vector Streaming BROADCAST',
        'Vector SetOp',
        'Vector Append',
        'Row Adapter',
        'Vector Limit',
        'Vector Subquery Scan'
        ]
# 创建编码字典
jointype_encoding = {jointype: idx for idx, jointype in enumerate(jointypes)}
table_names_encoding = {table_name: idx for idx, table_name in enumerate(table_names)}
operator_encoding = {operator_type: idx for idx, operator_type in enumerate(operator_type)}

parallel_op = [
        'CStore Scan',
        'Vector Materialize',
        'Vector Aggregate',
        'Vector Sort',
        'Vector Hash Aggregate',
        'Vector Sonic Hash Aggregate',
        'Vector Hash Join',
        'Vector Sonic Hash Join',
        'Vector Streaming LOCAL GATHER',
        'Vector Streaming LOCAL REDISTRIBUTE', 
        'Vector Streaming BROADCAST',
        'Vector SetOp',
        'Vector Append',
        'Vector Limit',
        'Vector Subquery Scan',
        'Vector Sort Aggregate',
        'Aggregate',
        'Hash',
        'Hash Join',
        'Row Adapter',
]

operator_lists = [
        'CStore Index Scan',
        'Vector Nest Loop',
        'Vector Merge Join',
        'Aggregate',
        'Hash',
        'Vector WindowAgg',
        'Append',
        'Index Only Scan',
        'Hash Join',
        'CStore Scan',
        'Vector Materialize',
        'Vector Aggregate',
        'Vector Sort',
        'Vector Hash Aggregate',
        'Vector Sonic Hash Aggregate',
        'Vector Hash Join',
        'Vector Sonic Hash Join',
        'Vector Streaming LOCAL GATHER',
        'Vector Streaming LOCAL REDISTRIBUTE', 
        'Vector Streaming BROADCAST',
        'Vector SetOp',
        'Vector Append',
]
no_dop_operators_exec = [
        'CStore Index Scan',
        'Vector Nest Loop',
        'Vector WindowAgg',
        'Index Only Scan',
        'Vector Merge Join',
]

dop_operators_exec = [
        'CStore Scan',
        'Vector Materialize',
        'Vector Aggregate',
        'Vector Sort',
        'Vector Hash Aggregate',
        'Vector Sonic Hash Aggregate',
        'Vector Hash Join',
        'Vector Sonic Hash Join',
        'Vector Streaming LOCAL GATHER',
        'Vector Streaming LOCAL REDISTRIBUTE', 
        'Vector Streaming BROADCAST',
        'Vector SetOp',
        'Vector Append',
        'Aggregate',
        'Hash',
        'Append',
        'Hash Join',
]

no_dop_operators_mem = [
    'Vector Materialize',
    'Vector Aggregate',
    'Vector Sort',
    'Vector Hash Aggregate',
    'Vector Sonic Hash Aggregate',
    'Vector Hash Join',
    'Vector Sonic Hash Join',
    'Vector WindowAgg',
    'Aggregate',
    'Hash',
]

dop_operators_mem = [
        # 'Vector Materialize',
        # 'Vector Aggregate',
        # 'Vector Sort',
        # 'Vector Hash Aggregate',
        # 'Vector Sonic Hash Aggregate',
        # 'Vector Hash Join',
        # 'Vector Sonic Hash Join',
        # 'Vector SetOp',
]


no_dop_operator_features = {
    'CStore Index Scan': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'index_cost', 'predicate_cost'],
        'mem': ['l_input_rows', 'actual_rows', 'width', 'query_dop']
    },
    'CTE Scan': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'predicate_cost'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    # Add mappings for other operators here
    'Vector Nest Loop': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'query_dop'],
        'mem': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'query_dop']
    },
    'Vector Merge Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype'],
        'mem': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width','jointype']
    },
    'Vector WindowAgg': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Index Only Scan': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'index_cost', 'predicate_cost'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector Aggregate': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'agg_width'],
        'mem': ['l_input_rows', 'actual_rows', 'width', 'agg_width']
    },
    'Vector Sort': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector Materialize': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    # 'Vector Hash Aggregate': {
    #     'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio'],
    #     'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    # },
    'Vector Hash Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    # 'Vector Sonic Hash Aggregate': {
    #     'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio'],
    #     'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    # },
    'Vector Sonic Hash Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    # 'Vector Hash Join': {
    #     'exec': ['l_input_rows', 'r_input_rows',  'width', 'jointype', 'predicate_cost', 'hash_table_size'],
    #     'mem': ['r_input_rows', 'width', 'hash_table_size']
    # },
    # 'Vector Sonic Hash Join': {
    #     'exec': ['l_input_rows', 'r_input_rows',  'width', 'jointype', 'predicate_cost','hash_table_size'],
    #     'mem': ['r_input_rows', 'width', 'hash_table_size']
    # },
    'Vector Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'width', 'jointype', 'predicate_cost'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['r_input_rows', 'width'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    'Vector Sonic Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'width', 'jointype', 'predicate_cost'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['r_input_rows', 'width'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    'Vector SetOp': {
        'exec': ['l_input_rows',  'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width'],
        'mem': ['l_input_rows', 'width', 'agg_col', 'agg_width']
    },
    'Vector Sort Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width'],
        'mem': ['l_input_rows', 'width', 'agg_col', 'agg_width']
    },
    # 'Hash Join': {
    #     'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size'],
    #     'mem': ['r_input_rows', 'width', 'jointype', 'hash_table_size']
    # },
    'Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['r_input_rows', 'width', 'jointype'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    'Hash': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Append': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    # Add more operators and their corresponding feature sets here as needed
}

dop_operator_features = {
    'CStore Scan': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'predicate_cost'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector Aggregate': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'agg_width'],
        'mem': ['l_input_rows', 'actual_rows', 'width', 'agg_width']
    },
    'Vector Sort': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector Materialize': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    # 'Vector Hash Aggregate': {
    #     'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'],
    #     'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    # },
    'Vector Hash Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'],
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    # 'Vector Sonic Hash Aggregate': {
    #     'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'],
    #     'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    # },
    'Vector Sonic Hash Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width', 'disk_ratio'],
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width',  'disk_ratio'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    # 'Vector Hash Join': {
    #     'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size'],
    #     'mem': ['r_input_rows', 'width', 'hash_table_size']
    # },
    'Vector Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['r_input_rows', 'width'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    # 'Vector Sonic Hash Join': {
    #     'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost','hash_table_size'],
    #     'mem': ['r_input_rows', 'width', 'hash_table_size']
    # },
    'Vector Sonic Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['r_input_rows', 'width'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    'Vector Streaming LOCAL GATHER': {
        'exec': ['l_input_rows',  'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector Streaming LOCAL REDISTRIBUTE': {
        'exec': ['l_input_rows',  'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector Streaming BROADCAST': {
        'exec': ['l_input_rows',  'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector SetOp': {
        'exec': ['l_input_rows',  'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector Append': {
        'exec': ['l_input_rows',  'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width'],
        'mem': ['l_input_rows', 'width', 'agg_col', 'agg_width']
    },
    'Vector Sort Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width'],
        'mem': ['l_input_rows', 'width', 'agg_col', 'agg_width']
    },
    # 'Hash Join': {
    #     'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size'],
    #     'mem': ['r_input_rows', 'width', 'jointype', 'hash_table_size']
    # },
    'Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else []),
        'mem': ['r_input_rows', 'jointype', 'width'] + (['hash_table_size'] if USE_HASH_TABLE_SIZE_FEATURE else [])
    },
    'Hash': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Append': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },

    # Add more operators and their corresponding feature sets here as needed
}

dop_train_epochs = {
    'CStore Scan': {
        'exec': 150,
        'mem': 20
    },
    'Vector Aggregate': {
        'exec': 100,
        'mem': 100
    },
    'Vector Sort': {
        'exec': 100,
        'mem': 100
    },
    'Vector Materialize': {
        'exec': 100,
        'mem': 100
    },
    'Vector Hash Aggregate': {
        'exec': 100,
        'mem': 100
    },
    'Vector Sonic Hash Aggregate': {
        'exec': 200,
        'mem': 100
    },
    'Vector Hash Join': {
        'exec': 200,
        'mem': 100
    },
    'Vector Sonic Hash Join': {
        'exec': 100,
        'mem': 100
    },
    'Vector Streaming LOCAL GATHER': {
        'exec': 100,
        'mem': 100
    },
    'Vector Streaming LOCAL REDISTRIBUTE': {
        'exec': 120,
        'mem': 100
    },
    'Vector Streaming BROADCAST': {
        'exec': 100,
        'mem': 100
    },
    'Vector SetOp': {
        'exec': 100,
        'mem': 100
    },
    'Vector Append': {
        'exec': 100,
        'mem': 100
    },
    'Aggregate': {
        'exec': 100,
        'mem': 50
    },
    'Append': {
        'exec': 100,
        'mem': 50
    },
    'Hash Join': {
        'exec': 100,
        'mem': 50
    },
    'Hash': {
        'exec': 100,
        'mem': 50
    },
    'Vector Append': {
        'exec': 100,
        'mem': 50
    },
    'Vector Sort Aggregate': {
        'exec': 100,
        'mem': 50
    },

    # Add more operators and their corresponding feature sets here as needed
}