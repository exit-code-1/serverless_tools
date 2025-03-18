from math import sqrt

# 1. 定义列类型到开销的映射
column_type_cost_dict = {
    'INT': 4,              # INT 类型开销 4 字节
    'BIGINT': 8,           # BIGINT 类型开销 8 字节
    'CHAR': lambda s: 4 + s,   # CHAR(n) 类型开销为 n 字节
    'VARCHAR': lambda s: 4 + s, # VARCHAR(n) 类型开销为 n 字节
    'DECIMAL': lambda p, s: 4 + 4 + (p + s) // 2,  # DECIMAL(p, s) 假设开销为 (p + s) / 2 字节
    'DATE': 8              # DATE 类型开销 8 字节
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
table_names = ['none', 'region', 'nation', 'supplier', 'customer', 'part', 'partsupp', 'orders', 'lineitem']

# 创建编码字典
jointype_encoding = {jointype: idx for idx, jointype in enumerate(jointypes)}
table_names_encoding = {table_name: idx for idx, table_name in enumerate(table_names)}

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
        'Vector Merge Join',
        'Aggregate',
        'Hash',
        'Vector WindowAgg',
        'Append',
        'Index Only Scan',
        'Hash Join'
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
]

no_dop_operators_mem = [
        'Vector Merge Join',
        'Aggregate',
        'Hash',
        'Vector WindowAgg',
        'Append',
        'Hash Join',
        'Vector Materialize',
        'Vector Aggregate',
        'Vector Sort',
        'Vector Hash Aggregate',
        'Vector Sonic Hash Aggregate',
        'Vector Hash Join',
        'Vector Sonic Hash Join',
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
    'Aggregate': {
        'exec': ['l_input_rows', 'width', 'agg_col', 'agg_width'],
        'mem': ['l_input_rows', 'width', 'agg_col', 'agg_width']
    },
    'Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size'],
        'mem': ['r_input_rows', 'width', 'jointype', 'hash_table_size']
    },
    'Hash': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Vector WindowAgg': {
        'exec': ['l_input_rows', 'actual_rows', 'width'],
        'mem': ['l_input_rows', 'actual_rows', 'width']
    },
    'Append': {
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
    'Vector Hash Aggregate': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio'],
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    },
    'Vector Sonic Hash Aggregate': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio'],
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    },
    'Vector Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size'],
        'mem': ['r_input_rows', 'width', 'hash_table_size']
    },
    'Vector Sonic Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost','hash_table_size'],
        'mem': ['r_input_rows', 'width', 'hash_table_size']
    },
    'Vector SetOp': {
        'exec': ['l_input_rows',  'actual_rows', 'width'],
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
    'Vector Hash Aggregate': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio'],
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    },
    'Vector Sonic Hash Aggregate': {
        'exec': ['l_input_rows', 'actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio'],
        'mem': ['actual_rows', 'width', 'agg_col', 'agg_width', 'hash_table_size', 'disk_ratio']
    },
    'Vector Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost', 'hash_table_size'],
        'mem': ['r_input_rows', 'width', 'hash_table_size']
    },
    'Vector Sonic Hash Join': {
        'exec': ['l_input_rows', 'r_input_rows', 'actual_rows', 'width', 'jointype', 'predicate_cost','hash_table_size'],
        'mem': ['r_input_rows', 'width', 'hash_table_size']
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

    # Add more operators and their corresponding feature sets here as needed
}

dop_train_epochs = {
    'CStore Scan': {
        'exec': 100,
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
        'exec': 100,
        'mem': 100
    },
    'Vector Hash Join': {
        'exec': 100,
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
        'exec': 100,
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


    # Add more operators and their corresponding feature sets here as needed
}