"""
Simple analysis of startup time relationship with operator number and pipeline count.
This script analyzes the data and suggests optimal parameters for estimate_startup_time.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_environment

def count_streaming_operators(df_plans, query_id, query_dop):
    """Count streaming operators in a query plan."""
    query_plan = df_plans[
        (df_plans['query_id'] == query_id) & 
        (df_plans['query_dop'] == query_dop)
    ]
    
    if 'operator_type' in query_plan.columns:
        streaming_ops = query_plan[
            query_plan['operator_type'].str.contains('Streaming', case=False, na=False)
        ]
        return len(streaming_ops)
    return 0

def analyze_startup_time_simple(dataset: str):
    """
    Analyze startup time relationship and suggest optimal parameters.
    """
    print("=" * 80)
    print(f"分析启动时间关系 - 数据集: {dataset}")
    print("=" * 80)
    
    # Setup environment
    setup_environment()
    
    # Load data
    if dataset == 'tpcds':
        query_csv_path = "data_kunpeng/tpcds_100g_output_test/query_info.csv"
        plan_csv_path = "data_kunpeng/tpcds_100g_output_test/plan_info.csv"
    elif dataset == 'tpch':
        query_csv_path = "data_kunpeng/tpch_output_22/query_info.csv"
        plan_csv_path = "data_kunpeng/tpch_output_22/plan_info.csv"
    else:
        print(f"❌ 未知数据集: {dataset}")
        return None
    
    if not os.path.exists(query_csv_path):
        print(f"❌ 查询信息文件不存在: {query_csv_path}")
        return None
    if not os.path.exists(plan_csv_path):
        print(f"❌ 计划信息文件不存在: {plan_csv_path}")
        return None
    
    # Load data
    df_query = pd.read_csv(query_csv_path, sep=';')
    df_plans = pd.read_csv(plan_csv_path, sep=';')
    
    print(f"加载了 {len(df_query)} 条查询记录")
    print(f"加载了 {len(df_plans)} 条计划记录")
    
    # Prepare analysis data
    analysis_data = []
    
    for _, row in df_query.iterrows():
        query_id = row['query_id']
        query_dop = row['dop']
        startup_time = row['executor_start_time']
        operator_num = row['operator_num']
        
        # Count streaming operators
        streaming_count = count_streaming_operators(df_plans, query_id, query_dop)
        
        analysis_data.append({
            'query_id': query_id,
            'dop': query_dop,
            'startup_time': startup_time,
            'operator_num': operator_num,
            'streaming_count': streaming_count
        })
    
    df_analysis = pd.DataFrame(analysis_data)
    
    print("\n" + "=" * 80)
    print("数据统计")
    print("=" * 80)
    print(df_analysis.describe())
    
    print("\n相关性分析:")
    corr = df_analysis[['startup_time', 'operator_num', 'streaming_count']].corr()
    print(corr)
    
    # Simple linear regression: startup_time = a + b * operator_num + c * streaming_count
    # Using normal equations: (X^T * X) * beta = X^T * y
    X = np.column_stack([
        np.ones(len(df_analysis)),  # intercept
        df_analysis['operator_num'].values,
        df_analysis['streaming_count'].values
    ])
    y = df_analysis['startup_time'].values
    
    # Solve: beta = (X^T * X)^(-1) * X^T * y
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    beta = np.linalg.solve(XTX, XTy)
    
    intercept = beta[0]
    coef_operator = beta[1]
    coef_streaming = beta[2]
    
    print("\n" + "=" * 80)
    print("线性回归结果")
    print("=" * 80)
    print(f"模型: startup_time = {intercept:.4f} + {coef_operator:.4f} * operator_num + {coef_streaming:.4f} * streaming_count")
    
    # Calculate R²
    y_pred = np.dot(X, beta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate MAE
    mae = np.mean(np.abs(y - y_pred))
    
    print(f"R² = {r2:.4f}")
    print(f"MAE = {mae:.4f} ms")
    print(f"RMSE = {np.sqrt(ss_res / len(y)):.4f} ms")
    
    # Simple model with only operator_num
    X_simple = np.column_stack([
        np.ones(len(df_analysis)),
        df_analysis['operator_num'].values
    ])
    XTX_simple = np.dot(X_simple.T, X_simple)
    XTy_simple = np.dot(X_simple.T, y)
    beta_simple = np.linalg.solve(XTX_simple, XTy_simple)
    
    intercept_simple = beta_simple[0]
    coef_operator_simple = beta_simple[1]
    
    y_pred_simple = np.dot(X_simple, beta_simple)
    ss_res_simple = np.sum((y - y_pred_simple) ** 2)
    r2_simple = 1 - (ss_res_simple / ss_tot)
    mae_simple = np.mean(np.abs(y - y_pred_simple))
    
    print("\n" + "=" * 80)
    print("简单模型 (仅算子数量)")
    print("=" * 80)
    print(f"模型: startup_time = {intercept_simple:.4f} + {coef_operator_simple:.4f} * operator_num")
    print(f"R² = {r2_simple:.4f}")
    print(f"MAE = {mae_simple:.4f} ms")
    print(f"RMSE = {np.sqrt(ss_res_simple / len(y)):.4f} ms")
    
    print("\n" + "=" * 80)
    print("建议的参数")
    print("=" * 80)
    print(f"对于 estimate_startup_time 函数:")
    print(f"  base_startup = {intercept:.2f}  # 基础启动时间 (ms)")
    print(f"  per_operator = {coef_operator:.2f}  # 每个算子的开销 (ms)")
    print(f"  per_streaming = {coef_streaming:.2f}  # 每个流式算子的开销 (ms)")
    
    return {
        'intercept': intercept,
        'coef_operator': coef_operator,
        'coef_streaming': coef_streaming,
        'r2': r2,
        'mae': mae,
        'df_analysis': df_analysis
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze startup time relationships')
    parser.add_argument('--dataset', type=str, default='tpcds', choices=['tpcds', 'tpch'],
                       help='Dataset to analyze')
    
    args = parser.parse_args()
    
    result = analyze_startup_time_simple(args.dataset)
    
    if result:
        print("\n✅ 分析完成!")

