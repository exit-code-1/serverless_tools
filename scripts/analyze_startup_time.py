"""
Analyze the relationship between executor startup time and:
1. Number of operators
2. Number of pipelines (streaming operators)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_environment

def count_pipelines(df_plans, query_id, query_dop):
    """
    Count the number of pipelines (streaming operators) in a query plan.
    A pipeline is identified by streaming operators (Vector Streaming LOCAL GATHER, 
    Vector Streaming LOCAL REDISTRIBUTE, Vector Streaming BROADCAST, etc.)
    """
    query_plan = df_plans[
        (df_plans['query_id'] == query_id) & 
        (df_plans['query_dop'] == query_dop)
    ]
    
    # Count streaming operators
    streaming_ops = query_plan[
        query_plan['operator_type'].str.contains('Streaming', case=False, na=False)
    ]
    
    # Count unique pipelines by looking at up_dop and down_dop
    # Each unique combination of up_dop/down_dop might represent a pipeline
    pipeline_count = 0
    if 'up_dop' in streaming_ops.columns and 'down_dop' in streaming_ops.columns:
        # Count distinct pipeline segments
        # A pipeline segment is defined by a change in DOP
        unique_dop_changes = streaming_ops[['up_dop', 'down_dop']].drop_duplicates()
        pipeline_count = len(unique_dop_changes)
    
    # Alternative: count streaming operators as proxy for pipeline count
    streaming_count = len(streaming_ops)
    
    return pipeline_count, streaming_count

def analyze_startup_time(dataset: str):
    """
    Analyze the relationship between executor startup time and:
    - Number of operators
    - Number of pipelines
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
    
    # Load query info
    df_query = pd.read_csv(query_csv_path, sep=';')
    print(f"加载了 {len(df_query)} 条查询记录")
    
    # Load plan info
    df_plans = pd.read_csv(plan_csv_path, sep=';')
    print(f"加载了 {len(df_plans)} 条计划记录")
    
    # Prepare analysis data
    analysis_data = []
    
    for _, row in df_query.iterrows():
        query_id = row['query_id']
        query_dop = row['dop']
        startup_time = row['executor_start_time']
        operator_num = row['operator_num']
        
        # Count pipelines
        pipeline_count, streaming_count = count_pipelines(df_plans, query_id, query_dop)
        
        analysis_data.append({
            'query_id': query_id,
            'dop': query_dop,
            'startup_time': startup_time,
            'operator_num': operator_num,
            'pipeline_count': pipeline_count,
            'streaming_count': streaming_count
        })
    
    df_analysis = pd.DataFrame(analysis_data)
    
    print("\n" + "=" * 80)
    print("数据统计")
    print("=" * 80)
    print(df_analysis.describe())
    print("\n相关性分析:")
    print(df_analysis[['startup_time', 'operator_num', 'pipeline_count', 'streaming_count']].corr())
    
    # Model 1: Linear regression with operator_num
    print("\n" + "=" * 80)
    print("模型1: 启动时间 = f(算子数量)")
    print("=" * 80)
    
    X1 = df_analysis[['operator_num']].values
    y = df_analysis['startup_time'].values
    
    model1 = LinearRegression()
    model1.fit(X1, y)
    y_pred1 = model1.predict(X1)
    
    r2_1 = r2_score(y, y_pred1)
    mae_1 = mean_absolute_error(y, y_pred1)
    
    print(f"线性模型: startup_time = {model1.intercept_:.4f} + {model1.coef_[0]:.4f} * operator_num")
    print(f"R² = {r2_1:.4f}")
    print(f"MAE = {mae_1:.4f} ms")
    
    # Model 2: Polynomial regression with operator_num
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X1_poly = poly_features.fit_transform(X1)
    model2 = LinearRegression()
    model2.fit(X1_poly, y)
    y_pred2 = model2.predict(X1_poly)
    
    r2_2 = r2_score(y, y_pred2)
    mae_2 = mean_absolute_error(y, y_pred2)
    
    print(f"\n多项式模型 (degree=2): startup_time = f(operator_num)")
    print(f"R² = {r2_2:.4f}")
    print(f"MAE = {mae_2:.4f} ms")
    
    # Model 3: Multiple linear regression with operator_num and streaming_count
    print("\n" + "=" * 80)
    print("模型2: 启动时间 = f(算子数量, 流式算子数量)")
    print("=" * 80)
    
    X3 = df_analysis[['operator_num', 'streaming_count']].values
    model3 = LinearRegression()
    model3.fit(X3, y)
    y_pred3 = model3.predict(X3)
    
    r2_3 = r2_score(y, y_pred3)
    mae_3 = mean_absolute_error(y, y_pred3)
    
    print(f"线性模型: startup_time = {model3.intercept_:.4f} + {model3.coef_[0]:.4f} * operator_num + {model3.coef_[1]:.4f} * streaming_count")
    print(f"R² = {r2_3:.4f}")
    print(f"MAE = {mae_3:.4f} ms")
    
    # Model 4: Multiple linear regression with operator_num and pipeline_count
    print("\n" + "=" * 80)
    print("模型3: 启动时间 = f(算子数量, pipeline数量)")
    print("=" * 80)
    
    X4 = df_analysis[['operator_num', 'pipeline_count']].values
    model4 = LinearRegression()
    model4.fit(X4, y)
    y_pred4 = model4.predict(X4)
    
    r2_4 = r2_score(y, y_pred4)
    mae_4 = mean_absolute_error(y, y_pred4)
    
    print(f"线性模型: startup_time = {model3.intercept_:.4f} + {model4.coef_[0]:.4f} * operator_num + {model4.coef_[1]:.4f} * pipeline_count")
    print(f"R² = {r2_4:.4f}")
    print(f"MAE = {mae_4:.4f} ms")
    
    # Select best model
    models = [
        ('Linear (operator_num)', model1, r2_1, mae_1, X1),
        ('Polynomial (operator_num)', model2, r2_2, mae_2, X1_poly),
        ('Multiple (operator_num, streaming_count)', model3, r2_3, mae_3, X3),
        ('Multiple (operator_num, pipeline_count)', model4, r2_4, mae_4, X4),
    ]
    
    best_model = max(models, key=lambda x: x[2])  # Best R²
    print("\n" + "=" * 80)
    print(f"最佳模型: {best_model[0]}")
    print(f"R² = {best_model[2]:.4f}, MAE = {best_model[3]:.4f} ms")
    print("=" * 80)
    
    # Save model parameters
    model_info = {
        'dataset': dataset,
        'best_model_name': best_model[0],
        'best_model': best_model[1],
        'r2': best_model[2],
        'mae': best_model[3],
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'model4': model4,
        'poly_features': poly_features if 'Polynomial' in best_model[0] else None
    }
    
    return model_info, df_analysis

def estimate_startup_time(operator_num, streaming_count=0, pipeline_count=0, model_info=None):
    """
    Estimate startup time based on operator number and pipeline/streaming count.
    
    Args:
        operator_num: Number of operators
        streaming_count: Number of streaming operators (optional)
        pipeline_count: Number of pipelines (optional)
        model_info: Model information from analyze_startup_time
    
    Returns:
        Estimated startup time in ms
    """
    if model_info is None:
        # Default simple linear model
        return 10.0 + 5.0 * operator_num
    
    model_name = model_info['best_model_name']
    
    if 'Linear (operator_num)' in model_name:
        model = model_info['model1']
        return model.predict([[operator_num]])[0]
    elif 'Polynomial (operator_num)' in model_name:
        model = model_info['model2']
        poly_features = model_info['poly_features']
        X = poly_features.transform([[operator_num]])
        return model.predict(X)[0]
    elif 'Multiple (operator_num, streaming_count)' in model_name:
        model = model_info['model3']
        return model.predict([[operator_num, streaming_count]])[0]
    elif 'Multiple (operator_num, pipeline_count)' in model_name:
        model = model_info['model4']
        return model.predict([[operator_num, pipeline_count]])[0]
    else:
        # Fallback to simple linear
        return 10.0 + 5.0 * operator_num

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze startup time relationships')
    parser.add_argument('--dataset', type=str, default='tpcds', choices=['tpcds', 'tpch'],
                       help='Dataset to analyze')
    
    args = parser.parse_args()
    
    model_info, df_analysis = analyze_startup_time(args.dataset)
    
    if model_info:
        print("\n✅ 分析完成!")
        print(f"\n使用示例:")
        print(f"  estimated_time = estimate_startup_time(operator_num=20, streaming_count=5)")
        print(f"  # 结果: {estimate_startup_time(20, 5, 0, model_info):.2f} ms")

