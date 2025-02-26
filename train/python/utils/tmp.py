import numpy as np
import pandas as pd
import ast

# 读取 CSV 文件
df = pd.read_csv('/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/test_queries_results.csv', delimiter=',')

# 将包含数组的字符串转换为单一数值
def extract_first_element(x):
    try:
        # 如果是字符串类型且包含数组表示形式，使用 ast.literal_eval 转换
        if isinstance(x, str):
            value = ast.literal_eval(x)  # 转换为 Python 列表
        else:
            value = x
        
        # 如果是列表或 ndarray，则取第一个元素
        if isinstance(value, (list, np.ndarray)):
            return value[0]  # 返回数组的第一个元素
        return value  # 如果不是数组，则直接返回原值
        
    except Exception as e:
        # 如果无法转换，则返回原值
        print(f"Error converting {x}: {e}")
        return x

# 对相关列应用提取函数
df['Execution Time Q-error'] = df['Execution Time Q-error'].apply(extract_first_element)
df['Memory Q-error'] = df['Memory Q-error'].apply(extract_first_element)
df['Time Calculation Duration (s)'] = df['Time Calculation Duration (s)'].apply(extract_first_element)

# 转换为数值（如果提取过程中存在字符串或其他类型）
df['Execution Time Q-error'] = pd.to_numeric(df['Execution Time Q-error'], errors='coerce')
df['Memory Q-error'] = pd.to_numeric(df['Memory Q-error'], errors='coerce')
df['Time Calculation Duration (s)'] = pd.to_numeric(df['Time Calculation Duration (s)'], errors='coerce')

# 计算 Execution Time Q-error 和 Memory Q-error 的平均值
avg_time_q_error = df['Execution Time Q-error'].mean()
avg_memory_q_error = df['Memory Q-error'].mean()

# 计算推理时间的平均值
avg_exec_time = df['Time Calculation Duration (s)'].mean()
avg_mem_time = df['Memory Calculation Duration (s)'].mean()

# 打印结果
print(f"Average Execution Time Q-error: {avg_time_q_error}")
print(f"Average Memory Q-error: {avg_memory_q_error}")
print(f"Average Inference Time: {avg_exec_time + avg_mem_time} seconds")
