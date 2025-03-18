import numpy as np
import pandas as pd
import ast

# 读取 CSV 文件
df_plans = pd.read_csv('/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/test_queries_results.csv', delimiter=',')
df = df_plans[df_plans['Query DOP'] == 8].copy()

# 将包含数组的字符串转换为单一数值
def extract_first_element(x):
    try:
        if isinstance(x, str):
            value = ast.literal_eval(x)  # 转换为 Python 列表
        else:
            value = x
        
        if isinstance(value, (list, np.ndarray)):
            return value[0]  # 返回数组的第一个元素
        return value  # 如果不是数组，则直接返回原值
    except Exception as e:
        print(f"Error converting {x}: {e}")
        return np.nan

# 对相关列应用提取函数
df['Execution Time Q-error'] = df['Execution Time Q-error'].apply(extract_first_element)
df['Memory Q-error'] = df['Memory Q-error'].apply(extract_first_element)
df['Actual Execution Time (s)'] = df['Actual Execution Time (s)'].apply(extract_first_element)
df['Actual Memory Usage (MB)'] = df['Actual Memory Usage (MB)'].apply(extract_first_element)

# 转换为数值
df['Execution Time Q-error'] = pd.to_numeric(df['Execution Time Q-error'], errors='coerce')
df['Memory Q-error'] = pd.to_numeric(df['Memory Q-error'], errors='coerce')
df['Actual Execution Time (s)'] = pd.to_numeric(df['Actual Execution Time (s)'], errors='coerce')
df['Actual Memory Usage (MB)'] = pd.to_numeric(df['Actual Memory Usage (MB)'], errors='coerce')

# 计算执行时间的四分位数
time_quartiles = df['Actual Execution Time (s)'].quantile([0.25, 0.5, 0.75]).values
memory_quartiles = df['Actual Memory Usage (MB)'].quantile([0.25, 0.5, 0.75]).values

# 定义区间
time_bins = [0] + list(time_quartiles) + [float('inf')]
time_labels = ["≤{:.2f}s".format(time_bins[1]),
               "{:.2f}-{:.2f}s".format(time_bins[1], time_bins[2]),
               "{:.2f}-{:.2f}s".format(time_bins[2], time_bins[3]),
               ">{:.2f}s".format(time_bins[3])]
df['Time Bin'] = pd.cut(df['Actual Execution Time (s)'], bins=time_bins, labels=time_labels, right=False)

# 定义内存区间（单位 MB）
memory_bins = [0] + list(memory_quartiles) + [float('inf')]
memory_labels = ["≤{:.2f}MB".format(memory_bins[1]),
                  "{:.2f}-{:.2f}MB".format(memory_bins[1], memory_bins[2]),
                  "{:.2f}-{:.2f}MB".format(memory_bins[2], memory_bins[3]),
                  ">{:.2f}MB".format(memory_bins[3])]
df['Memory Bin'] = pd.cut(df['Actual Memory Usage (MB)'], bins=memory_bins, labels=memory_labels, right=False)

# 统计不同区间的平均 Q-error
time_q_error_stats = df.groupby('Time Bin')['Execution Time Q-error'].mean().reset_index()
memory_q_error_stats = df.groupby('Memory Bin')['Memory Q-error'].mean().reset_index()

# 保存到 CSV
output_path = "/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/q_error_stats.csv"
time_q_error_stats.to_csv(output_path, index=False, mode='w', header=True)
memory_q_error_stats.to_csv(output_path, index=False, mode='a', header=True)

print(f"Q-error statistics saved to {output_path}")

