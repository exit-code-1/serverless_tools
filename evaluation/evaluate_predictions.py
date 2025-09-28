# 文件路径: evaluation/evaluate_predictions.py

import numpy as np
import pandas as pd
import ast # 用于安全地评估字符串表达式
import os  # 用于处理路径

# ==============================================================================
# 辅助函数 (来自原 tmp.py)
# ==============================================================================

def extract_first_element(x):
    """
    将可能包含数组的字符串或对象转换为单一数值。
    如果输入是列表或Numpy数组，则返回第一个元素。
    (逻辑来自原 tmp.py)
    """
    try:
        value = x
        # 检查是否为字符串，如果是，尝试用 ast.literal_eval 解析
        # ast.literal_eval 比 eval 更安全，只能解析基本字面量
        if isinstance(x, str):
            try:
                # 尝试解析，如果失败（例如不是有效的字面量字符串），保持原样
                value = ast.literal_eval(x)
            except (ValueError, SyntaxError, TypeError):
                # print(f"DEBUG: ast.literal_eval failed for '{x}', keeping original value.")
                pass # 解析失败，value 保持为原始字符串 x

        # 检查解析后或原始值是否为列表或Numpy数组
        if isinstance(value, (list, np.ndarray)):
            # 如果是列表/数组且不为空，返回第一个元素
            return value[0] if len(value) > 0 else np.nan # 空列表/数组返回 NaN
        # 如果不是列表/数组，直接返回值 (可能是数字、字符串或其他)
        return value
    except Exception as e:
        # 捕获其他潜在错误
        print(f"Error converting value '{x}': {e}")
        return np.nan # 出错返回 NaN

# ==============================================================================
# 主要评估函数
# ==============================================================================

def generate_qerror_report(predictions_csv_path, output_stats_csv_path, target_dop=96):
    """
    读取预测结果CSV，计算Q-error的分桶统计信息，并保存报告。
    (逻辑改编自原 tmp.py)

    Args:
        predictions_csv_path (str): 包含预测和实际值的CSV文件路径。
                                   需要包含列: 'Query DOP', 'Execution Time Q-error',
                                             'Memory Q-error', 'Actual Execution Time (s)',
                                             'Actual Memory Usage (MB)'。
        output_stats_csv_path (str): 保存分桶统计结果的CSV文件路径。
        target_dop (int, optional): 只针对哪个 Query DOP 计算统计。默认为 8。
    """
    print(f"开始生成 Q-error 报告，输入文件: {predictions_csv_path}")
    print(f"目标 Query DOP: {target_dop}")

    # --- 1. 读取 CSV 文件 ---
    try:
        # 原始代码使用了逗号分隔符，我们保持一致
        df_predictions = pd.read_csv(predictions_csv_path, delimiter=';') # 注意分隔符是 ';' 还是 ','
        # 检查必需的列是否存在
        required_cols = ['Query DOP', 'Execution Time Q-error', 'Memory Q-error',
                         'Actual Execution Time (s)', 'Actual Memory Usage (MB)']
        if not all(col in df_predictions.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df_predictions.columns]
             print(f"错误: 输入CSV文件 '{predictions_csv_path}' 缺少必需列: {missing}")
             return
        print(f"成功读取 {len(df_predictions)} 行数据。")
    except FileNotFoundError:
        print(f"错误: 预测结果文件未找到: {predictions_csv_path}")
        return
    except Exception as e:
        print(f"读取预测结果CSV时出错: {e}")
        return

    # --- 2. 筛选目标 DOP ---
    df = df_predictions[df_predictions['Query DOP'] <= target_dop].copy()
    if df.empty:
        print(f"错误: 未找到 Query DOP <= {target_dop} 的数据。无法生成报告。")
        return
    print(f"筛选出 {len(df)} 行 Query DOP <= {target_dop} 的数据。")
    
    # --- 3. 数据清洗和转换 (来自原 tmp.py) ---
    print("开始数据清洗和转换...")
    # --- 额外过滤：去除实际执行时间或内存为0的行 ---
    rows_before_filter_zero = len(df)
    df = df[(df['Actual Execution Time (s)'] > 0) & (df['Actual Memory Usage (MB)'] > 0)]
    rows_after_filter_zero = len(df)
    if rows_after_filter_zero < rows_before_filter_zero:
        print(f"警告: 移除了 {rows_before_filter_zero - rows_after_filter_zero} 行实际执行时间或内存为0的数据。")

    if df.empty:
        print("错误: 所有数据的执行时间或内存为0，无法继续生成报告。")
        return

    # 对可能包含数组的列应用提取函数
    # 注意：原始 tmp.py 对 Q-error 列也用了 extract，但 Q-error 理论上应该是单一数值
    # 如果你的 prediction CSV 中 Q-error 列已经是数值，可以跳过这两行
    df['Execution Time Q-error'] = df['Execution Time Q-error'].apply(extract_first_element)
    df['Memory Q-error'] = df['Memory Q-error'].apply(extract_first_element)
    # 对时间和内存列应用提取函数 (以防万一它们也是数组形式)
    df['Actual Execution Time (s)'] = df['Actual Execution Time (s)'].apply(extract_first_element)
    df['Actual Memory Usage (MB)'] = df['Actual Memory Usage (MB)'].apply(extract_first_element)

    # 转换为数值类型，无法转换的设为 NaN
    df['Execution Time Q-error'] = pd.to_numeric(df['Execution Time Q-error'], errors='coerce')
    df['Memory Q-error'] = pd.to_numeric(df['Memory Q-error'], errors='coerce')
    df['Actual Execution Time (s)'] = pd.to_numeric(df['Actual Execution Time (s)'], errors='coerce')
    df['Actual Memory Usage (MB)'] = pd.to_numeric(df['Actual Memory Usage (MB)'], errors='coerce')

    # 删除包含 NaN 值的行，这些行无法参与统计
    rows_before_dropna = len(df)
    df.dropna(subset=['Execution Time Q-error', 'Memory Q-error',
                     'Actual Execution Time (s)', 'Actual Memory Usage (MB)'], inplace=True)
    rows_after_dropna = len(df)
    if rows_after_dropna < rows_before_dropna:
        print(f"警告: 移除了 {rows_before_dropna - rows_after_dropna} 行包含无效数值的数据。")

    if df.empty:
        print("错误: 数据清洗后没有有效数据。无法生成报告。")
        return
    print("数据清洗完成。")

    # --- 4. 计算分位数和定义分桶 ---
    print("开始计算分位数和进行分桶...")
    try:
        time_quartiles = df['Actual Execution Time (s)'].quantile([0.25, 0.5, 0.75]).values
        memory_quartiles = df['Actual Memory Usage (MB)'].quantile([0.25, 0.5, 0.75]).values

        # 执行时间分桶
        time_bins = [-np.inf] + list(time_quartiles) + [np.inf] # 使用负无穷和正无穷覆盖所有范围
        time_labels = ["Q1 (<= {:.2f}s)".format(time_quartiles[0]),
                       "Q2 ({:.2f}s - {:.2f}s]".format(time_quartiles[0], time_quartiles[1]),
                       "Q3 ({:.2f}s - {:.2f}s]".format(time_quartiles[1], time_quartiles[2]),
                       "Q4 (> {:.2f}s)".format(time_quartiles[2])]
        df['Time Bin'] = pd.cut(df['Actual Execution Time (s)'], bins=time_bins, labels=time_labels, right=True) # right=True 更符合分位数定义

        # 内存分桶
        memory_bins = [-np.inf] + list(memory_quartiles) + [np.inf]
        memory_labels = ["Q1 (<= {:.2f}MB)".format(memory_quartiles[0]),
                         "Q2 ({:.2f}MB - {:.2f}MB]".format(memory_quartiles[0], memory_quartiles[1]),
                         "Q3 ({:.2f}MB - {:.2f}MB]".format(memory_quartiles[1], memory_quartiles[2]),
                         "Q4 (> {:.2f}MB)".format(memory_quartiles[2])]
        df['Memory Bin'] = pd.cut(df['Actual Memory Usage (MB)'], bins=memory_bins, labels=memory_labels, right=True)
        print("分桶完成。")
    except Exception as e:
        print(f"计算分位数或分桶时出错: {e}")
        return

    # --- 5. 统计各分桶的平均 Q-error ---
    print("开始统计各分桶的平均 Q-error...")
    try:
        # 按分桶标签排序可能更好看
        time_q_error_stats = df.groupby('Time Bin', observed=False)['Execution Time Q-error'].mean().reset_index()
        memory_q_error_stats = df.groupby('Memory Bin', observed=False)['Memory Q-error'].mean().reset_index()
        print("统计完成。")
    except Exception as e:
        print(f"计算分组统计时出错: {e}")
        return

    # --- 6. 保存结果到 CSV ---
    print(f"开始保存 Q-error 统计结果到: {output_stats_csv_path}")
    try:
        output_dir = os.path.dirname(output_stats_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 将两个统计结果写入同一个 CSV 文件，用空行分隔
        with open(output_stats_csv_path, 'w', newline='', encoding='utf-8') as f:
             time_q_error_stats.to_csv(f, index=False, sep=';') # 使用分号分隔符
             f.write('\n') # 写入一个空行
             memory_q_error_stats.to_csv(f, index=False, sep=';') # 使用分号分隔符

        # --- 或者像原始代码那样分开写入/追加 ---
        # time_q_error_stats.to_csv(output_stats_csv_path, index=False, mode='w', header=True, sep=';')
        # memory_q_error_stats.to_csv(output_stats_csv_path, index=False, mode='a', header=True, sep=';')

        print(f"Q-error 统计结果已成功保存。")

    except Exception as e:
        print(f"保存统计结果到 CSV 时出错: {e}")

# ==============================================================================
# 脚本入口 (不应在此文件内执行)
# ==============================================================================
# if __name__ == "__main__":
#     # 这里的逻辑会被移到 scripts/run_evaluation.py
#     # 需要定义 predictions_csv, output_csv 的路径
#     # generate_qerror_report(predictions_csv, output_csv)
#     pass