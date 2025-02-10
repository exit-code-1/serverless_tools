import glob
import os
import sys
import pandas as pd
import no_dop_train
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
# Load and merge data
csv_path_pattern='/home/zhy/opengauss/data_file/tpch_*_output/plan_info.csv'
csv_files = glob.glob(csv_path_pattern, recursive=True)
data_list = [pd.read_csv(file, delimiter=';', encoding='utf-8') for file in csv_files]
data = pd.concat(data_list, ignore_index=True)
no_dop_train.train_all_operators(data)


# 设置最大显示行数和列数
# pd.set_option('display.max_rows', 200)  # 或者设定为一个较大的数字，如1000
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)  # 自动调整宽度
# pd.set_option('display.max_colwidth', None)  # 防止列宽被截断
# # Output results
# print("\n===== Training Times (Execution) =====")
# print(results['training_times_exec'])

# print("\n===== Training Times (Memory) =====")
# print(results['training_times_mem'])

# print("\n===== Performance (Execution) =====")
# print(results['performance_exec'])

# print("\n===== Performance (Memory) =====")
# print(results['performance_mem'])

# # Function to merge comparison DataFrames
# def merge_comparisons(comparisons):
#     """
#     Merge all comparison DataFrames into a single DataFrame.

#     Parameters:
#     - comparisons: dict, where keys are labels and values are DataFrames or dicts

#     Returns:
#     - pd.DataFrame, merged DataFrame
#     """
#     merged_df = pd.DataFrame()

#     for label, comparison_df in comparisons.items():
#         print(f"Comparison for label {label}:")
        
#         # Ensure comparison_df is a DataFrame
#         if not isinstance(comparison_df, pd.DataFrame):
#             comparison_df = pd.DataFrame(comparison_df)
        
#         # Add label as prefix to column names for distinction
#         comparison_df = comparison_df.add_prefix(f"{label}_")
        
#         # Merge DataFrames
#         if merged_df.empty:
#             merged_df = comparison_df
#         else:
#             merged_df = pd.concat([merged_df, comparison_df], axis=1)

#     return merged_df

# # Merge and print execution comparisons
# merged_exec_df = merge_comparisons(results['comparisons_exec'])
# print("\n===== Merged Execution Comparisons =====")
# print(merged_exec_df.head(200))

