import pandas as pd
import no_dop_train

# Example usage
train_queries = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 11, 13, 15, 16, 17, 20, 21]
test_queries = [8, 14, 18, 19, 22]

results = no_dop_train.process_and_train(
    csv_path_pattern='/home/zhy/opengauss/data_file/tpch_*_output/plan_info.csv',
    operator="CStore Index Scan",
    output_prefix="xgboost",
    train_queries=train_queries,
    test_queries=test_queries
)

# 设置最大显示行数和列数
pd.set_option('display.max_rows', 200)  # 或者设定为一个较大的数字，如1000
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # 自动调整宽度
pd.set_option('display.max_colwidth', None)  # 防止列宽被截断
# Output results
print("\n===== Training Times (Execution) =====")
print(results['training_times_exec'])

print("\n===== Training Times (Memory) =====")
print(results['training_times_mem'])

print("\n===== Performance (Execution) =====")
print(results['performance_exec'])

print("\n===== Performance (Memory) =====")
print(results['performance_mem'])

# Function to merge comparison DataFrames
def merge_comparisons(comparisons):
    """
    Merge all comparison DataFrames into a single DataFrame.

    Parameters:
    - comparisons: dict, where keys are labels and values are DataFrames or dicts

    Returns:
    - pd.DataFrame, merged DataFrame
    """
    merged_df = pd.DataFrame()

    for label, comparison_df in comparisons.items():
        print(f"Comparison for label {label}:")
        
        # Ensure comparison_df is a DataFrame
        if not isinstance(comparison_df, pd.DataFrame):
            comparison_df = pd.DataFrame(comparison_df)
        
        # Add label as prefix to column names for distinction
        comparison_df = comparison_df.add_prefix(f"{label}_")
        
        # Merge DataFrames
        if merged_df.empty:
            merged_df = comparison_df
        else:
            merged_df = pd.concat([merged_df, comparison_df], axis=1)

    return merged_df

# Merge and print execution comparisons
merged_exec_df = merge_comparisons(results['comparisons_exec'])
print("\n===== Merged Execution Comparisons =====")
print(merged_exec_df.head(200))

# # Merge and print memory comparisons
# merged_mem_df = merge_comparisons(results['comparisons_mem'])
# print("\n===== Merged Memory Comparisons =====")
# print(merged_mem_df.head(100))
