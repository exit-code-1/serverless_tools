import pandas as pd
import no_dop_train


# Example usage
train_queries = [1, 2, 3, 4, 5, 6, 9, 10, 12, 11, 13, 15, 16, 17, 20, 21, 22]
test_queries = [7, 8, 14, 18, 19]

results = no_dop_train.process_and_train(
    csv_path_pattern='/home/zhy/opengauss/data_file/tpch_*_output/plan_info.csv',
    operator="Vector Nest Loop",
    output_prefix="xgboost",
    train_queries=train_queries,
    test_queries=test_queries
)

# Output results
print("\n===== Training Times (Execution) =====")
print(results['training_times_exec'])

print("\n===== Training Times (Memory) =====")
print(results['training_times_mem'])

print("\n===== Performance (Execution) =====")
print(results['performance_exec'])

print("\n===== Performance (Memory) =====")
print(results['performance_mem'])

print("\n===== ONNX Model Paths =====")
print(results['onnx_paths_exec'])
print(results['onnx_paths_mem'])

# 先创建一个空的 DataFrame 来保存拼接的结果
merged_df = pd.DataFrame()

# 遍历所有的标签和 DataFrame
for label, comparison_df in results['comparisons_exec'].items():
    print(f"Execution Comparison for label {label}:")
    
    # 确保每个 comparison_df 都是 DataFrame 类型
    comparison_df = pd.DataFrame(comparison_df)  # 转换为 DataFrame，如果还没有
    
    # 拼接当前 DataFrame 中的列
    if merged_df.empty:
        merged_df = comparison_df  # 第一次时直接赋值
    else:
        merged_df = pd.concat([merged_df, comparison_df], axis=1)  # 横向拼接

# 打印拼接后的前50行
print(merged_df.head(50))
merged_df = pd.DataFrame()
# 遍历所有的标签和 DataFrame
for label, comparison_df in results['comparisons_mem'].items():
    print(f"Execution Comparison for label {label}:")
    
    # 确保每个 comparison_df 都是 DataFrame 类型
    comparison_df = pd.DataFrame(comparison_df)  # 转换为 DataFrame，如果还没有
    
    # 拼接当前 DataFrame 中的列
    if merged_df.empty:
        merged_df = comparison_df  # 第一次时直接赋值
    else:
        merged_df = pd.concat([merged_df, comparison_df], axis=1)  # 横向拼接
        
print(merged_df.head(50))





