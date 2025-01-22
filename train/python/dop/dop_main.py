import dop_model

# 示例训练和测试查询
train_queries = [1, 2, 3, 4, 5, 6, 9, 10, 12, 11, 13, 15, 16, 17, 20, 21, 22]
test_queries = [7, 8, 14, 18, 19]

# 数据路径和操作符名称
csv_path_pattern = '/home/zhy/opengauss/data_file/tpch_*_output/plan_info.csv'
operator = "CStore Scan"
output_prefix = "curve_fit"

# 调用 process_and_train_curve 进行处理和训练
results = dop_model.process_and_train_curve(
    csv_path_pattern=csv_path_pattern,
    operator=operator,
    output_prefix=output_prefix,
    train_queries=train_queries,
    test_queries=test_queries,
    epochs=1000,
    lr=0.05
)

# 输出结果
print("\n===== Execution Time Model Results =====")
exec_model = results['execution_time_model']
print(f"Native Prediction Time: {exec_model['native_time']:.6f} seconds")
if exec_model['onnx_time'] is not None:
    print(f"ONNX Prediction Time: {exec_model['onnx_time']:.6f} seconds")
print(f"Native MSE: {exec_model['test_mse']:.6f}")

print("\n===== Memory Model Results =====")
mem_model = results['memory_model']
print(f"Native Prediction Time: {mem_model['native_time']:.6f} seconds")
if mem_model['onnx_time'] is not None:
    print(f"ONNX Prediction Time: {mem_model['onnx_time']:.6f} seconds")
print(f"Native MSE: {mem_model['test_mse']:.6f}")

# 比较预测参数
def merge_comparisons(pred_params, suffix):
    """
    将预测参数整合为一个 DataFrame。

    Parameters:
    - pred_params: Tensor, 预测的参数 (a, b, c)
    - suffix: str, 数据后缀

    Returns:
    - DataFrame, 包含预测参数和后缀的表
    """
    import pandas as pd
    df = pd.DataFrame(pred_params.numpy(), columns=["a", "b", "c"])
    return df.add_prefix(f"{suffix}_")

print("\n===== Predicted Parameters (Execution Time) =====")
native_exec_params_df = merge_comparisons(exec_model['predicted_params'], "native_exec")
print(native_exec_params_df.head())

print("\n===== Predicted Parameters (Memory) =====")
native_mem_params_df = merge_comparisons(mem_model['predicted_params'], "native_mem")
print(native_mem_params_df.head())

