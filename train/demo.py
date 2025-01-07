import glob
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from onnxmltools.convert import convert_xgboost
import onnx
import onnxruntime as ort

# 定义实例归一化函数
def instance_normalize(data, epsilon=1e-5):
    """
    Perform instance normalization on the data.
    Parameters:
    - data: pd.DataFrame of shape (N, C)
    - epsilon: a small constant to avoid division by zero
    Returns:
    - normalized_data: pd.DataFrame of shape (N, C)
    """
    num_instances, num_channels = data.shape
    normalized_data = data.copy()
    for i in range(num_instances):
        instance = data.iloc[i]
        mean = np.mean(instance)
        variance = np.var(instance)
        normalized_data.iloc[i] = (instance - mean) / np.sqrt(variance + epsilon)
    return normalized_data


# 获取所有子文件夹中的 plan_info.csv 文件路径
csv_files = glob.glob('/home/zhy/opengauss/data_file/tpch_*_output/plan_info.csv', recursive=True)

# 读取并合并所有 plan_info.csv 文件
data_list = []
for file in csv_files:
    # 读取每个文件，并将其添加到列表
    df = pd.read_csv(file, delimiter=';', encoding='utf-8')
    data_list.append(df)

# 合并所有文件的数据
data = pd.concat(data_list, ignore_index=True)

# 指定算子类型
operator = "CStore Index Scan"

# 筛选当前算子类型的数据
operator_data = data[data['operator_type'] == operator]

# 选择特征和目标变量
X = operator_data[['l_input_rows', 'actural_rows', 'instance_mem', 'width']]
y = operator_data[['execution_time', 'peak_mem']]

X_standardized = instance_normalize(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# 重命名特征为 f0, f1, f2, ...
X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]


# 训练第一个模型，预测 execution_time
y_exec = y_train[['execution_time']]  # 只用 execution_time
print(f"X_train shape: {X_train.shape}")
print(f"y_exec shape: {y_exec.shape}")

model_exec = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, objective='reg:squarederror', random_state=42)
start_time = time.time()
model_exec.fit(X_train, y_exec)
training_time_exec = time.time() - start_time

# 训练第二个模型，预测 peak_mem
y_mem = y_train[['peak_mem']]  # 只用 peak_mem
model_mem = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, objective='reg:squarederror', random_state=42)
start_time = time.time()
model_mem.fit(X_train, y_mem)
training_time_mem = time.time() - start_time


# 保存转换后的 ONNX 模型
onnx_exec_model = convert_xgboost(model_exec, initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))])
onnx_mem_model = convert_xgboost(model_mem, initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))])

onnx_exec_model_path = f"xgboost_exec_{operator.replace(' ', '_')}.onnx"
onnx_mem_model_path = f"xgboost_mem_{operator.replace(' ', '_')}.onnx"

with open(onnx_exec_model_path, 'wb') as f:
    f.write(onnx_exec_model.SerializeToString())

with open(onnx_mem_model_path, 'wb') as f:
    f.write(onnx_mem_model.SerializeToString())

# 打印信息
print(f"Execution time model saved as ONNX format at: {onnx_exec_model_path}")
print(f"Peak memory model saved as ONNX format at: {onnx_mem_model_path}")
# 记录推理开始时间
start_time = time.time()

# 预测执行时间
pred_exec_time = model_exec.predict(X_test)

# 预测峰值内存
pred_peak_mem = model_mem.predict(X_test)

# 记录推理结束时间
end_time = time.time()

# 计算推理时间
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

# 防止除零错误
epsilon = 1e-2
# 计算加权 Q-error：权重是根据 execution_time 的大小来计算的
weights = y_test['execution_time'] / y_test['execution_time'].sum()  # 使用相对权重

# 计算加权 Q-error for Execution Time
Q_error_time_weighted = (abs(y_test['execution_time'] - pred_exec_time) / (y_test['execution_time'] + epsilon)) * weights

# 计算加权 Q-error for Peak Memory
Q_error_mem_weighted = (abs(y_test['peak_mem'] - pred_peak_mem) / (y_test['peak_mem'] + epsilon)) * weights

# 输出加权 Q-error 统计信息
print("\n===== Weighted Q-error for Execution Time =====")
print(Q_error_time_weighted.describe())

print("\n===== Weighted Q-error for Peak Memory =====")
print(Q_error_mem_weighted.describe())

print("\n===== Execution Time (Raw) =====")
print(y_test['execution_time'].describe())

# 打印前50个预测值与实际值对比
print("\n===== Predicted vs Actual Execution Time =====")
comparison_exec_time = pd.DataFrame({
    'Actual Execution Time': y_test['execution_time'][:50],
    'Predicted Execution Time': pred_exec_time[:50]
})

print(comparison_exec_time)

# 打印前50个预测值与实际值对比的差异
comparison_exec_time['Difference'] = comparison_exec_time['Actual Execution Time'] - comparison_exec_time['Predicted Execution Time']
print("\n===== Difference Between Actual and Predicted Execution Time =====")
print(comparison_exec_time['Difference'])
