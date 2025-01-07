import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

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


def prepare_data(csv_path_pattern, operator, feature_columns, target_columns, train_queries, test_queries, epsilon=1e-5):
    # Load and merge data
    csv_files = glob.glob(csv_path_pattern, recursive=True)
    data_list = [pd.read_csv(file, delimiter=';', encoding='utf-8') for file in csv_files]
    data = pd.concat(data_list, ignore_index=True)

    # Filter by operator type
    operator_data = data[data['operator_type'] == operator].copy()  # Add .copy() to avoid chained assignment warnings

    # Assign train/test split based on query_id
    operator_data.loc[:, 'set'] = operator_data['query_id'].apply(
        lambda qid: 'train' if (qid % 22 in train_queries) else ('test' if (qid % 22 in test_queries) else 'exclude')
    )

    train_data = operator_data[operator_data['set'] == 'train']
    test_data = operator_data[operator_data['set'] == 'test']

    # Select features and targets
    X_train = instance_normalize(train_data[feature_columns], epsilon)
    y_train = train_data[target_columns]
    X_test = instance_normalize(test_data[feature_columns], epsilon)
    y_test = test_data[target_columns]

    return X_train, X_test, y_train, y_test
