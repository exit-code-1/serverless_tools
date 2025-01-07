
import time
import pandas as pd
import utils
import xgboost as xgb
import pickle
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from onnxmltools.convert import convert_xgboost
import onnx
import onnxruntime as ort
import warnings





def train_models(X_train, y_train, operator, output_prefix):
    """
    Train models for execution time and peak memory, and save them as ONNX models.

    Parameters:
    - X_train: DataFrame, training features
    - y_train: DataFrame, training targets
    - operator: str, the operator type to filter data
    - output_prefix: str, prefix for saving ONNX models

    Returns:
    - models: dict, trained models for execution time and peak memory
    - onnx_paths: dict, paths to saved ONNX models
    - training_times: dict, training times for each model
    """
    models = {}
    training_times = {}
    onnx_paths = {}

    for target, label in zip(['execution_time', 'peak_mem'], ['exec', 'mem']):
        y_target = y_train[[target]]
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, objective='reg:squarederror', random_state=42)
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_target)
        training_times[label] = time.time() - start_time
        models[label] = model

        # Save model as ONNX
        onnx_model = convert_xgboost(model, initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))])
        onnx_path = f"../model/no_dop/{operator}/{output_prefix}_{label}_{operator.replace(' ', '_')}.onnx"
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        onnx_paths[label] = onnx_path

    return models, onnx_paths, training_times


def predict_and_evaluate(models, X_test, y_test, epsilon=1e-2):
    """
    Predict and evaluate model performance.

    Parameters:
    - models: dict, trained models for execution time and peak memory
    - X_test: DataFrame, test features
    - y_test: DataFrame, test targets
    - epsilon: float, a small constant to avoid division by zero in Q-error calculations

    Returns:
    - performance: dict, containing weighted Q-error statistics
    - comparisons: dict, containing actual vs predicted comparisons
    """
    predictions = {label: model.predict(X_test) for label, model in models.items()}
    weights = y_test['execution_time'] / y_test['execution_time'].sum()
    performance = {}
    comparisons = {}

    for target, label in zip(['execution_time', 'peak_mem'], ['exec', 'mem']):
        q_error_weighted = (abs(y_test[target] - predictions[label]) / (y_test[target] + epsilon)) * weights
        performance[label] = q_error_weighted.describe()

        # Create comparison DataFrame
        comparisons[label] = pd.DataFrame({
            'query_id': y_test['query_id'].values,
            'Actual': y_test[target].values,
            'Predicted': predictions[label],
            'Difference': y_test[target].values - predictions[label]
        })

    return performance, comparisons


# Main function for processing and running the pipeline
def process_and_train(csv_path_pattern, operator, output_prefix, train_queries, test_queries, epsilon=1e-2):
    """
    Process data, train models, and evaluate performance.

    Parameters:
    - csv_path_pattern: str, glob pattern to find plan_info.csv files
    - operator: str, the operator type to filter data
    - output_prefix: str, prefix for saving ONNX models
    - train_queries: list of int, query IDs to use for training
    - test_queries: list of int, query IDs to use for testing
    - epsilon: float, a small constant to avoid division by zero in Q-error calculations

    Returns:
    - results: dict, containing models, ONNX paths, performance, and comparisons
    """
    # Prepare data
    X_train, X_test, y_train, y_test = utils.prepare_data(
        csv_path_pattern=csv_path_pattern,
        operator=operator,
        feature_columns=['l_input_rows', 'actural_rows', 'instance_mem', 'width'],
        target_columns=['query_id', 'execution_time', 'peak_mem'],
        train_queries=train_queries,
        test_queries=test_queries,
        epsilon=epsilon
    )

    # Rename features for compatibility with ONNX
    X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

    # Train models
    models, onnx_paths, training_times = train_models(X_train, y_train, operator, output_prefix)

    # Predict and evaluate
    performance, comparisons = predict_and_evaluate(models, X_test, y_test, epsilon)

    return {
        "models": models,
        "onnx_paths": onnx_paths,
        "performance": performance,
        "training_times": training_times,
        "comparisons": comparisons
    }


# Example usage
train_queries = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 16, 17, 20, 21, 22]
test_queries = [8, 10, 12, 14, 18, 19]

results = process_and_train(
    csv_path_pattern='/home/zhy/opengauss/data_file/tpch_*_output/plan_info.csv',
    operator="CStore Index Scan",
    output_prefix="xgboost",
    train_queries=train_queries,
    test_queries=test_queries
)

# Output results
print("\n===== Training Times =====")
print(results['training_times'])

print("\n===== Performance =====")
print(results['performance'])

print("\n===== ONNX Model Paths =====")
print(results['onnx_paths'])

print("\n===== Predicted vs Actual =====")
for label, comparison_df in results['comparisons'].items():
    print(f"\nComparison for {label}:")
    print(comparison_df.head(50))
