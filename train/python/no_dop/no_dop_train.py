
import os
import time
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from onnxmltools.convert import convert_xgboost
from sklearn.svm import SVR
from skl2onnx import convert_sklearn
import onnxruntime as ort
import sys

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils

def train_models(X_train, y_train, operator, output_prefix, suffix="", perform_grid_search=True):
    """
    Train a model for either execution time or peak memory, and save it as an ONNX model.
    Optionally, perform hyperparameter tuning using GridSearchCV.

    Parameters:
    - X_train: DataFrame, training features
    - y_train: DataFrame, training target (execution_time or peak_mem)
    - operator: str, the operator type to filter data
    - output_prefix: str, prefix for saving ONNX models
    - suffix: str, 'exec' for execution time or 'mem' for peak memory
    - perform_grid_search: bool, whether to perform grid search for hyperparameter tuning

    Returns:
    - model: trained model
    - onnx_path: path to the saved ONNX model
    - training_time: training time for the model
    """

    # Define the XGBoost model
    model = xgb.XGBRegressor(random_state=42)

    # Perform hyperparameter tuning if specified
    if perform_grid_search:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 10, 100]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        
        # Train and tune the model
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Get the best model
        model = grid_search.best_estimator_
        print("Best parameters found: ", grid_search.best_params_)

    else:
        # Train the model without hyperparameter tuning
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

    # Save model as ONNX
    onnx_model = convert_xgboost(model, initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))])
    onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/no_dop/{operator}/{output_prefix}_{suffix}_{operator.replace(' ', '_')}.onnx"
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    return model, onnx_path, training_time



def predict_and_evaluate(model, X_test, y_test, epsilon=1e-2, features=None, target_column='execution_time'):
    """
    Predict and evaluate model performance based on the target (execution time or memory).

    Parameters:
    - model: the trained model (XGBRegressor)
    - X_test: DataFrame, test features
    - y_test: DataFrame, test targets
    - epsilon: small constant to avoid division by zero in Q-error calculation
    - features: list of features to be used for prediction
    - target_column: str, the target column (either 'execution_time' or 'peak_mem')

    Returns:
    - performance: dict, containing Q-error statistics
    - comparisons: DataFrame, containing actual vs predicted comparisons
    """
    # Select relevant features for prediction
    X_pred = X_test[features]

    # Predictions
    predictions = model.predict(X_pred)

    # Weighted Q-error calculation
    weights = y_test[target_column] / y_test[target_column].sum()
    
    # Calculate Q-error based on the target_column (execution_time or peak_mem)
    q_error_weighted = (abs(y_test[target_column] - predictions) / (y_test[target_column] + epsilon)) * weights

    # Describe performance
    performance = q_error_weighted.describe()

    # Create comparison DataFrame
    comparisons = pd.DataFrame({
        'query_id': y_test['query_id'].values,
        'Actual': y_test[target_column].values,
        'Predicted': predictions,
        'Difference': y_test[target_column].values - predictions
    })

    return performance, comparisons



def process_and_train(csv_path_pattern, operator, output_prefix, train_queries, test_queries, epsilon=1e-2):
    """
    Process data, train models, and evaluate performance. Execution time and memory are handled separately,
    with distinct feature sets for each task.

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
    # Prepare data for both execution time and memory prediction
    X_train, X_test, y_train, y_test = utils.prepare_data(
        csv_path_pattern=csv_path_pattern,
        operator=operator,
        feature_columns=['l_input_rows', 'r_input_rows', 'actural_rows', 'instance_mem', 'width'],  # General features
        target_columns=['query_id', 'execution_time', 'peak_mem'],
        train_queries=train_queries,
        test_queries=test_queries,
        epsilon=epsilon
    )

    # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = ['l_input_rows', 'r_input_rows', 'actural_rows', 'width']  # Example features for execution time
    features_mem = ['l_input_rows', 'r_input_rows', 'actural_rows', 'width']  # Example features for memory prediction

    # Prepare training data based on the specific features for execution time and memory
    X_train_exec = X_train[features_exec]
    X_test_exec = X_test[features_exec]
    X_train_mem = X_train[features_mem]
    X_test_mem = X_test[features_mem]

    # Rename features for compatibility with ONNX
    X_train_exec.columns = [f"f{i}" for i in range(X_train_exec.shape[1])]
    X_test_exec.columns = [f"f{i}" for i in range(X_test_exec.shape[1])]
    X_train_mem.columns = [f"f{i}" for i in range(X_train_mem.shape[1])]
    X_test_mem.columns = [f"f{i}" for i in range(X_test_mem.shape[1])]

    # Train models for execution time and memory separately
    models_exec, onnx_paths_exec, training_times_exec = train_models(X_train_exec, y_train_exec, operator, output_prefix, suffix="exec")
    models_mem, onnx_paths_mem, training_times_mem = train_models(X_train_mem, y_train_mem, operator, output_prefix, suffix="mem")
    features_exec = [f"f{i}" for i in range(len(['l_input_rows', 'r_input_rows', 'actural_rows', 'width']))]
    features_mem = [f"f{i}" for i in range(len(['l_input_rows', 'r_input_rows', 'actural_rows', 'width']))]
    # Predict and evaluate execution time models
    performance_exec, comparisons_exec = predict_and_evaluate(models_exec, X_test_exec, y_test, epsilon=1e-2, features=features_exec, target_column='execution_time')

    # Predict and evaluate memory models
    performance_mem, comparisons_mem = predict_and_evaluate(models_mem, X_test_mem, y_test, epsilon=1e-2, features=features_mem, target_column='peak_mem')

    return {
        "models_exec": models_exec,
        "onnx_paths_exec": onnx_paths_exec,
        "performance_exec": performance_exec,
        "training_times_exec": training_times_exec,
        "comparisons_exec": comparisons_exec,
        "models_mem": models_mem,
        "onnx_paths_mem": onnx_paths_mem,
        "performance_mem": performance_mem,
        "training_times_mem": training_times_mem,
        "comparisons_mem": comparisons_mem
    }