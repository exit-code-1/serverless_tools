import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import torch.onnx
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
from utils.structure import operator_features

def train_models(X_train, y_train):
    """
    Train a model for either execution time or peak memory.
    Optionally, perform hyperparameter tuning using GridSearchCV.

    Parameters:
    - X_train: DataFrame, training features
    - y_train: DataFrame, training target (execution_time or peak_mem)
    - perform_grid_search: bool, whether to perform grid search for hyperparameter tuning

    Returns:
    - model: trained model
    - training_time: training time for the model
    """

    # Define the XGBoost model with some default hyperparameters
    model = xgb.XGBRegressor(random_state=32, 
                             n_estimators=200,   # Default number of trees
                             learning_rate=0.05,  # Default learning rate
                             max_depth=10,        # Default max depth
                             subsample=0.8,      # Default subsample rate
                             colsample_bytree=0.8,  # Default column sample rate
                             gamma=0.1)         # Default gamma)       # Default L2 regularization

    # Train the model without hyperparameter tuning
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    return model, training_time


def predict_and_evaluate(model, X_test, y_test, epsilon=1e-2, features=None, target_column='execution_time', operator=None, suffix=""):
    """
    Predict and evaluate model performance based on the target (execution time or memory).
    Save the ONNX model during the evaluation step and compare prediction times between Python and ONNX models.

    Parameters:
    - model: the trained model (XGBRegressor)
    - X_test: DataFrame, test features
    - y_test: DataFrame, test targets
    - epsilon: small constant to avoid division by zero in Q-error calculation
    - features: list of features to be used for prediction
    - target_column: str, the target column (either 'execution_time' or 'peak_mem')
    - operator: str, the operator type (used for naming the saved ONNX file)
    - suffix: str, suffix for ONNX model filename

    Returns:
    - performance: dict, containing Q-error statistics and additional metrics
    - comparisons: DataFrame, containing actual vs predicted comparisons
    """
    # Select relevant features for prediction
    X_pred = X_test[features]

    # Python native model prediction
    start_time = time.time()
    predictions_native = model.predict(X_pred)
    native_time = time.time() - start_time

    # Save model as ONNX
    onnx_path = None
    if operator :
        onnx_model = convert_xgboost(model, initial_types=[("float_input", FloatTensorType([None, len(features)]))])
        onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/no_dop/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
        onnx_dir = os.path.dirname(onnx_path)
        os.makedirs(onnx_dir, exist_ok=True)
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to: {onnx_path}")

    # ONNX model prediction
    predictions_onnx = None
    onnx_time = None
    if onnx_path:
        # Load ONNX model and perform inference
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        start_time = time.time()
        predictions_onnx = session.run(None, {input_name: X_pred.to_numpy().astype(np.float32)})[0]
        onnx_time = time.time() - start_time

    # Calculate standard mean absolute error (MAE) for native predictions
    mae_native = np.mean(np.abs(y_test[target_column] - predictions_native))

    # Calculate Prediction Accuracy for native model
    Q_error = np.mean(
         np.abs((y_test[target_column] - predictions_native) / (y_test[target_column] + epsilon))
    )

    # Create comparison DataFrame
    comparisons = pd.DataFrame({
        'query_id': y_test['query_id'].values,
        'Actual': y_test[target_column].values,
        'Predicted_Native': predictions_native,
        'Difference_Native': y_test[target_column].values - predictions_native,
    })

    # Calculate Prediction Accuracy for ONNX model
    time_accuracy_onnx = None
    if onnx_time is not None:
        time_accuracy_onnx = np.mean(
            (1 - np.abs(y_test[target_column] - predictions_native) / (y_test[target_column] + epsilon)) * 100
        )

    # Calculate the average of the actual target values
    avg_actual_value = np.mean(y_test[target_column])

    # Print model prediction times
    print(f"Native model prediction time: {native_time:.6f} seconds")
    if onnx_time is not None:
        print(f"ONNX model prediction time: {onnx_time:.6f} seconds")

    # Organize the performance metrics into one dictionary
    performance = {
        "metrics": {
            "MAE_error": mae_native,
            "Q_error": Q_error,
            "average_actual_value": avg_actual_value
        },
        "comparisons": comparisons,
        "native_time": native_time,
        "onnx_time": onnx_time,
        "onnx_accuracy": time_accuracy_onnx
    }

    return performance

def train_CStoreScan(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }
 
def train_CStoreIndexScan(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }   
    
def train_CTEScan(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    } 
    
def train_VectorMaterialize(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }  
       
def train_VectorNestLoop(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }     
    
def train_VectorAggregate(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }
    
def train_VectorSort(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }  
    
def train_VectorHashAggregate(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }  
    
def train_VectorSonicHashAggregate(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }
    
def train_VectorHashJoin(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }       
    
def train_VectorSonicHashJoin(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }   
    
def train_VectorStreaming(X_train, X_test, y_train, y_test, operator, epsilon=1e-2):
     # Separate target variables for execution time and memory
    y_train_exec = y_train['execution_time']
    y_train_mem = y_train['peak_mem']

    # Define feature sets for execution time and memory (can be customized based on your domain knowledge)
    features_exec = operator_features[operator]['exec']  # Example features for execution time
    features_mem = operator_features[operator]['mem']  # Example features for memory prediction

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
    models_exec, training_times_exec = train_models(X_train_exec, y_train_exec)
    models_mem, training_times_mem = train_models(X_train_mem, y_train_mem)

    # Define features after renaming
    features_exec = [f"f{i}" for i in range(len(features_exec))]
    features_mem = [f"f{i}" for i in range(len(features_mem))]

    # Predict and evaluate execution time models
    results_exec = predict_and_evaluate(
        model=models_exec,
        X_test=X_test_exec,
        y_test=y_test,
        epsilon=epsilon,
        features=features_exec,
        target_column='execution_time',
        operator=operator,
        suffix="exec"
    )

    # Predict and evaluate memory models
    results_mem = predict_and_evaluate(
        model=models_mem,
        X_test=X_test_mem,
        y_test=y_test,
        epsilon=epsilon,
        features=features_mem,
        target_column='peak_mem',
        operator=operator,
        suffix="mem"
    )

    # Combine results
    return {
        "models_exec": models_exec,
        "performance_exec": results_exec["metrics"],
        "training_times_exec": training_times_exec,
        "comparisons_exec": results_exec["comparisons"],
        "native_time_exec": results_exec["native_time"],
        "onnx_time_exec": results_exec["onnx_time"],

        "models_mem": models_mem,
        "performance_mem": results_mem["metrics"],
        "training_times_mem": training_times_mem,
        "comparisons_mem": results_mem["comparisons"],
        "native_time_mem": results_mem["native_time"],
        "onnx_time_mem": results_mem["onnx_time"]
    }          