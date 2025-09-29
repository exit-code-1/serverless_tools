# -*- coding: utf-8 -*-
"""
公共工具模块
包含所有脚本的公共函数和工具
"""

import os
import sys
import pandas as pd
import time
from typing import Dict, Any, Optional

# ==================== 导入配置 ====================
from config import (
    PROJECT_ROOT, DATASETS, METHODS, TRAIN_MODES, 
    DEFAULT_CONFIG, validate_dataset, validate_method, validate_train_mode
)

# ==================== 环境设置 ====================
def setup_environment():
    """设置项目环境"""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    print(f"项目根目录: {PROJECT_ROOT}")

def setup_config_structure():
    """设置配置结构"""
    from config import structure
    structure.USE_HASH_TABLE_SIZE_FEATURE = False
    print("配置结构已设置")

# ==================== 路径工具 ====================
def ensure_dir_exists(dir_path: str) -> None:
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)
    print(f"目录已创建: {dir_path}")

def check_file_exists(file_path: str, description: str = "文件") -> bool:
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        print(f"错误: {description}未找到: {file_path}")
        return False
    return True

def get_experiment_name(train_mode: str, eval_mode: str) -> str:
    """生成实验名称"""
    return f"{train_mode}_{eval_mode}"

# ==================== 数据加载工具 ====================
def load_csv_safe(file_path: str, delimiter: str = ';', description: str = "CSV文件") -> Optional[pd.DataFrame]:
    """安全加载CSV文件"""
    try:
        if not check_file_exists(file_path, description):
            return None
        
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
        print(f"成功加载{description}: {file_path} ({len(df)} 行)")
        return df
    except Exception as e:
        print(f"加载{description}时出错: {e}")
        return None

def save_csv_safe(df: pd.DataFrame, file_path: str, delimiter: str = ';', description: str = "CSV文件") -> bool:
    """安全保存CSV文件"""
    try:
        ensure_dir_exists(os.path.dirname(file_path))
        df.to_csv(file_path, sep=delimiter, index=False)
        print(f"成功保存{description}: {file_path}")
        return True
    except Exception as e:
        print(f"保存{description}时出错: {e}")
        return False

# ==================== 模型管理工具 ====================
def get_model_paths(train_mode: str, method: str) -> Dict[str, str]:
    """获取模型路径"""
    from config import get_model_dir
    
    model_dir = get_model_dir(train_mode)
    
    if method in ['dop_aware', 'non_dop_aware']:
        model_type = f"operator_{method}"
        return {
            'model_dir': os.path.join(model_dir, model_type),
            'exec_model': os.path.join(model_dir, model_type, "execution_time_model.onnx"),
            'mem_model': os.path.join(model_dir, model_type, "memory_usage_model.onnx")
        }
    elif method == 'query_level':
        return {
            'model_dir': os.path.join(model_dir, "query_level"),
            'exec_model': os.path.join(model_dir, "query_level", "execution_time_model.onnx"),
            'mem_model': os.path.join(model_dir, "query_level", "memory_usage_model.onnx")
        }
    elif method == 'ppm':
        return {
            'model_dir': os.path.join(model_dir, "PPM"),
            'exec_model': None,  # PPM有自己的模型结构
            'mem_model': None
        }
    else:
        raise ValueError(f"Unknown method: {method}")

# ==================== 计时工具 ====================
class Timer:
    """计时器类"""
    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"开始 {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} 完成，耗时: {duration:.4f} 秒")
    
    def get_duration(self) -> float:
        """获取持续时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

# ==================== 日志工具 ====================
def log_experiment_start(dataset: str, method: str, train_mode: str, eval_mode: str = None):
    """记录实验开始"""
    print("=" * 60)
    print(f"开始实验: {method.upper()} 方法")
    print(f"数据集: {dataset}")
    print(f"训练模式: {train_mode}")
    if eval_mode:
        print(f"评估模式: {eval_mode}")
    print("=" * 60)

def log_experiment_end(dataset: str, method: str, output_path: str = None):
    """记录实验结束"""
    print("=" * 60)
    print(f"实验完成: {method.upper()} 方法")
    print(f"数据集: {dataset}")
    if output_path:
        print(f"输出路径: {output_path}")
    print("=" * 60)

# ==================== 错误处理工具 ====================
def safe_import(module_name: str, function_name: str = None):
    """安全导入模块"""
    try:
        module = __import__(module_name, fromlist=[function_name] if function_name else None)
        if function_name:
            return getattr(module, function_name)
        return module
    except ImportError as e:
        print(f"导入错误: {e}")
        print(f"请检查模块 {module_name} 是否存在")
        return None
    except AttributeError as e:
        print(f"属性错误: {e}")
        print(f"请检查模块 {module_name} 中是否有函数 {function_name}")
        return None

# ==================== 配置验证工具 ====================
def validate_experiment_config(dataset: str, method: str, train_mode: str) -> bool:
    """验证实验配置"""
    try:
        validate_dataset(dataset)
        validate_method(method)
        validate_train_mode(train_mode)
        return True
    except ValueError as e:
        print(f"配置验证失败: {e}")
        return False

# ==================== 输出路径工具 ====================
def get_output_paths(dataset: str, method: str, train_mode: str, eval_mode: str = None) -> Dict[str, str]:
    """获取输出路径"""
    from config import get_output_dir
    
    output_dir = get_output_dir(dataset)
    experiment_name = get_experiment_name(train_mode, eval_mode) if eval_mode else train_mode
    
    paths = {
        'output_dir': output_dir,
        'model_dir': os.path.join(output_dir, "models", train_mode),
        'prediction_dir': os.path.join(output_dir, "predictions"),
        'evaluation_dir': os.path.join(output_dir, "evaluations"),
        'optimization_dir': os.path.join(output_dir, "optimization_results")
    }
    
    # 确保所有目录存在
    for path in paths.values():
        ensure_dir_exists(path)
    
    return paths
