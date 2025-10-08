# -*- coding: utf-8 -*-
"""
系统工具模块
包含环境设置、配置验证、安全导入等功能
"""

import sys

# ==================== 环境设置 ====================
def setup_environment():
    """设置项目环境"""
    from config import PROJECT_ROOT
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    print(f"项目根目录: {PROJECT_ROOT}")

def setup_config_structure():
    """设置配置结构"""
    from config.main_config import USE_HASH_TABLE_SIZE_FEATURE
    # 这里可以设置全局配置，如果需要的话
    print("配置结构已设置")

# ==================== 安全导入 ====================
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

def validate_experiment_config(dataset: str, method: str, train_mode: str) -> bool:
    """验证实验配置"""
    try:
        # Use lazy import to avoid circular import
        import config.main_config
        config.main_config.validate_dataset(dataset)
        config.main_config.validate_method(method)
        config.main_config.validate_train_mode(train_mode)
        return True
    except ValueError as e:
        print(f"配置验证失败: {e}")
        return False
