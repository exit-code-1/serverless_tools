"""
MCI Model Main Entry Point
MCI模型主入口文件

统一管理训练和推理功能
"""

import os
import sys
import subprocess

# =============================================================================
# 配置参数 - 在这里修改所有参数
# =============================================================================

# 模式选择: 'train', 'infer', 或 'optimize'
MODE = 'train'  # 修改这里选择训练、推理或优化模式

# 配置文件路径
CONFIG_FILE = 'mci_config_small.json'  # 修改配置文件路径




# =============================================================================
# 主函数
# =============================================================================

def run_training():
    """Run MCI model training"""
    print("Starting MCI model training...")
    
    # Build training command
    cmd = [sys.executable, "mci_train.py", "--config", CONFIG_FILE]
    
    # Run training
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode


def run_inference():
    """Run MCI model inference"""
    print("Starting MCI model inference...")
    
    # Build inference command
    cmd = [sys.executable, "mci_inference.py", "--config", CONFIG_FILE]
    
    # Run inference
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode


def run_optimization():
    """Run MCI MOO optimization"""
    print("Starting MCI MOO optimization...")
    
    # Build optimization command
    cmd = [sys.executable, "mci_optimize.py", "--config", CONFIG_FILE]
    
    # Run optimization
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode


def main():
    """Main entry point"""
    print(f"Running MCI model in {MODE} mode...")
    print(f"Config file: {CONFIG_FILE}")
    
    if MODE == 'train':
        return run_training()
    elif MODE == 'infer':
        return run_inference()
    elif MODE == 'optimize':
        return run_optimization()
    else:
        print(f"Error: Unknown mode '{MODE}'. Use 'train', 'infer', or 'optimize'.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
