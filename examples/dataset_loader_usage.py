# -*- coding: utf-8 -*-
"""
数据集加载器使用示例
演示如何使用统一的数据集加载器
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_dataset_loader, load_dataset_data


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建TPCDS数据集加载器
    loader = create_dataset_loader('tpcds')
    
    # 加载训练数据
    train_data = loader.load_train_data(use_estimates=False)
    print(f"训练数据形状: {train_data.shape if train_data is not None else 'None'}")
    
    # 加载测试数据
    test_data = loader.load_test_data(use_estimates=False)
    print(f"测试数据形状: {test_data.shape if test_data is not None else 'None'}")
    
    # 加载查询信息
    train_query_info = loader.load_train_query_info()
    test_query_info = loader.load_test_query_info()
    print(f"训练查询信息形状: {train_query_info.shape if train_query_info is not None else 'None'}")
    print(f"测试查询信息形状: {test_query_info.shape if test_query_info is not None else 'None'}")


def example_with_estimates():
    """使用估计值模式示例"""
    print("\n=== 使用估计值模式示例 ===")
    
    # 创建TPCH数据集加载器
    loader = create_dataset_loader('tpch')
    
    # 加载使用估计值的数据
    train_data = loader.load_train_data(use_estimates=True)
    print(f"TPCH训练数据形状（使用估计值）: {train_data.shape if train_data is not None else 'None'}")


def example_load_all_data():
    """加载所有数据示例"""
    print("\n=== 加载所有数据示例 ===")
    
    # 创建TPCDS数据集加载器
    loader = create_dataset_loader('tpcds')
    
    # 一次性加载所有数据
    all_data = loader.load_all_data(use_estimates=False)
    
    if all_data is not None:
        print("成功加载所有数据:")
        for key, data in all_data.items():
            print(f"  {key}: {data.shape}")
    else:
        print("数据加载失败")


def example_convenience_function():
    """便捷函数使用示例"""
    print("\n=== 便捷函数使用示例 ===")
    
    # 使用便捷函数直接加载数据
    train_data = load_dataset_data('tpcds', mode='train', use_estimates=False)
    print(f"便捷函数加载训练数据形状: {train_data.shape if train_data is not None else 'None'}")
    
    # 加载包含查询信息的数据
    train_data_with_query = load_dataset_data('tpcds', mode='train', use_estimates=False, include_query_info=True)
    if train_data_with_query is not None:
        print("便捷函数加载训练数据（包含查询信息）:")
        for key, data in train_data_with_query.items():
            print(f"  {key}: {data.shape}")
    
    # 加载所有数据
    all_data = load_dataset_data('tpcds', mode='both', use_estimates=False)
    if all_data is not None:
        print("便捷函数加载所有数据:")
        for key, data in all_data.items():
            print(f"  {key}: {data.shape}")


def example_file_paths():
    """获取文件路径示例"""
    print("\n=== 获取文件路径示例 ===")
    
    # 创建数据集加载器
    loader = create_dataset_loader('tpcds')
    
    # 获取训练集文件路径
    train_paths = loader.get_file_paths('train')
    print("训练集文件路径:")
    for key, path in train_paths.items():
        print(f"  {key}: {path}")
    
    # 获取测试集文件路径
    test_paths = loader.get_file_paths('test')
    print("测试集文件路径:")
    for key, path in test_paths.items():
        print(f"  {key}: {path}")


def example_cache_management():
    """缓存管理示例"""
    print("\n=== 缓存管理示例 ===")
    
    # 创建数据集加载器
    loader = create_dataset_loader('tpcds')
    
    # 第一次加载（会从文件读取）
    print("第一次加载数据...")
    train_data1 = loader.load_train_data()
    
    # 第二次加载（会使用缓存）
    print("第二次加载数据（使用缓存）...")
    train_data2 = loader.load_train_data()
    
    # 清除缓存
    print("清除缓存...")
    loader.clear_cache()
    
    # 第三次加载（会重新从文件读取）
    print("第三次加载数据（重新读取文件）...")
    train_data3 = loader.load_train_data()
    
    print(f"数据一致性检查: {train_data1 is train_data2} (应该为True，因为使用了缓存)")
    print(f"缓存清除检查: {train_data1 is train_data3} (应该为False，因为清除了缓存)")


def main():
    """主函数"""
    print("数据集加载器使用示例")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_with_estimates()
        example_load_all_data()
        example_convenience_function()
        example_file_paths()
        example_cache_management()
        
        print("\n" + "=" * 50)
        print("所有示例执行完成！")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
