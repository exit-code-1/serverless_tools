# -*- coding: utf-8 -*-
"""
统一数据集加载器
提供统一的数据集读取接口，支持TPCH和TPCDS数据集
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from .data_utils import load_csv_safe


class DatasetLoader:
    """
    统一数据集加载器类
    
    支持：
    - TPCH数据集（训练集：tpch_output_500，测试集：tpch_output_22）
    - TPCDS数据集（训练集：tpcds_100g_output_train，测试集：tpcds_100g_output_test）
    """
    
    def __init__(self, dataset_name: str):
        """
        初始化数据集加载器
        
        Args:
            dataset_name: 数据集名称，支持 'tpch' 或 'tpcds'
        """
        if dataset_name not in ['tpch', 'tpcds']:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: 'tpch', 'tpcds'")
        
        self.dataset_name = dataset_name
        self._train_data = None
        self._test_data = None
        self._train_query_info = None
        self._test_query_info = None
        
        # 从配置获取路径
        from config.main_config import PROJECT_ROOT, DATASETS
        self.data_dir = os.path.join(PROJECT_ROOT, "data_kunpeng")
        
        dataset_config = DATASETS[dataset_name]
        self.train_dir = os.path.join(self.data_dir, dataset_config['train_dir'])
        self.test_dir = os.path.join(self.data_dir, dataset_config['test_dir'])
        self.plan_info_file = dataset_config['plan_info_file']
        self.query_info_file = dataset_config['query_info_file']
    
    def load_train_data(self, use_estimates: bool = False) -> Optional[pd.DataFrame]:
        """
        加载训练集计划数据
        
        Args:
            use_estimates: 是否使用估计值模式
            
        Returns:
            训练集计划数据DataFrame，如果加载失败返回None
        """
        if self._train_data is not None:
            return self._train_data
            
        plan_info_path = os.path.join(self.train_dir, self.plan_info_file)
        self._train_data = load_csv_safe(
            plan_info_path, 
            description=f"{self.dataset_name.upper()}训练集计划数据"
        )
        
        if self._train_data is not None and use_estimates:
            from .data_utils import propagate_estimates_in_dataframe
            self._train_data = propagate_estimates_in_dataframe(self._train_data)
            print(f"已对{self.dataset_name.upper()}训练集应用基数估计传播")
        
        return self._train_data
    
    def load_test_data(self, use_estimates: bool = False) -> Optional[pd.DataFrame]:
        """
        加载测试集计划数据
        
        Args:
            use_estimates: 是否使用估计值模式
            
        Returns:
            测试集计划数据DataFrame，如果加载失败返回None
        """
        if self._test_data is not None:
            return self._test_data
            
        plan_info_path = os.path.join(self.test_dir, self.plan_info_file)
        self._test_data = load_csv_safe(
            plan_info_path, 
            description=f"{self.dataset_name.upper()}测试集计划数据"
        )
        
        if self._test_data is not None and use_estimates:
            from .data_utils import propagate_estimates_in_dataframe
            self._test_data = propagate_estimates_in_dataframe(self._test_data)
            print(f"已对{self.dataset_name.upper()}测试集应用基数估计传播")
        
        return self._test_data
    
    def load_train_query_info(self) -> Optional[pd.DataFrame]:
        """
        加载训练集查询信息
        
        Returns:
            训练集查询信息DataFrame，如果加载失败返回None
        """
        if self._train_query_info is not None:
            return self._train_query_info
            
        query_info_path = os.path.join(self.train_dir, self.query_info_file)
        self._train_query_info = load_csv_safe(
            query_info_path, 
            description=f"{self.dataset_name.upper()}训练集查询信息"
        )
        
        return self._train_query_info
    
    def load_test_query_info(self) -> Optional[pd.DataFrame]:
        """
        加载测试集查询信息
        
        Returns:
            测试集查询信息DataFrame，如果加载失败返回None
        """
        if self._test_query_info is not None:
            return self._test_query_info
            
        query_info_path = os.path.join(self.test_dir, self.query_info_file)
        self._test_query_info = load_csv_safe(
            query_info_path, 
            description=f"{self.dataset_name.upper()}测试集查询信息"
        )
        
        return self._test_query_info
    
    def load_all_data(self, use_estimates: bool = False) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载所有数据（训练集和测试集的计划和查询信息）
        
        Args:
            use_estimates: 是否使用估计值模式
            
        Returns:
            包含所有数据的字典，如果任何数据加载失败返回None
        """
        train_data = self.load_train_data(use_estimates)
        test_data = self.load_test_data(use_estimates)
        train_query_info = self.load_train_query_info()
        test_query_info = self.load_test_query_info()
        
        if any(data is None for data in [train_data, test_data, train_query_info, test_query_info]):
            return None
        
        return {
            'train_plan': train_data,
            'test_plan': test_data,
            'train_query': train_query_info,
            'test_query': test_query_info
        }
    
    def get_file_paths(self, mode: str = 'train') -> Dict[str, str]:
        """
        获取指定模式的文件路径
        
        Args:
            mode: 模式，支持 'train', 'test'
            
        Returns:
            包含文件路径的字典
        """
        if mode == 'train':
            return {
                'dir': self.train_dir,
                'plan_info': os.path.join(self.train_dir, self.plan_info_file),
                'query_info': os.path.join(self.train_dir, self.query_info_file)
            }
        elif mode == 'test':
            return {
                'dir': self.test_dir,
                'plan_info': os.path.join(self.test_dir, self.plan_info_file),
                'query_info': os.path.join(self.test_dir, self.query_info_file)
            }
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: 'train', 'test'")
    
    def clear_cache(self):
        """清除缓存的数据"""
        self._train_data = None
        self._test_data = None
        self._train_query_info = None
        self._test_query_info = None


def create_dataset_loader(dataset_name: str) -> DatasetLoader:
    """
    创建数据集加载器的工厂函数
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        DatasetLoader实例
    """
    return DatasetLoader(dataset_name)


def load_dataset_data(dataset_name: str, mode: str = 'train', 
                     use_estimates: bool = False, 
                     include_query_info: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame], None]:
    """
    便捷函数：直接加载数据集数据
    
    Args:
        dataset_name: 数据集名称
        mode: 加载模式，'train', 'test', 或 'both'
        use_estimates: 是否使用估计值
        include_query_info: 是否包含查询信息
        
    Returns:
        根据参数返回相应的数据
    """
    loader = create_dataset_loader(dataset_name)
    
    if mode == 'train':
        if include_query_info:
            return {
                'plan': loader.load_train_data(use_estimates),
                'query': loader.load_train_query_info()
            }
        else:
            return loader.load_train_data(use_estimates)
    
    elif mode == 'test':
        if include_query_info:
            return {
                'plan': loader.load_test_data(use_estimates),
                'query': loader.load_test_query_info()
            }
        else:
            return loader.load_test_data(use_estimates)
    
    elif mode == 'both':
        return loader.load_all_data(use_estimates)
    
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes: 'train', 'test', 'both'")


# 向后兼容的函数
def get_tpch_loader() -> DatasetLoader:
    """获取TPCH数据集加载器"""
    return create_dataset_loader('tpch')


def get_tpcds_loader() -> DatasetLoader:
    """获取TPCDS数据集加载器"""
    return create_dataset_loader('tpcds')
