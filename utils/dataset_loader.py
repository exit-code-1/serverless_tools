# -*- coding: utf-8 -*-
"""
统一数据集加载器
提供统一的数据集读取接口，支持TPCH和TPCDS数据集
"""

import os
import random
import pandas as pd
from typing import Dict, Optional, Union, Tuple, Set, List
from .data_utils import load_csv_safe, save_csv_safe


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
            dataset_name: Dataset name (must exist in DATASETS in config/main_config.py)
        """
        from config.main_config import DATASETS
        if dataset_name not in DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. Available datasets: {list(DATASETS.keys())}"
            )
        
        self.dataset_name = dataset_name
        # Cache format:
        # - tpch/tpcds: key=(use_estimates,)
        # - job datasets: key=(use_estimates, train_ratio, seed)
        self._train_data_cache: Dict[Tuple, Optional[pd.DataFrame]] = {}
        self._test_data_cache: Dict[Tuple, Optional[pd.DataFrame]] = {}
        self._train_query_cache: Dict[Tuple, Optional[pd.DataFrame]] = {}
        self._test_query_cache: Dict[Tuple, Optional[pd.DataFrame]] = {}
        self._job_split_files_cache: Dict[Tuple[float, int], Dict[str, str]] = {}
        
        # 从配置获取路径
        from config.main_config import PROJECT_ROOT, DATASETS
        self.data_dir = os.path.join(PROJECT_ROOT, "data_kunpeng")
        
        dataset_config = DATASETS[dataset_name]
        self.train_dir = os.path.join(self.data_dir, dataset_config['train_dir'])
        self.test_dir = os.path.join(self.data_dir, dataset_config['test_dir'])
        self.plan_info_file = dataset_config['plan_info_file']
        self.query_info_file = dataset_config['query_info_file']
        self.split_mode = dataset_config.get("split_mode")  # None for tpch/tpcds

    def _is_job_dataset(self) -> bool:
        return self.split_mode == "query_id_ratio"

    def _default_train_ratio(self) -> float:
        from config.main_config import DEFAULT_CONFIG
        return float(DEFAULT_CONFIG.get("train_ratio", 0.8))

    def _job_split_root(self, train_ratio: float, seed: int) -> str:
        """
        Put generated split CSVs under output/<dataset_name>/splits/.
        """
        from config.main_config import PROJECT_ROOT
        safe_ratio = f"{train_ratio:.6f}".rstrip("0").rstrip(".")
        return os.path.join(PROJECT_ROOT, "output", self.dataset_name, "splits", f"ratio_{safe_ratio}", f"seed_{seed}")

    def _clamp_train_ratio(self, train_ratio: float) -> float:
        """
        Ensure train_ratio is not extreme so both train/test are non-empty.
        """
        r = float(train_ratio)
        if r <= 0.0:
            return 0.2
        if r >= 1.0:
            # Keep behavior safe for existing configs with train_ratio=1.0
            print(f"Warning: train_ratio={r} is too large for job dataset; clamping to 0.8 to keep non-empty test set.")
            return 0.8
        return r

    def _ensure_job_split_files(self, train_ratio: float, seed: int) -> Dict[str, str]:
        """
        Generate (train/test) filtered plan_info.csv and query_info.csv for job datasets.
        Returns file paths: {train_plan, test_plan, train_query, test_query, split_csv}.
        """
        if not self._is_job_dataset():
            raise RuntimeError("_ensure_job_split_files called for non-job dataset.")

        train_ratio = self._clamp_train_ratio(train_ratio)
        cache_key = (train_ratio, seed)
        if cache_key in self._job_split_files_cache:
            return self._job_split_files_cache[cache_key]

        split_root = self._job_split_root(train_ratio, seed)
        split_csv_path = os.path.join(split_root, "query_split.csv")
        train_plan_path = os.path.join(split_root, "plan_info_train.csv")
        test_plan_path = os.path.join(split_root, "plan_info_test.csv")
        train_query_path = os.path.join(split_root, "query_info_train.csv")
        test_query_path = os.path.join(split_root, "query_info_test.csv")

        # If all outputs exist, reuse them.
        if all(os.path.exists(p) for p in [split_csv_path, train_plan_path, test_plan_path, train_query_path, test_query_path]):
            result = {
                "split_csv": split_csv_path,
                "train_plan": train_plan_path,
                "test_plan": test_plan_path,
                "train_query": train_query_path,
                "test_query": test_query_path,
            }
            self._job_split_files_cache[cache_key] = result
            return result

        os.makedirs(split_root, exist_ok=True)

        base_plan_path = os.path.join(self.train_dir, self.plan_info_file)
        base_query_path = os.path.join(self.train_dir, self.query_info_file)

        query_df = load_csv_safe(base_query_path, description=f"{self.dataset_name.upper()} query_info (base)")
        if query_df is None or "query_id" not in query_df.columns:
            raise ValueError(f"Failed to load base query_info for job dataset: {base_query_path}")

        # Use unique query_id to create a stable train/test split.
        query_ids = sorted(set(query_df["query_id"].astype(int).tolist()))
        if not query_ids:
            raise ValueError(f"No query_id found in {base_query_path}")

        rng = random.Random(seed)
        rng.shuffle(query_ids)

        split_point = int(len(query_ids) * train_ratio)
        if split_point <= 0 or split_point >= len(query_ids):
            raise ValueError(
                f"Invalid split computed for job dataset: len(query_ids)={len(query_ids)}, "
                f"train_ratio={train_ratio} -> split_point={split_point}. "
                f"Please use train_ratio between 0 and 1 (exclusive)."
            )

        train_qids = set(query_ids[:split_point])
        test_qids = set(query_ids[split_point:])

        query_split_df = pd.DataFrame({
            "query_id": query_ids,
            "split": ["train" if qid in train_qids else "test" for qid in query_ids],
        })
        query_split_df.to_csv(split_csv_path, index=False, sep=";")

        plan_df = load_csv_safe(base_plan_path, description=f"{self.dataset_name.upper()} plan_info (base)")
        if plan_df is None or "query_id" not in plan_df.columns:
            raise ValueError(f"Failed to load base plan_info for job dataset: {base_plan_path}")

        train_plan_df = plan_df[plan_df["query_id"].astype(int).isin(train_qids)].copy()
        test_plan_df = plan_df[plan_df["query_id"].astype(int).isin(test_qids)].copy()
        train_query_df = query_df[query_df["query_id"].astype(int).isin(train_qids)].copy()
        test_query_df = query_df[query_df["query_id"].astype(int).isin(test_qids)].copy()

        # Keep delimiter consistent with the rest of the codebase.
        train_plan_df.to_csv(train_plan_path, index=False, sep=";")
        test_plan_df.to_csv(test_plan_path, index=False, sep=";")
        train_query_df.to_csv(train_query_path, index=False, sep=";")
        test_query_df.to_csv(test_query_path, index=False, sep=";")

        result = {
            "split_csv": split_csv_path,
            "train_plan": train_plan_path,
            "test_plan": test_plan_path,
            "train_query": train_query_path,
            "test_query": test_query_path,
        }
        self._job_split_files_cache[cache_key] = result
        print(f"Job dataset split generated for {self.dataset_name}: train_ratio={train_ratio}, seed={seed}")
        return result
    
    def load_train_data(
        self,
        use_estimates: bool = False,
        train_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Optional[pd.DataFrame]:
        """
        加载训练集计划数据
        
        Args:
            use_estimates: 是否使用估计值模式
            
        Returns:
            训练集计划数据DataFrame，如果加载失败返回None
        """
        if self._is_job_dataset():
            if train_ratio is None:
                train_ratio = self._default_train_ratio()
            cache_key = (use_estimates, float(train_ratio), int(seed))
            if cache_key in self._train_data_cache:
                return self._train_data_cache[cache_key]

            paths = self._ensure_job_split_files(float(train_ratio), int(seed))
            plan_info_path = paths["train_plan"]
            train_df = load_csv_safe(plan_info_path, description=f"{self.dataset_name.upper()} train plan (split)")
            if train_df is not None and use_estimates:
                from .data_utils import propagate_estimates_in_dataframe
                train_df = propagate_estimates_in_dataframe(train_df)
                print(f"Applied cardinality-estimate propagation to {self.dataset_name.upper()} train plan")
            self._train_data_cache[cache_key] = train_df
            return train_df

        # tpch/tpcds
        cache_key = (use_estimates,)
        if cache_key in self._train_data_cache:
            return self._train_data_cache[cache_key]
            
        plan_info_path = os.path.join(self.train_dir, self.plan_info_file)
        train_df = load_csv_safe(
            plan_info_path, 
            description=f"{self.dataset_name.upper()}训练集计划数据"
        )
        
        if train_df is not None and use_estimates:
            from .data_utils import propagate_estimates_in_dataframe
            train_df = propagate_estimates_in_dataframe(train_df)
            print(f"已对{self.dataset_name.upper()}训练集应用基数估计传播")
        
        self._train_data_cache[cache_key] = train_df
        return train_df
    
    def load_test_data(
        self,
        use_estimates: bool = False,
        train_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Optional[pd.DataFrame]:
        """
        加载测试集计划数据
        
        Args:
            use_estimates: 是否使用估计值模式
            
        Returns:
            测试集计划数据DataFrame，如果加载失败返回None
        """
        if self._is_job_dataset():
            if train_ratio is None:
                train_ratio = self._default_train_ratio()
            cache_key = (use_estimates, float(train_ratio), int(seed))
            if cache_key in self._test_data_cache:
                return self._test_data_cache[cache_key]

            paths = self._ensure_job_split_files(float(train_ratio), int(seed))
            plan_info_path = paths["test_plan"]
            test_df = load_csv_safe(plan_info_path, description=f"{self.dataset_name.upper()} test plan (split)")
            if test_df is not None and use_estimates:
                from .data_utils import propagate_estimates_in_dataframe
                test_df = propagate_estimates_in_dataframe(test_df)
                print(f"Applied cardinality-estimate propagation to {self.dataset_name.upper()} test plan")
            self._test_data_cache[cache_key] = test_df
            return test_df

        cache_key = (use_estimates,)
        if cache_key in self._test_data_cache:
            return self._test_data_cache[cache_key]
            
        plan_info_path = os.path.join(self.test_dir, self.plan_info_file)
        test_df = load_csv_safe(
            plan_info_path, 
            description=f"{self.dataset_name.upper()}测试集计划数据"
        )
        
        if test_df is not None and use_estimates:
            from .data_utils import propagate_estimates_in_dataframe
            test_df = propagate_estimates_in_dataframe(test_df)
            print(f"已对{self.dataset_name.upper()}测试集应用基数估计传播")
        
        self._test_data_cache[cache_key] = test_df
        return test_df
    
    def load_train_query_info(
        self,
        train_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Optional[pd.DataFrame]:
        """
        加载训练集查询信息
        
        Returns:
            训练集查询信息DataFrame，如果加载失败返回None
        """
        if self._is_job_dataset():
            if train_ratio is None:
                train_ratio = self._default_train_ratio()
            cache_key = (float(train_ratio), int(seed))
            if cache_key in self._train_query_cache:
                return self._train_query_cache[cache_key]

            paths = self._ensure_job_split_files(float(train_ratio), int(seed))
            query_info_path = paths["train_query"]
            query_df = load_csv_safe(query_info_path, description=f"{self.dataset_name.upper()} train query (split)")
            self._train_query_cache[cache_key] = query_df
            return query_df

        cache_key = tuple()
        if cache_key in self._train_query_cache:
            return self._train_query_cache[cache_key]
            
        query_info_path = os.path.join(self.train_dir, self.query_info_file)
        query_df = load_csv_safe(
            query_info_path, 
            description=f"{self.dataset_name.upper()}训练集查询信息"
        )
        
        self._train_query_cache[cache_key] = query_df
        return query_df
    
    def load_test_query_info(
        self,
        train_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Optional[pd.DataFrame]:
        """
        加载测试集查询信息
        
        Returns:
            测试集查询信息DataFrame，如果加载失败返回None
        """
        if self._is_job_dataset():
            if train_ratio is None:
                train_ratio = self._default_train_ratio()
            cache_key = (float(train_ratio), int(seed))
            if cache_key in self._test_query_cache:
                return self._test_query_cache[cache_key]

            paths = self._ensure_job_split_files(float(train_ratio), int(seed))
            query_info_path = paths["test_query"]
            query_df = load_csv_safe(query_info_path, description=f"{self.dataset_name.upper()} test query (split)")
            self._test_query_cache[cache_key] = query_df
            return query_df

        cache_key = tuple()
        if cache_key in self._test_query_cache:
            return self._test_query_cache[cache_key]
            
        query_info_path = os.path.join(self.test_dir, self.query_info_file)
        query_df = load_csv_safe(
            query_info_path, 
            description=f"{self.dataset_name.upper()}测试集查询信息"
        )
        
        self._test_query_cache[cache_key] = query_df
        return query_df
    
    def load_all_data(
        self,
        use_estimates: bool = False,
        train_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载所有数据（训练集和测试集的计划和查询信息）
        
        Args:
            use_estimates: 是否使用估计值模式
            
        Returns:
            包含所有数据的字典，如果任何数据加载失败返回None
        """
        train_data = self.load_train_data(use_estimates, train_ratio=train_ratio, seed=seed)
        test_data = self.load_test_data(use_estimates, train_ratio=train_ratio, seed=seed)
        train_query_info = self.load_train_query_info(train_ratio=train_ratio, seed=seed)
        test_query_info = self.load_test_query_info(train_ratio=train_ratio, seed=seed)
        
        if any(data is None for data in [train_data, test_data, train_query_info, test_query_info]):
            return None
        
        return {
            'train_plan': train_data,
            'test_plan': test_data,
            'train_query': train_query_info,
            'test_query': test_query_info
        }
    
    def get_file_paths(
        self,
        mode: str = 'train',
        train_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Dict[str, str]:
        """
        获取指定模式的文件路径
        
        Args:
            mode: 模式，支持 'train', 'test'
            
        Returns:
            包含文件路径的字典
        """
        if not self._is_job_dataset():
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

        # Job dataset: return generated split CSV paths.
        if train_ratio is None:
            train_ratio = self._default_train_ratio()
        paths = self._ensure_job_split_files(float(train_ratio), int(seed))

        if mode == 'train':
            return {
                'dir': os.path.dirname(paths["train_plan"]),
                'plan_info': paths["train_plan"],
                'query_info': paths["train_query"],
            }
        elif mode == 'test':
            return {
                'dir': os.path.dirname(paths["test_plan"]),
                'plan_info': paths["test_plan"],
                'query_info': paths["test_query"],
            }
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'train', 'test'")
    
    def clear_cache(self):
        """清除缓存的数据"""
        self._train_data_cache.clear()
        self._test_data_cache.clear()
        self._train_query_cache.clear()
        self._test_query_cache.clear()
        self._job_split_files_cache.clear()


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
