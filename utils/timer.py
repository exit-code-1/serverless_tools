# -*- coding: utf-8 -*-
"""
计时工具模块
包含计时器类
"""

import time

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
