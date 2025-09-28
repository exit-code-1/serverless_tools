import os
import time
import numpy as np
import math
import onnxruntime
import torch
# import utils
# from structure import no_dop_operator_features, no_dop_operators_exec, no_dop_operators_mem, dop_operators_exec, dop_operators_mem, parallel_op
from .plan_node import PlanNode # 如果需要取消注释

class ThreadBlock:
    def __init__(self, thread_id, nodes):
        """
        :param thread_id: 线程块标识
        :param nodes: 属于该线程块的所有 PlanNode
        """
        self.thread_id = thread_id
        self.visit = False
        self.nodes = nodes
        self.candidate_dops = set()  # 可后续设置
        self.real_dop_exec_time = {}     
        self.pred_dop_exec_time = {}  
        self.thread_execution_time = 0
        self.thread_complete_time = 0
        self.local_data_transfer_start_time = 0
        self.up_data_transfer_start_time = 0
        self.child_thread_ids = set()  # 用于记录子线程块的 thread_id
        self.child_max_execution_time = 0  # 用于记录子线程内的最大执行时间
        self.optimal_dop = None        # 用于记录最终最优 dop
        self.blocking_interval = {}
        self.pred_time = 0
        self.candidate_optimal_dops = []
        self.total_exec_time = {} #加上发送时间

    def aggregate_metrics(self):
        """
        聚合线程块内所有节点的指标:
          - thread_execution_time:累加所有节点的执行时间
          - thread_complete_time:所有节点中最大的完成时间
          - local/up_data_transfer_start_time:取所有节点中的最大值(仅作参考)
          - candidate_dops:如果未设置,则取所有阻塞节点(materialized==True)的 matched_dops 的交集
          - blocking_interval:在 candidate_dops 下,
              从所有阻塞节点中选取 thread_execution_time 最大的那个作为“最上层阻塞节点”,
              然后直接使用该节点的 pred_dop_exec_map 作为阻塞时间
          - 同时,从该节点的 dop_exec_time_map 取得真实的累加执行时间
        """
        if not self.nodes:
            return

        # 聚合基本指标
        self.thread_execution_time = sum(node.execution_time for node in self.nodes)
        self.thread_complete_time = max(node.thread_complete_time for node in self.nodes)
        self.local_data_transfer_start_time = max(node.local_data_transfer_start_time for node in self.nodes)
        self.up_data_transfer_start_time = max(node.up_data_transfer_start_time for node in self.nodes)
        
        # 如果没有 candidate_dops,则取所有节点 matched_dops 的交集
        if not self.candidate_dops:
            candidate_dops = None
            for node in self.nodes:
                if not node.is_parallel or node.parent_node is None:
                    candidate_dops = {1}
                    break
                if candidate_dops is None:
                    candidate_dops = set(node.matched_dops)
                else:
                    candidate_dops = candidate_dops.intersection(node.matched_dops)
            self.candidate_dops = candidate_dops if candidate_dops is not None else set()
            
        # 在线程块内选出“最上层阻塞节点”:定义为 materialized==True 且 thread_execution_time 最大的那个节点
        top_blocking = None
        top_node = self.nodes[0]
        max_block_exec = -1
        max_exec = -1
        for node in self.nodes:
            if node.materialized and node.up_data_transfer_start_time > max_block_exec:
                max_block_exec = node.up_data_transfer_start_time
                top_blocking = node
            if node.thread_execution_time > max_exec:
                max_exec = node.thread_execution_time
                top_node = node

        self.real_dop_exec_time = top_node.true_dop_exec_map.copy()
        self.pred_dop_exec_time = top_node.pred_dop_exec_map.copy()
        # 如果存在阻塞节点,就直接从该节点获取累加的真实和预测执行时间
        if top_blocking:
            self.blocking_interval = {
            dop: top_blocking.pred_dop_exec_map.get(dop, float('inf')) - top_blocking.build_dop_exec_map.get(dop, 0)
            for dop in top_blocking.pred_dop_exec_map
        }
        else:
            self.blocking_interval = {dop: 0 for dop in self.candidate_dops}
            
    def choose_optimal_dop(self,
                           child_upload_times: dict,
                           min_improvement_ratio=0.15,
                           min_reduction_threshold=100,
                           ):
        """
        选择最优 DOP:
          1. 根据预测执行时间曲线，选取右边界 d_right
          2. 根据各子线程块上传时长，生成一组左边界 L
          3. 将 L 与 d_right 组合为候选集并返回

        :param child_upload_times: dict, key=child_thread_id, value=upload time (ms)
        :return: tuple (L, d_right)
        """
        if not self.candidate_dops:
            return None
        if len(self.candidate_dops) == 1:
            only = next(iter(self.candidate_dops))
            self.optimal_dop = only
            self.pred_time = self.pred_dop_exec_time.get(only, float('inf'))
            self.candidate_optimal_dops = [only]
            return [only]

        # 1. 按 DOP 升序排序
        sorted_dops = sorted(self.candidate_dops)

        # 2. 确定右边界 d_right
        d_right = sorted_dops[0]
        for d in sorted_dops[1:]:
            t_prev = self.pred_dop_exec_time.get(d_right, float('inf'))
            t_cur = self.pred_dop_exec_time.get(d, float('inf'))
            # 绝对降低
            delta = t_prev - t_cur
            if delta < min_reduction_threshold:
                continue
            # 归一化收益
            factor = 1 + math.log2(d - d_right) if d > d_right else 1
            adj_gain = (delta / t_prev) / factor
            dynamic_min_improvement = min_improvement_ratio / math.log(delta)
            if adj_gain >= dynamic_min_improvement:
                d_right = d
            else:
                continue
        if d_right < 8:
            d_right = 8
        # 3. 生成左边界集合 L
        L = set()
        # 这里用预测执行时间匹配子上传时长
        for child_id, up_time in child_upload_times.items():
            best_d, best_diff = None, float('inf')
            for d in sorted_dops:
                if d == 1:
                    continue
                if d > d_right:
                    break
                t = self.pred_dop_exec_time.get(d, float('inf'))
                diff = abs(t - up_time)
                if diff < best_diff:
                    best_diff = diff
                    best_d = d
            if best_d is not None:
                L.add(best_d)

        # 汇总候选
        candidates = sorted(L) + [d_right]
        self.candidate_optimal_dops = candidates
        # 默认取最大的
        self.optimal_dop = candidates[-1]
        self.pred_time = self.pred_dop_exec_time.get(self.optimal_dop)
        return candidates
    def recursive_choose_optimal_dop(self,
                                  all_thread_blocks: dict,
                                  min_improvement_ratio=0.2,
                                  min_reduction_threshold=200):
        """
        自底向上递归选择最优 DOP 并构建候选路径。
        """
        if getattr(self, 'visit', True):
            return self.pred_time  # 已处理则返回自己的上传时间
        self.visit = True
        # 先处理所有子线程块
        child_upload_times = {}
        for child_id in self.child_thread_ids:
            child_tb = all_thread_blocks.get(child_id)
            if child_tb is None:
                continue
            up_time = child_tb.recursive_choose_optimal_dop(
                all_thread_blocks,
                min_improvement_ratio=min_improvement_ratio,
                min_reduction_threshold=min_reduction_threshold
            )
            if up_time is not None:
                blocking = child_tb.blocking_interval.get(child_tb.optimal_dop, 0)
                child_upload_times[child_id] = up_time - blocking

        # 再处理自身
        self.choose_optimal_dop(
            child_upload_times=child_upload_times,
            min_improvement_ratio=min_improvement_ratio,
            min_reduction_threshold=min_reduction_threshold
        )

        self.visit = True
        return self.pred_time

