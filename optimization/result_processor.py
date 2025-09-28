# 文件路径: optimization/result_processor.py

import csv
import json
import math
import os
import re
import numpy as np
import pandas as pd

# --- 导入重构后的模块 ---
from core.plan_node import PlanNode # 需要访问节点属性
from core.thread_block import ThreadBlock # 需要访问线程块属性
from config.structure import thread_cost, thread_mem, default_dop # 需要常量
# 可能还需要导入其他模块，取决于函数的具体实现
# from .threading_utils import calculate_query_execution_time # 如果需要重新计算时间
# --- 结束导入 ---

# ==============================================================================
# 结果输出与格式化 (来自原 dop_choosen.py)
# ==============================================================================

def collect_all_nodes_from_tree(root_nodes):
    """通过 DFS 收集树中所有节点 (这个函数在 threading_utils 里也有，这里再放一份或从那里导入)"""
    all_nodes = []
    visited = set()
    # 确保 root_nodes 是可迭代的
    stack = list(root_nodes) if isinstance(root_nodes, (list, tuple)) else []
    while stack:
        node = stack.pop()
        # 确保是 PlanNode 且有 plan_id
        if not isinstance(node, PlanNode) or not hasattr(node, 'plan_id') or node.plan_id in visited:
            continue
        visited.add(node.plan_id)
        all_nodes.append(node)
        # 确保 child_plans 是列表
        children = node.child_plans if isinstance(node.child_plans, list) else []
        stack.extend(children)
    return all_nodes

def write_query_details_to_file(query_trees, query_thread_blocks, output_file):
    """
    将优化后的详细查询信息写入 JSON 文件。
    (逻辑来自原 dop_choosen.py 的同名函数)

    Args:
        query_trees (dict): {query_id: [PlanNode list]} (可能需要用于获取所有节点)
        query_thread_blocks (dict): {query_id: {thread_id: ThreadBlock}}
        output_file (str): 输出 JSON 文件的路径。
    """
    print(f"开始将详细查询信息写入 JSON 文件: {output_file}")
    queries_info = []
    processed_queries = 0

    # 遍历有线程块信息的查询
    for query_id, thread_blocks in query_thread_blocks.items():
        if not thread_blocks: continue # 跳过没有线程块的查询

        query_dict = {"query_id": str(query_id)} # JSON 中通常用字符串 ID

        # 获取该查询的所有节点 (需要从 query_trees 中获取)
        # 注意：这里假设 query_trees 的键也是 query_id
        base_nodes = query_trees.get(query_id)
        if not base_nodes:
             print(f"警告: Query {query_id} 在 query_trees 中未找到节点信息，无法计算总时间和内存。")
             all_nodes_in_query = []
             # 尝试从线程块中收集节点，但不一定完整
             # for tb in thread_blocks.values(): all_nodes_in_query.extend(tb.nodes)
        else:
             # 需要找到根节点才能正确收集
             # root_nodes = get_root_nodes(base_nodes) # 需要 get_root_nodes 函数
             # all_nodes_in_query = collect_all_nodes_from_tree(root_nodes)
             # 或者直接用 base_nodes (如果它包含了所有节点)
             all_nodes_in_query = base_nodes # 假设 base_nodes 包含所有节点

             # 重置访问状态以重新计算时间和内存 (如果需要的话)
             # for node in all_nodes_in_query: node.visit = False
             # query_real_time = calculate_query_execution_time(all_nodes_in_query, thread_blocks) # 需要导入
             # total_mem, _ = calculate_query_memory(all_nodes_in_query) # 需要导入
             # query_dict["query_real_execution_time"] = query_real_time
             # query_dict["query_total_mem"] = int(total_mem) # 保持原始转换


        # 处理线程块信息
        cpu_time_sum = 0 # Query 级别的 CPU 时间累加
        total_threads_in_query = 0
        max_dop_in_query = 0
        tb_list = []

        for tb_id, tb in thread_blocks.items():
            if not isinstance(tb, ThreadBlock): continue

            tb_info = {"thread_block_id": tb_id}
            optimal_dop = tb.optimal_dop if tb.optimal_dop is not None else default_dop # 使用默认值
            max_dop_in_query = max(max_dop_in_query, optimal_dop)
            total_threads_in_query += optimal_dop
            tb_info["optimal_dop"] = optimal_dop
            # 使用 getattr 获取属性，避免 AttributeError
            tb_info["predicted_time"] = getattr(tb, 'pred_time', None)
            # 获取线程块的累加真实时间（可能来自 aggregate_metrics）
            tb_info["real_execution_time"] = getattr(tb, 'thread_execution_time', None) # 确认属性名
            if tb_info["real_execution_time"] is not None:
                 cpu_time_sum += tb_info["real_execution_time"]

            # 处理线程块内的算子信息
            ops = []
            for node in tb.nodes:
                if not isinstance(node, PlanNode): continue

                op_info = {
                    "plan_id": node.plan_id,
                    "operator_type": node.operator_type,
                    "width": node.width, # 原始 JSON 包含 width
                    "parent_child": node.parent_node.plan_id if node.parent_node else -1, # 父节点 ID
                    "left_child": node.child_plans[0].plan_id if node.child_plans else -1, # 左子节点 ID (简单假设)
                    # 预测时间 (最优 DOP 下)
                    "predicted_time": node.pred_dop_exec_map.get(optimal_dop),
                    # 真实/插值时间 (最优 DOP 下)
                    "real_time": node.true_dop_exec_map.get(optimal_dop)
                }
                # (原始代码中对 numpy array 的转换已在别处处理或不需要)
                ops.append(op_info)

            tb_info["operators"] = ops
            tb_list.append(tb_info)

        # 添加 Query 级别的汇总信息 (保持原始计算逻辑)
        query_dict["total_cpu_time"] = cpu_time_sum + total_threads_in_query * thread_cost
        query_dict["query_total_threads"] = total_threads_in_query
        # 内存计算需要所有节点信息，这里暂时省略或使用线程块内存累加？
        # 保持原始 JSON 格式，可能没有 query_total_mem？或者需要重新计算
        # query_dict["query_total_mem"] = int(total_mem + total_threads_in_query * thread_mem) # 如需计算
        query_dict["max_dop"] = max_dop_in_query
        query_dict["thread_blocks"] = tb_list
        queries_info.append(query_dict)
        processed_queries += 1

    output_data = {"queries": queries_info}

    # 写入 JSON 文件 (确保目录存在)
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        with open(output_file, "w", encoding="utf-8") as f:
            # 使用 ensure_ascii=False 保留中文字符（如果算子名等包含中文）
            # 使用 default=str 处理无法序列化的类型（例如 numpy 类型，虽然最好提前转换）
            json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)
        print(f"成功将 {processed_queries} 个查询的详细信息写入到 {output_file}")

    except Exception as e:
        print(f"写入 JSON 文件时出错: {e}")


# ==============================================================================
# 结果对比相关函数 (来自原 dop_choosen.py)
# ==============================================================================

def calculate_cpu_time(df_plans):
    """计算 CPU 时间 (直接搬运)"""
    # 确保 dop 和 execution_time 是数值
    df_plans['dop_numeric'] = pd.to_numeric(df_plans['dop'], errors='coerce').fillna(1) # 失败用1
    df_plans['exec_time_numeric'] = pd.to_numeric(df_plans['execution_time'], errors='coerce').fillna(0) # 失败用0
    df_plans['cpu_time'] = df_plans['exec_time_numeric'] * df_plans['dop_numeric']
    cpu_time = df_plans.groupby('query_id')['cpu_time'].sum().reset_index()
    return cpu_time

def calculate_thread_count(df_plans):
    """计算线程数 (直接搬运)"""
    # 检查 downdop 列是否存在且为数值
    if 'down_dop' not in df_plans.columns:
         print("警告: calculate_thread_count 找不到 'down_dop' 列。返回默认线程数 1。")
         all_q = pd.DataFrame({'query_id': df_plans['query_id'].unique()})
         all_q['thread_count'] = 1
         return all_q[['query_id', 'thread_count']]

    df_plans['down_dop_numeric'] = pd.to_numeric(df_plans['down_dop'], errors='coerce').fillna(0) # 失败用0
    df_plans['is_streaming'] = df_plans['operator_type'].str.lower().str.contains('streaming', na=False) # 处理 NaN
    df_stream = df_plans[df_plans['is_streaming']].copy()
    thread_count = df_stream.groupby('query_id')['down_dop_numeric'].sum().reset_index()
    thread_count.rename(columns={'down_dop_numeric': 'streaming_dop_sum'}, inplace=True)

    # 原始逻辑是 + 1
    thread_count['thread_count'] = thread_count['streaming_dop_sum'] + 1

    # 补全没有 streaming 的 query
    all_q = pd.DataFrame({'query_id': df_plans['query_id'].unique()})
    thread_count = pd.merge(all_q, thread_count[['query_id', 'thread_count']], on='query_id', how='left')
    thread_count['thread_count'].fillna(1, inplace=True) # 没有 streaming 的线程数为 1
    thread_count['thread_count'] = thread_count['thread_count'].astype(int) # 转整数
    return thread_count[['query_id', 'thread_count']]

def calculate_query_features(query_info_path, plan_info_path, prefix):
    """计算查询特征 (直接搬运)"""
    try:
        df_query = pd.read_csv(query_info_path, delimiter=';', encoding='utf-8')
        df_plan = pd.read_csv(plan_info_path, delimiter=';', encoding='utf-8')
    except FileNotFoundError as e:
        print(f"错误: 计算查询特征时文件未找到 - {e}")
        return pd.DataFrame() # 返回空 DataFrame
    except Exception as e:
        print(f"错误: 加载数据时出错 - {e}")
        return pd.DataFrame()

    cpu_time = calculate_cpu_time(df_plan).rename(columns={'cpu_time': f'{prefix}_cpu_time'})
    thread_count = calculate_thread_count(df_plan).rename(columns={'thread_count': f'{prefix}_thread_count'})

    # 合并特征 (确保 query_id 类型一致)
    try:
        df_query['query_id'] = df_query['query_id'].astype(int)
        cpu_time['query_id'] = cpu_time['query_id'].astype(int)
        thread_count['query_id'] = thread_count['query_id'].astype(int)

        df = pd.merge(df_query, cpu_time, on='query_id', how='left')
        df = pd.merge(df, thread_count, on='query_id', how='left')
    except KeyError as e:
         print(f"错误: 合并特征时缺少列 'query_id': {e}")
         return pd.DataFrame()
    except Exception as e:
        print(f"错误: 合并特征时出错: {e}")
        return pd.DataFrame()


    # 重命名列 (检查列是否存在)
    rename_map = {}
    if 'execution_time' in df.columns: rename_map['execution_time'] = f'{prefix}_execution_time'
    if 'query_used_mem' in df.columns: rename_map['query_used_mem'] = f'{prefix}_memory_usage'
    df = df.rename(columns=rename_map)

    # 只保留需要的列，并确保 query_id 在内
    required_cols = ['query_id'] + [col for col in df.columns if col.startswith(prefix)]
    return df[required_cols]

def compare_results(baseline_query_info_path, baseline_plan_info_path,
                             method_info_list, output_csv_path):
    """
    比较基线和多个新方案的结果
    :param baseline_query_info_path: 基线 query_info 路径
    :param baseline_plan_info_path: 基线 plan_info 路径
    :param method_info_list: 列表，每个元素为 (方法名, query_info_path, plan_info_path)
    :param output_csv_path: 输出路径
    """
    print("开始比较基线和多个新方案结果...")

    # 加载基线
    base = calculate_query_features(baseline_query_info_path, baseline_plan_info_path, 'base')
    if base.empty:
        print("错误: 无法计算基线特征，比较中止。")
        return

    # 初始化结果表
    comp = base[['query_id', 'base_execution_time', 'base_thread_count']].copy()
    comp['base_cost'] = comp['base_execution_time'] * comp['base_thread_count']

    # 定义安全除法函数
    def safe_divide(numerator, denominator):
        num = pd.to_numeric(numerator, errors='coerce')
        den = pd.to_numeric(denominator, errors='coerce')
        return np.where(np.isclose(den, 0) | np.isnan(num) | np.isnan(den), np.nan, num / den)

    # 逐个方法处理
    for method_name, query_info_path, plan_info_path in method_info_list:
        print(f"处理方法: {method_name}")
        new = calculate_query_features(query_info_path, plan_info_path, method_name)
        if new.empty:
            print(f"警告: 方法 {method_name} 的特征为空，跳过。")
            continue

        try:
            merged = pd.merge(comp[['query_id']], new, on='query_id', how='left')
        except Exception as e:
            print(f"错误: 合并方法 {method_name} 的结果失败: {e}")
            continue

        # 添加字段
        comp[f'{method_name}_execution_time'] = merged[f'{method_name}_execution_time']
        comp[f'{method_name}_thread_count'] = merged[f'{method_name}_thread_count']
        comp[f'{method_name}_cost'] = merged[f'{method_name}_execution_time'] * merged[f'{method_name}_thread_count']

        # 比值
        comp[f'{method_name}_time_ratio'] = safe_divide(comp[f'{method_name}_execution_time'], comp['base_execution_time'])
        comp[f'{method_name}_cost_ratio'] = safe_divide(comp[f'{method_name}_cost'], comp['base_cost'])

    # 保存
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        comp.to_csv(output_csv_path, sep=';', index=False, encoding='utf-8', float_format='%.6f')
        print(f'对比结果已保存至 {output_csv_path}')
    except Exception as e:
        print(f"错误: 保存结果时出错: {e}")



# ==============================================================================
# DOP 选择相关函数 (来自原 dop_utils.py)
# ==============================================================================

def select_optimal_dop(exec_time_map, min_improvement_ratio=0.15, min_reduction_threshold=300):
    """选择最优 DOP (直接搬运 dop_utils 逻辑)"""
    if not exec_time_map: return default_dop # 如果 map 为空，返回默认值

    # 先按 DOP 排序
    # 确保键是可排序的数值类型
    try:
        sorted_dops = sorted([k for k in exec_time_map.keys() if isinstance(k, (int, float))])
    except TypeError:
        print(f"警告 (select_optimal_dop): exec_time_map 的键包含不可排序类型。 {exec_time_map.keys()}")
        return default_dop # 返回默认

    if not sorted_dops: return default_dop

    # 过滤阶段
    filtered_dops = [sorted_dops[0]]
    for dop in sorted_dops[1:]:
        prev_dop = filtered_dops[-1]
        # 获取时间值，处理 None 或非数值
        time_prev = exec_time_map.get(prev_dop)
        time_curr = exec_time_map.get(dop)
        if time_prev is None or time_curr is None or not isinstance(time_prev, (int, float)) or not isinstance(time_curr, (int, float)):
             continue # 跳过无效数据点

        if time_curr <= 0: continue # 跳过非正时间

        improvement_ratio = (time_prev - time_curr) / time_prev if time_prev > 0 else float('inf')
        absolute_reduction = time_prev - time_curr
        diff = dop - prev_dop
        factor = 1 + math.log2(diff) if diff > 0 else 1
        adjusted_improvement = improvement_ratio / factor if factor > 0 else float('inf')

        dynamic_min_improvement = min_improvement_ratio # 保持原始逻辑，没有动态调整
        if absolute_reduction >= min_reduction_threshold and adjusted_improvement >= dynamic_min_improvement:
            filtered_dops.append(dop)

    # 返回过滤后的最后一个 DOP
    return filtered_dops[-1]


def select_optimal_dops_from_prediction_file(csv_file, output_file=None, **kwargs):
    """从预测结果 CSV 选择最优 DOP (直接搬运 dop_utils 逻辑)"""
    print(f"开始从预测文件选择最优 DOP: {csv_file}")
    try:
        df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
    except FileNotFoundError:
        print(f"错误: 预测文件未找到: {csv_file}")
        return {}
    except Exception as e:
        print(f"错误: 读取预测文件时出错: {e}")
        return {}

    # 检查必需的列是否存在
    required_cols = ['query_id', 'dop', 'predicted_time']
    if not all(col in df.columns for col in required_cols):
         print(f"错误: 预测文件缺少必需列 ({required_cols})")
         return {}

    optimal_dops = {}
    # 按 query_id 分组
    groups = df.groupby('query_id')
    for query_id, group in groups:
        # 构造 {dop: predicted_time} 字典 (确保 dop 和 time 是数值)
        dop_dict = {}
        for _, row in group.iterrows():
            dop = pd.to_numeric(row['dop'], errors='coerce')
            pred_time = pd.to_numeric(row['predicted_time'], errors='coerce')
            if pd.notna(dop) and pd.notna(pred_time):
                 dop_dict[int(dop)] = pred_time # 转 int 作为 key

        if dop_dict:
            # 调用 select_optimal_dop 选择最优 DOP
            optimal_dop = select_optimal_dop(dop_dict, **kwargs) # 传递其他参数
            optimal_dops[query_id] = optimal_dop
        else:
             print(f"警告: Query {query_id} 没有有效的 (dop, predicted_time) 数据。")


    # 保存结果到 CSV (如果指定了输出文件)
    if output_file is not None:
        if optimal_dops: # 只有在有结果时才保存
            out_df = pd.DataFrame(list(optimal_dops.items()), columns=['query_id', 'optimal_dop'])
            try:
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                     os.makedirs(output_dir)
                out_df.to_csv(output_file, index=False, sep=';')
                print(f"最优查询 DOP 结果已保存至 {output_file}")
            except Exception as e:
                 print(f"错误: 保存最优 DOP 结果时出错: {e}")
        else:
             print("没有最优 DOP 结果可保存。")

    return optimal_dops


# 这个函数可以放在这里，也可以放在 result_processor.py
def update_nodes_with_optimal_dop(query_thread_blocks):
    """
    遍历线程块，更新每个节点的时间为最优DOP下的值。
    (逻辑来自原 dop_choosen.py 的同名函数)

    Args:
        query_thread_blocks (dict): {query_id: {thread_id: ThreadBlock}}

    Returns:
        dict: 更新后的 query_thread_blocks (原地更新)
    """
    print("开始根据最优 DOP 更新节点执行时间...")
    updated_node_count = 0
    for query_id, tb_dict in query_thread_blocks.items():
        if not tb_dict: continue
        # print(f"  Updating Query ID: {query_id}")
        for tb_id, tb in tb_dict.items():
            if not isinstance(tb, ThreadBlock) or tb.optimal_dop is None:
                 # print(f"    Skipping Thread {tb_id}: Not a ThreadBlock or optimal_dop is None.")
                 # 如果没有最优 DOP，使用默认值？
                 opt_dop = default_dop # 假设从 config 导入了 default_dop
            else:
                 opt_dop = tb.optimal_dop

            # 遍历线程块内的所有节点
            for node in tb.nodes:
                if not isinstance(node, PlanNode): continue

                # 从节点的 exec_time_map 获取最优 DOP 对应的“真实”时间（可能是插值的）
                # 如果找不到，使用节点的原始预测时间或者一个小的默认值？
                # 原始逻辑似乎是直接用 exec_time_map 的值更新 pred_execution_time
                optimal_time = node.exec_time_map.get(opt_dop)

                if optimal_time is not None:
                    node.pred_execution_time = optimal_time
                    # print(f"    Node {node.plan_id}: Updated pred_execution_time to {optimal_time} (for DOP {opt_dop})")
                    updated_node_count += 1
                else:
                    # 如果 exec_time_map 中没有最优 DOP 的时间，如何处理？
                    # 1. 使用原始预测时间？ node.pred_execution_time = node.pred_execution_time
                    # 2. 使用插值或预测函数重新计算？ node.pred_execution_time = node.compute_pred_exec(opt_dop)
                    # 3. 使用默认值？ node.pred_execution_time = 0.05
                    # 原始代码似乎是直接用 exec_time_map 的值，如果 key 不存在则不更新？
                    # 我们暂时也保持不更新的行为，但打印警告
                    # print(f"    警告: Node {node.plan_id} (DOP {opt_dop}): 在 exec_time_map 中未找到对应时间，未更新 pred_execution_time。")
                    pass # 保持节点当前的 pred_execution_time

    print(f"... {updated_node_count} 个节点的预测执行时间已根据最优 DOP 更新。")
    return query_thread_blocks # 返回更新后的对象 (虽然是原地更新)

# --- build_query_exec_time_map 和 select_optimal_query_dop_from_tru ---
# 这两个函数处理的是 "真实" (tru) 运行数据，如果优化流程只基于预测，可能不需要它们
# 如果需要，也从 dop_utils.py 搬运过来，注意修改路径和错误处理

# def build_query_exec_time_map(csv_file):
#     # ... (搬运逻辑) ...

# def select_optimal_query_dop_from_tru(csv_file, output_file=None, **kwargs):
#     # ... (搬运逻辑) ...


# ==============================================================================
# 结束搬运 dop_utils.py 和 dop_choosen.py 中与结果处理相关的函数
# ==============================================================================
# --- 新增：从JSON提取线程数并保存为CSV的函数 ---
def save_thread_counts_from_json(json_path, output_csv_path):
    """
    读取优化结果JSON文件，提取每个查询的ID和总线程数，
    并保存到一个新的CSV文件中。

    Args:
        json_path (str): 输入的 query_details_optimized.json 文件路径。
        output_csv_path (str): 输出的CSV文件路径。
    """
    if not os.path.exists(json_path):
        print(f"警告: JSON文件未找到，无法提取线程数: {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        thread_count_data = []
        for query in data.get('queries', []):
            query_id = query.get('query_id')
            # 使用 .get() 并提供默认值0，以防JSON中没有这个字段
            thread_count = query.get('query_total_threads', 0)
            if query_id is not None:
                thread_count_data.append([query_id, thread_count])

        if thread_count_data:
            # 创建包含表头的DataFrame
            df = pd.DataFrame(thread_count_data, columns=['query_id', 'thread_count'])
            # 确保目录存在
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            # 保存为CSV
            df.to_csv(output_csv_path, index=False, sep=';')
            print(f"查询线程数已成功保存到: {output_csv_path}")
        else:
            print("警告: JSON文件中没有找到任何查询数据来提取线程数。")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"解析JSON或提取线程数时出错: {e}")
    except Exception as e:
        print(f"保存线程数CSV时发生未知错误: {e}")