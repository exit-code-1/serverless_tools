# 文件路径: optimization/optimizer.py

import copy
import math
import time
import pandas as pd
import numpy as np # 如果需要处理 numpy 数据

# --- 导入重构后的模块 ---
from core.onnx_manager import ONNXModelManager # 需要初始化管理器
from core.plan_node import PlanNode # 虽然不直接创建，但类型提示可能需要
from core.thread_block import ThreadBlock # 需要调用 choose_optimal_dop

from config.structure_config import default_dop # 可能需要默认 DOP

# 从同包的其他模块导入函数
from .tree_updater import build_query_trees, update_dop_information
from .threading_utils import update_thread_blocks, generate_aligned_dop_configurations, calculate_query_execution_time # 需要收集节点
# result_processor 稍后创建，先注释掉导入
from .result_processor import write_query_details_to_file, update_nodes_with_optimal_dop
# --- 结束导入 ---


def compute_optimal_dops_for_query(thread_blocks_for_query, **kwargs):
    """
    为单个查询的所有线程块计算最优 DOP。
    (逻辑来自原 dop_choosen.py 的 compute_optimal_dops_on_thread_blocks)

    Args:
        thread_blocks_for_query (dict): {thread_id: ThreadBlock}
        **kwargs: 传递给 choose_optimal_dop 的其他参数 (如阈值等)

    Returns:
        dict: {thread_id: {'optimal_dop': X, 'pred_time': Y}}
    """
    optimal_dops_info = {}
    if not thread_blocks_for_query:
        return optimal_dops_info

    # 提取 choose_optimal_dop 的参数，设置默认值
    child_time_default = kwargs.get('child_time_default', 1e-6) # 原始默认值
    min_improvement_ratio = kwargs.get('min_improvement_ratio', 0.2) # 保持原始默认值
    min_reduction_threshold = kwargs.get('min_reduction_threshold', 200) # 新增参数，来自ThreadBlock
    interval_tolerance = kwargs.get('interval_tolerance', 0.3)  # Tolerance for interval matching

    # Find root thread blocks (topmost pipelines) - those with no parent
    # In the pipeline context: root = topmost (gather operation, DOP=1)
    # Build parent-child mapping
    parent_map = {}
    for tid, tb in thread_blocks_for_query.items():
        if not isinstance(tb, ThreadBlock):
            continue
        for child_id in tb.child_thread_ids:
            if child_id in thread_blocks_for_query:
                parent_map[child_id] = tid
    
    # Find root thread blocks (those not in parent_map)
    root_thread_ids = [tid for tid in thread_blocks_for_query.keys() 
                      if tid not in parent_map]
    
    # Reset visit flags
    for tb in thread_blocks_for_query.values():
        if isinstance(tb, ThreadBlock):
            tb.visit = False
    
    # Process from topmost (root) pipelines downward (top-down approach)
    # Paper: Starting from top-level pipeline, use its consumption rate 
    #        as reference to determine feasible DOP range of downstream pipeline
    for root_tid in root_thread_ids:
        root_tb = thread_blocks_for_query[root_tid]
        if not isinstance(root_tb, ThreadBlock):
            continue
        if root_tb.visit:
            continue
        
        # Start top-down propagation from root pipeline
        root_tb.choose_optimal_dop_top_down(
            thread_blocks_for_query,
            parent_production_rate=None,  # Root has no parent
            min_improvement_ratio=min_improvement_ratio,
            min_reduction_threshold=min_reduction_threshold,
            interval_tolerance=interval_tolerance
        )

    return optimal_dops_info


def analyze_dop_boundary_selection(all_thread_blocks):
    """
    Analyze how many thread blocks selected boundary DOPs vs middle DOPs
    
    Args:
        all_thread_blocks: dict, {query_id: {thread_id: ThreadBlock}}
        
    Returns:
        dict: Statistics about boundary vs middle selections
    """
    stats = {
        'boundary_selections': 0,  # Selected min or max of candidate range
        'middle_selections': 0,    # Selected value in the middle
        'left_boundary': 0,        # Selected minimum candidate DOP
        'right_boundary': 0,       # Selected maximum candidate DOP
        'total_thread_blocks': 0,
        'details': []              # Detailed information for each selection
    }
    
    for query_id, thread_blocks in all_thread_blocks.items():
        for tid, tb in thread_blocks.items():
            if not isinstance(tb, ThreadBlock):
                continue
                
            optimal_dop = tb.optimal_dop
            if optimal_dop is None:
                continue
                
            candidates = getattr(tb, 'candidate_optimal_dops', [])
            if not candidates:
                # If no candidates, skip (might be root pipeline with DOP=1)
                continue
            
            sorted_candidates = sorted(candidates)
            min_candidate = sorted_candidates[0]
            max_candidate = sorted_candidates[-1]
            
            stats['total_thread_blocks'] += 1
            
            # Determine if selected DOP is boundary or middle
            is_boundary = False
            selection_type = None
            
            if optimal_dop == min_candidate:
                stats['left_boundary'] += 1
                stats['boundary_selections'] += 1
                is_boundary = True
                selection_type = 'left_boundary'
            elif optimal_dop == max_candidate:
                stats['right_boundary'] += 1
                stats['boundary_selections'] += 1
                is_boundary = True
                selection_type = 'right_boundary'
            else:
                stats['middle_selections'] += 1
                selection_type = 'middle'
            
            # Store detail
            stats['details'].append({
                'query_id': query_id,
                'thread_id': tid,
                'optimal_dop': optimal_dop,
                'min_candidate': min_candidate,
                'max_candidate': max_candidate,
                'candidate_range_size': len(sorted_candidates),
                'selection_type': selection_type
            })
    
    return stats


# --- (可选) 迁移 update_nodes_with_optimal_dop 函数 ---



# --- 主优化流程函数 ---
def run_dop_optimization(df_plans_all_dops, onnx_manager: ONNXModelManager,
                         output_json_path, # <--- 添加 JSON 输出路径参数
                         base_dop=64,use_estimates=False, # <-- 新增参数
                         **kwargs):
    """
    执行端到端的 DOP 优化流程。

    Args:
        df_plans_all_dops (pd.DataFrame): 包含所有 DOP 运行数据的 DataFrame。
        onnx_manager (ONNXModelManager): ONNX 模型管理器。
        output_json_path (str): 输出详细结果 JSON 文件的路径。 # <--- 添加参数说明
        base_dop (int): 用于构建基准树的 query_dop 值。
        **kwargs: 其他传递给 choose_optimal_dop 的参数。

    Returns:
        dict: 包含优化结果的字典。
    """
    print("="*30)
    print("开始 DOP 优化流程")
    print("="*30)
    start_time = time.time()
    # --- 1. 计时器和日志初始化 ---
    total_start_time = time.time()
    timing_info = {
        'data_prep_and_prediction_s': 0,
        'path_generation_and_evaluation_s': 0, # 这将成为所有查询决策时间的总和
        'per_query_decision_s': {}, # <-- 新增：用于存储每个查询的决策时间
        'avg_query_decision_s': 0,
        'total_optimization_s': 0
    }
    # 0. 过滤掉 query_dop > 96 的行
    df_plans_all_dops = df_plans_all_dops[df_plans_all_dops['query_dop'] <= 96]
    
    # 1. 构建基准查询树 (不变)
    print(f"\n步骤 1: 构建基准查询树 (base_dop={base_dop})...")
    if use_estimates:
        print("!! 注意：正在以基数估计模拟模式运行 !!")
    df_plans_base = df_plans_all_dops[df_plans_all_dops['query_dop'] == base_dop].copy()
    if df_plans_base.empty:
        print(f"错误: 未找到基准 DOP={base_dop} 的数据。")
        return None
    query_trees = build_query_trees(df_plans_base, onnx_manager, use_estimates=use_estimates) # <-- 传递
    print(f"基准查询树构建完成，共 {len(query_trees)} 个查询。")

    # 2. 更新节点信息 (不变)
    print("\n步骤 2: 使用所有 DOP 数据更新节点信息...")
    query_trees = update_dop_information(df_plans_all_dops, query_trees, onnx_manager, base_dop)
    print("节点信息更新完成。")

    print("\n步骤 3&4: 划分线程块，计算最优DOP并评估路径...")
    all_thread_blocks = {}
    all_optimal_dops = {}
    processed_queries_for_threading = 0
    
    for query_id, base_nodes in query_trees.items():
        if not base_nodes:
            continue
        # 划分线程块
        thread_blocks_for_query = update_thread_blocks(base_nodes)
        if not thread_blocks_for_query:
            continue

        all_thread_blocks[query_id] = thread_blocks_for_query
        processed_queries_for_threading += 1
        if query_id == 97:
            print()
        # 计算每个线程块的候选 DOP（写入 candidate_optimal_dops，不影响 optimal_dop）
        compute_optimal_dops_for_query(thread_blocks_for_query, **kwargs)

        # Check if we should use MOO optimization
        use_moo = kwargs.get('use_moo', False)
        
        if use_moo:
            # Use MOO optimizer for DOP selection
            print(f"正在使用MOO优化器为 query {query_id} 选择最优DOP配置...")
            try:
                from .moo_dop_optimizer import optimize_thread_block_dops_with_moo
                
                moo_population = kwargs.get('moo_population_size', 10)
                moo_generations = kwargs.get('moo_generations', 20)
                moo_weight_latency = kwargs.get('moo_weight_latency', 0.8)
                moo_weight_cost = kwargs.get('moo_weight_cost', 0.2)
                # Default to discrete mode for faster execution (continuous requires more iterations)
                use_continuous_dop = kwargs.get('use_continuous_dop', False)
                
                optimal_dop_config, moo_execution_time = optimize_thread_block_dops_with_moo(
                    thread_blocks=thread_blocks_for_query,
                    population_size=moo_population,
                    generations=moo_generations,
                    weight_latency=moo_weight_latency,
                    weight_cost=moo_weight_cost,
                    use_continuous_dop=use_continuous_dop
                )
                
                # Record MOO execution time
                timing_info['per_query_decision_s'][query_id] = moo_execution_time
                print(f"  - Query {query_id} NSGA-II算法耗时: {moo_execution_time:.4f} 秒")
                
                # Update thread blocks with MOO results
                for tid, dop in optimal_dop_config.items():
                    tb = thread_blocks_for_query[tid]
                    tb.optimal_dop = dop
                    tb.pred_time = tb.pred_dop_exec_time.get(dop)
                
            except Exception as e:
                print(f"  MOO optimization failed: {e}, falling back to path enumeration")
                use_moo = False
        
        if not use_moo:
            # Original path enumeration approach
            print(f"正在生成候选路径并评估 query {query_id} 的最优路径...")
            query_decision_start_time = time.time()  # <-- 开始计时：只统计路径枚举时间
            candidate_paths = generate_aligned_dop_configurations(thread_blocks_for_query)
            best_path_info = evaluate_candidate_paths(
                thread_blocks_for_query, 
                candidate_paths,
                cost_func=cost_func,
                allowed_time_margin=0.4
            )
            query_decision_end_time = time.time()  # <-- 结束计时：路径枚举完成
            decision_duration = query_decision_end_time - query_decision_start_time
            timing_info['per_query_decision_s'][query_id] = decision_duration
            print(f"  - Query {query_id} 路径枚举耗时: {decision_duration:.4f} 秒")

            if best_path_info is not None:
                # 用最优路径更新每个线程块的 DOP
                for tid, dop in best_path_info['path'].items():
                    tb = thread_blocks_for_query[tid]
                    tb.optimal_dop = dop
                    tb.pred_time = tb.pred_dop_exec_time.get(dop)

        # After MOO/path enumeration, apply child DOP adjustments
        # to reduce redistribution overhead
        for tid in sorted(thread_blocks_for_query.keys()):
            tb = thread_blocks_for_query[tid]
            if tb.child_thread_ids:
                tb._adjust_child_dops_for_redistribution(
                    thread_blocks_for_query,
                    min_improvement_ratio=kwargs.get('min_improvement_ratio', 0.2),
                    min_reduction_threshold=kwargs.get('min_reduction_threshold', 200)
                )
        
        # 保存最终使用的 DOP 结果
        final_dops = {
            tid: {
                'optimal_dop': tb.optimal_dop,
                'pred_time': tb.pred_time,
                'candidates': tb.candidate_optimal_dops
            } for tid, tb in thread_blocks_for_query.items()
        }
        all_optimal_dops[query_id] = final_dops
        
        print(f"Query {query_id} 的DOP配置完成")
    

    # --- 第二阶段计时结束 ---
    print(f"线程块划分和最优DOP计算完成, 共处理 {processed_queries_for_threading} 个查询。")




    # 5. (可选) 根据最优 DOP 更新节点执行时间 (不变)
    # print("\n步骤 5: (可选) 更新节点预测时间以反映最优 DOP...")
    # all_thread_blocks = update_nodes_with_optimal_dop(all_thread_blocks)
    # print("节点时间更新完成。")
    print("\n步骤 5: 更新节点预测时间以反映最优 DOP...")
    all_thread_blocks = update_nodes_with_optimal_dop(all_thread_blocks) # 调用导入的函数
    print("节点时间更新完成。")

    # --- 步骤 6: 统计边界选择情况 ---
    print(f"\n步骤 6: 统计DOP边界选择情况...")
    boundary_stats = analyze_dop_boundary_selection(all_thread_blocks)
    
    total = boundary_stats['total_thread_blocks']
    if total > 0:
        boundary_pct = (boundary_stats['boundary_selections'] / total) * 100
        middle_pct = (boundary_stats['middle_selections'] / total) * 100
        left_pct = (boundary_stats['left_boundary'] / total) * 100
        right_pct = (boundary_stats['right_boundary'] / total) * 100
        
        print(f"\n{'='*50}")
        print(f"DOP边界选择统计:")
        print(f"{'='*50}")
        print(f"总线程块数: {total}")
        print(f"选择边界值: {boundary_stats['boundary_selections']} ({boundary_pct:.2f}%)")
        print(f"  - 左边界(min): {boundary_stats['left_boundary']} ({left_pct:.2f}%)")
        print(f"  - 右边界(max): {boundary_stats['right_boundary']} ({right_pct:.2f}%)")
        print(f"选择中间值: {boundary_stats['middle_selections']} ({middle_pct:.2f}%)")
        print(f"{'='*50}\n")
    else:
        print("  警告: 没有找到可统计的线程块")
    
    # --- 步骤 7: 调用 result_processor 输出结果 ---
    print(f"\n步骤 7: 输出详细结果到 {output_json_path}...")
    try:
        # 需要 query_trees (用于获取所有节点?) 和 all_thread_blocks
        # 注意 write_query_details_to_file 可能需要调整以正确获取所有节点
        write_query_details_to_file(query_trees, all_thread_blocks, output_json_path)
    except Exception as e:
        print(f"写入 JSON 文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
    # --- 结束调用 ---

    end_time = time.time()
    print("\n" + "="*30)
    print(f"DOP 优化流程完成，总耗时: {end_time - start_time:.2f} 秒")
    print("="*30)

    return {
        'query_trees': query_trees,
        'thread_blocks': all_thread_blocks,
        'optimal_dops': all_optimal_dops,
        'timing_info': timing_info, # <-- 将计时信息返回
        'boundary_stats': boundary_stats  # <-- 添加边界选择统计信息
    }
    
def cost_func(path, exec_time):
    total_dop = sum(path.values())
    return total_dop * exec_time
    
def evaluate_candidate_paths(all_thread_blocks, candidate_paths, cost_func, allowed_time_margin=0.5):
    """
    从多个候选路径中选出最优路径（在容忍时间范围内选成本最低的）。

    Args:
        all_thread_blocks (dict): 所有查询的线程块，结构为 {query_id: {thread_id: ThreadBlock}}。
        candidate_paths (list): 每个候选路径是一个 {thread_id: dop} 的字典。
        cost_func (Callable): 给定一个路径，返回其总代价（单位可为美元、资源分数等）。
        allowed_time_margin (float): 容忍时间劣化比例（例如 0.1 表示容忍比基准多 10% 的时间）。

    Returns:
        dict: 最优路径（{thread_id: dop}）。
    """

    def update_thread_blocks_with_path(tb_dict, dop_path):
        # 更新 thread block 中与 dop 相关的执行时间
        for tb_id, dop in dop_path.items():
            tb = tb_dict[tb_id]
            tb.pred_time = tb.pred_dop_exec_time.get(dop, float('inf'))
            tb.optimal_dop = dop
            for node in tb.nodes:
                node.pred_execution_time = node.compute_pred_exec(dop)

    results = []

    for path in candidate_paths:
        # 创建一个 deep copy 的 TB，用于评估不影响原结构
        update_thread_blocks_with_path(all_thread_blocks, path)
        exec_time = calculate_query_execution_time(thread_blocks=all_thread_blocks)
        cost = cost_func(path, exec_time)
        total_dop = sum(path.values())
        results.append({
            'path': path,
            'exec_time': exec_time,
            'cost': cost,
            'total_dop': total_dop
        })

    # 找到总 DOP 最大的配置作为 baseline
    baseline = max(results, key=lambda r: r['total_dop'])
    baseline_exec_time = baseline['exec_time']

    # 过滤在时间限制范围内的配置
    valid_paths = [r for r in results if r['exec_time'] <= baseline_exec_time * (1 + allowed_time_margin)]

    if not valid_paths:
        print("警告：没有路径满足时间限制，退化为 baseline 选择")
        best = baseline
    else:
        best = min(valid_paths, key=lambda r: r['cost'])

    # 更新 thread block 的最优解
    update_thread_blocks_with_path(all_thread_blocks, best['path'])

    return best



def run_query_dop_optimization(csv_file, df_plans_all_dops, onnx_manager: ONNXModelManager,
                         output_json_path, # <--- 添加 JSON 输出路径参数
                         algorithm, #全都一个dop
                         base_dop=64, **kwargs):
    """
    执行端到端的 DOP 优化流程。

    Args:
        df_plans_all_dops (pd.DataFrame): 包含所有 DOP 运行数据的 DataFrame。
        onnx_manager (ONNXModelManager): ONNX 模型管理器。
        output_json_path (str): 输出详细结果 JSON 文件的路径。 # <--- 添加参数说明
        base_dop (int): 用于构建基准树的 query_dop 值。
        **kwargs: 其他传递给 choose_optimal_dop 的参数。

    Returns:
        dict: 包含优化结果的字典。
    """
    print("="*30)
    print("开始 DOP 优化流程")
    print("="*30)
    start_time = time.time()

    # --- 1. 初始化计时器 ---
    total_start_time = time.time()
    timing_info = {
        'dop_selection_s': 0, # <-- 使用新的键名
    }

    # 0. 过滤掉 query_dop > 96 的行
    df_plans_all_dops = df_plans_all_dops[df_plans_all_dops['query_dop'] <= 96]
    
    # 1. 构建基准查询树 (不变)
    print(f"\n步骤 1: 构建基准查询树 (base_dop={base_dop})...")
    df_plans_base = df_plans_all_dops[df_plans_all_dops['query_dop'] == base_dop].copy()
    if df_plans_base.empty:
        print(f"错误: 未找到基准 DOP={base_dop} 的数据。")
        return None
    query_trees = build_query_trees(df_plans_base, onnx_manager)
    print(f"基准查询树构建完成，共 {len(query_trees)} 个查询。")

    # 2. 更新节点信息 (不变)
    print("\n步骤 2: 使用所有 DOP 数据更新节点信息...")
    query_trees = update_dop_information(df_plans_all_dops, query_trees, onnx_manager, base_dop)
    print("节点信息更新完成。")

    print("\n步骤 3&4: 划分线程块，计算最优DOP并评估路径...")
    all_thread_blocks = {}
    all_optimal_dops = {}
    processed_queries_for_threading = 0

    # --- 3. 对第二阶段（DOP选择）计时 ---
    dop_selection_start_time = time.time()

    optimal_dops = {}
    if algorithm == 'pipeline_query':
        df = pd.read_csv(csv_file, sep=';')
        optimal_dops = dict(zip(df['query_id'], df['optimal_dop']))
    elif algorithm == 'ppm':
        # PPM: use curve fitting model to select optimal DOP
        # csv_file should contain: plan_csv path, ppm_model_dir, model_type
        # Extract from kwargs or use defaults
        plan_csv = kwargs.get('plan_csv', csv_file)
        ppm_model_dir = kwargs.get('ppm_model_dir')
        model_type = kwargs.get('ppm_model_type', 'NN')
        use_estimates = kwargs.get('use_estimates', False)
        
        if ppm_model_dir is None:
            raise ValueError("PPM optimization requires 'ppm_model_dir' parameter")
        
        optimal_dops = select_optimal_dops_with_ppm_model(
            plan_csv=plan_csv,
            ppm_model_dir=ppm_model_dir,
            model_type=model_type,
            use_estimates=use_estimates
        )
    else:
        optimal_dops = select_optimal_dops_from_prediction_file(csv_file, kwargs)

    
    for query_id, base_nodes in query_trees.items():
        if not base_nodes:
            continue
        # 划分线程块
        thread_blocks_for_query = update_thread_blocks(base_nodes)
        if not thread_blocks_for_query:
            continue

        all_thread_blocks[query_id] = thread_blocks_for_query
        processed_queries_for_threading += 1

            # 按线程ID顺序处理可能有助于调试或理解依赖关系（如果存在）
        sorted_thread_ids = sorted(thread_blocks_for_query.keys())
        
        optimal_dop = optimal_dops[query_id]
        if algorithm == 'tru':
            optimal_dop = 64

        for tid in sorted_thread_ids:
            tb = thread_blocks_for_query[tid]
            if not isinstance(tb, ThreadBlock): continue
            
            # Check if this thread block contains any root node (parent_node is None)
            has_root_node = False
            for node in tb.nodes:
                if node.parent_node is None:
                    has_root_node = True
                    break
            
            # Root pipeline thread blocks should always have DOP=1
            if has_root_node:
                tb.optimal_dop = 1
            else:
                tb.optimal_dop = optimal_dop

    dop_selection_end_time = time.time()
    timing_info['dop_selection_s'] = dop_selection_end_time - dop_selection_start_time
    print(f"所有查询的DOP选择阶段总耗时: {timing_info['dop_selection_s']:.4f} 秒")
    # --- 第二阶段计时结束 ---

    # 5. (可选) 根据最优 DOP 更新节点执行时间 (不变)
    # print("\n步骤 5: (可选) 更新节点预测时间以反映最优 DOP...")
    # all_thread_blocks = update_nodes_with_optimal_dop(all_thread_blocks)
    # print("节点时间更新完成。")
    print("\n步骤 5: 更新节点预测时间以反映最优 DOP...")
    all_thread_blocks = update_nodes_with_optimal_dop(all_thread_blocks) # 调用导入的函数
    print("节点时间更新完成。")

    # --- 步骤 6: 调用 result_processor 输出结果 ---
    print(f"\n步骤 6: 输出详细结果到 {output_json_path}...")
    try:
        # 需要 query_trees (用于获取所有节点?) 和 all_thread_blocks
        # 注意 write_query_details_to_file 可能需要调整以正确获取所有节点
        write_query_details_to_file(query_trees, all_thread_blocks, output_json_path)
    except Exception as e:
        print(f"写入 JSON 文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
    # --- 结束调用 ---

    end_time = time.time()
    print("\n" + "="*30)
    print(f"DOP 优化流程完成，总耗时: {end_time - start_time:.2f} 秒")
    print("="*30)

    return {
        'query_trees': query_trees,
        'thread_blocks': all_thread_blocks,
        'optimal_dops': all_optimal_dops,
        'timing_info': timing_info # <-- 将计时信息返回
    }


def select_query_optimal_dop(exec_time_map, min_improvement_ratio=2,  min_reduction_threshold=400):
    """
    选择最优的并行度 DOP:
    1. 如果执行时间的下降幅度小于 min_gain_ms,则不增加 DOP。
    2. 如果执行时间的下降率低于 time_threshold,则不增加 DOP。
    
    :param exec_time_map: {dop: execution_time} 的字典
    :param time_threshold: 下降率阈值（默认 10)
    :param min_gain_ms: 最小时间收益（默认 100ms)
    :return: 最优的 DOP
    """
    sorted_dops = sorted(exec_time_map.keys())  # 先按照 DOP 排序
    best_dop = sorted_dops[0]  # 先选择最低的并行度
    if len(sorted_dops) == 1:
        return sorted_dops[0] 
    filtered_dops = [sorted_dops[0]]
    for dop in sorted_dops[1:]:
        prev_dop = filtered_dops[-1]
        time_prev = exec_time_map.get(prev_dop, float('inf'))
        time_curr = exec_time_map.get(dop, float('inf'))
        if time_curr <= 0:
            continue

        # 计算改善率和减少量
        improvement_ratio = (time_prev - time_curr) / time_prev
        absolute_reduction = time_prev - time_curr
        diff = dop - prev_dop  # 必然为正
        factor = 1 + math.log2(diff) if diff > 0 else 1
        adjusted_improvement = improvement_ratio / factor
        if absolute_reduction < min_reduction_threshold:
            continue
        else:
            dynamic_min_improvement = min_improvement_ratio / math.log10(absolute_reduction)
        if adjusted_improvement >= dynamic_min_improvement :
            best_dop = dop
        else:
            continue
    return best_dop

def select_optimal_dops_from_prediction_file(csv_file, time_threshold=0.1, min_gain_ms=200, output_file=None):
    """
    端到端函数：从包含预测结果的 CSV 文件中读取数据，
    该 CSV 文件需包含字段:query_id;dop;predicted_time;actual_time;predicted_memory;actual_memory;q_error_time;q_error_memory。
    对每个 query,根据 predicted_time 选择最优 DOP(单位均为毫秒)。
    
    :param csv_file: 输入 CSV 文件路径
    :param time_threshold: 下降率阈值（默认 0.1)
    :param min_gain_ms: 最小时间收益（默认 100ms)
    :param output_file: 输出文件路径（如果不为 None,则保存结果到 CSV)
    :return: dict, 格式 {query_id: optimal_dop, ...}
    """
    df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
    df = df[df['dop'] <= 96]
    
    optimal_dops = {}
    # 按 query_id 分组，每组包含该 query 不同 DOP 下的预测结果
    groups = df.groupby('query_id')
    for query_id, group in groups:
        # 构造 {dop: predicted_time} 字典
        dop_dict = dict(zip(group['dop'], group['predicted_time']))
        # 复用 select_optimal_dop 函数
        optimal_dop = select_query_optimal_dop(dop_dict)
        optimal_dops[query_id] = optimal_dop

    # 如果指定输出文件，则保存结果到 CSV
    if output_file is not None:
        out_df = pd.DataFrame(list(optimal_dops.items()), columns=['query_id', 'optimal_dop'])
        out_df.to_csv(output_file, index=False, sep=';')
        print(f"最优查询 DOP 结果已保存至 {output_file}")
    
    return optimal_dops


def select_optimal_dops_with_ppm_model(plan_csv, ppm_model_dir, model_type='NN', 
                                        use_estimates=False, candidate_dops=None):
    """
    Use PPM curve fitting model to select optimal DOP for each query
    
    Args:
        plan_csv: Path to plan_info.csv
        ppm_model_dir: Directory containing PPM ONNX models (execution_time_model.onnx, memory_usage_model.onnx)
        model_type: 'NN' or 'GNN'
        use_estimates: Whether to use estimated cardinalities
        candidate_dops: List of candidate DOPs to evaluate (default: [1, 2, 4, 8, 16, 32, 64, 96])
        
    Returns:
        dict: {query_id: optimal_dop}
    """
    import os
    import onnxruntime as ort
    
    if candidate_dops is None:
        candidate_dops = [1, 2, 4, 8, 16, 32, 64, 96]
    
    print(f"使用PPM-{model_type}模型进行DOP优化...")
    print(f"模型目录: {ppm_model_dir}")
    print(f"候选DOP: {candidate_dops}")
    
    # Load ONNX models
    exec_model_path = os.path.join(ppm_model_dir, 'execution_time_model.onnx')
    mem_model_path = os.path.join(ppm_model_dir, 'memory_usage_model.onnx')
    
    if not os.path.exists(exec_model_path):
        raise FileNotFoundError(f"PPM execution time model not found: {exec_model_path}")
    if not os.path.exists(mem_model_path):
        raise FileNotFoundError(f"PPM memory model not found: {mem_model_path}")
    
    print(f"加载ONNX模型: {exec_model_path}")
    exec_session = ort.InferenceSession(exec_model_path)
    print(f"加载ONNX模型: {mem_model_path}")
    mem_session = ort.InferenceSession(mem_model_path)
    
    optimal_dops = {}
    
    if model_type == 'NN':
        # NN model: uses aggregated query features
        from training.PPM.featurelize import process_query_features
        
        print("构建查询特征...")
        query_features = process_query_features(plan_csv, use_estimates=use_estimates)
        
        print(f"对 {len(query_features)} 个查询进行DOP优化...")
        
        for (query_id, query_dop), feature_vector in query_features.items():
            # Use ONNX model to predict curve parameters
            X_input = feature_vector.reshape(1, -1).astype(np.float32)
            
            # Predict execution time curve parameters (a, b, c)
            exec_params = exec_session.run(None, {'input': X_input})[0][0]
            a_exec, b_exec, c_exec = exec_params[0], exec_params[1], exec_params[2]
            
            # Calculate execution time for each candidate DOP using curve formula
            # Formula: time = b / (dop ** a) + c
            dop_times = {}
            for dop in candidate_dops:
                pred_time = max(b_exec / (dop ** a_exec) + c_exec, 1e-6)  # Avoid negative or zero
                dop_times[dop] = pred_time
            
            # Select optimal DOP using existing selection logic
            optimal_dop = select_query_optimal_dop(dop_times)
            optimal_dops[query_id] = optimal_dop
            
        print(f"完成DOP优化，共处理 {len(optimal_dops)} 个查询")
        
    elif model_type == 'GNN':
        # GNN model: uses graph structure
        from training.PPM.GNN_train import load_graph_data
        from torch_geometric.loader import DataLoader as PyGDataLoader
        import torch
        
        print("构建查询图数据...")
        # For GNN, we need query_info file - extract from plan_csv directory
        query_info_csv = plan_csv.replace('plan_info.csv', 'query_info.csv')
        if not os.path.exists(query_info_csv):
            raise FileNotFoundError(f"Query info file not found: {query_info_csv}")
        
        graph_data_list = load_graph_data(plan_csv, query_info_csv, use_estimates=use_estimates)
        
        print(f"对 {len(graph_data_list)} 个查询图进行DOP优化...")
        
        # Process each query graph
        for data in graph_data_list:
            query_id = data.query_id
            
            # Prepare ONNX inputs
            x_np = data.x.numpy().astype(np.float32)
            edge_index_np = data.edge_index.numpy().astype(np.int64)
            batch_np = data.batch.numpy().astype(np.int64)
            
            ort_inputs = {
                "x": x_np,
                "edge_index": edge_index_np,
                "batch": batch_np
            }
            
            # Predict curve parameters (a, b, c)
            exec_params = exec_session.run(["output"], ort_inputs)[0][0]
            a_exec, b_exec, c_exec = exec_params[0], exec_params[1], exec_params[2]
            
            # Calculate execution time for each candidate DOP
            # Formula: time = b / (dop ** a) + c
            dop_times = {}
            for dop in candidate_dops:
                pred_time = max(b_exec / (dop ** a_exec) + c_exec, 1e-6)
                dop_times[dop] = pred_time
            
            # Select optimal DOP
            optimal_dop = select_query_optimal_dop(dop_times)
            optimal_dops[query_id] = optimal_dop
        
        print(f"完成DOP优化，共处理 {len(optimal_dops)} 个查询")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}, must be 'NN' or 'GNN'")
    
    return optimal_dops

# ==============================================================================
# 脚本入口 (不应在此文件内执行)
# ==============================================================================
# if __name__ == "__main__":
#     # 这里的逻辑会被移到 scripts/run_optimization.py
#     # 需要加载数据，初始化 ONNXManager 等
#     # df_plans = pd.read_csv(...)
#     # onnx_manager = ONNXModelManager(...)
#     # results = run_dop_optimization(df_plans, onnx_manager)
#     pass