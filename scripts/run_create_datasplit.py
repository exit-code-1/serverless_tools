# 文件路径: scripts/run_create_datasplit.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

def create_dataset_split(dataset_name='tpcds_100g_new', train_ratio=0.8):
    """
    读取指定数据集，按query_id进行80/20的随机划分，并生成新的训练集和测试集目录。
    """
    # --- 1. 项目根目录和路径设置 ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
    
    source_dir = os.path.join(DATA_DIR, dataset_name)
    train_dir = os.path.join(DATA_DIR, f"{dataset_name}_train")
    test_dir = os.path.join(DATA_DIR, f"{dataset_name}_test")
    
    source_plan_info_path = os.path.join(source_dir, "plan_info.csv")
    source_query_info_path = os.path.join(source_dir, "query_info.csv")

    print(f"--- 开始为数据集 '{dataset_name}' 创建数据划分 ---")
    print(f"源数据目录: {source_dir}")

    # --- 2. 检查源文件是否存在 ---
    if not os.path.exists(source_plan_info_path) or not os.path.exists(source_query_info_path):
        print(f"错误: 源数据文件未找到。请确保以下文件存在:")
        print(f"  - {source_plan_info_path}")
        print(f"  - {source_query_info_path}")
        sys.exit(1)
        
    # --- 3. 创建输出目录 ---
    print(f"创建输出目录:")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"  - 训练集目录: {train_dir}")
    print(f"  - 测试集目录: {test_dir}")

    # --- 4. 读取数据并获取唯一的 query_id ---
    print("正在读取数据并提取唯一的查询ID...")
    try:
        df_query_info = pd.read_csv(source_query_info_path, delimiter=';')
        # 从 query_info 获取唯一的 query_id，因为 plan_info 可能很大
        all_query_ids = df_query_info['query_id'].unique()
        print(f"成功找到 {len(all_query_ids)} 个唯一的查询ID。")
    except Exception as e:
        print(f"读取 query_info.csv 时出错: {e}")
        sys.exit(1)
        
    # --- 5. 按80/20比例随机划分 query_id ---
    print(f"正在按 {train_ratio:.0%} / {1-train_ratio:.0%} 的比例随机划分查询ID...")
    train_ids, test_ids = train_test_split(
        all_query_ids, 
        train_size=train_ratio, 
        random_state=42  # 使用固定的随机种子以保证每次划分结果一致
    )
    print(f"划分完成: {len(train_ids)} 个用于训练, {len(test_ids)} 个用于测试。")

    # 为了方便核对，保存划分的ID列表
    split_summary_path = os.path.join(DATA_DIR, f"{dataset_name}_split_summary.csv")
    pd.DataFrame({
        'query_id': list(train_ids) + list(test_ids),
        'split': ['train'] * len(train_ids) + ['test'] * len(test_ids)
    }).sort_values('query_id').to_csv(split_summary_path, index=False)
    print(f"查询ID划分详情已保存至: {split_summary_path}")

    # --- 6. 筛选并保存 query_info.csv ---
    print("正在筛选并保存 query_info.csv...")
    train_query_info_df = df_query_info[df_query_info['query_id'].isin(train_ids)]
    test_query_info_df = df_query_info[df_query_info['query_id'].isin(test_ids)]

    train_query_info_df.to_csv(os.path.join(train_dir, "query_info.csv"), sep=';', index=False)
    test_query_info_df.to_csv(os.path.join(test_dir, "query_info.csv"), sep=';', index=False)
    print("  - query_info.csv 已保存。")

    # --- 7. 筛选并保存 plan_info.csv (分块处理以应对大文件) ---
    print("正在筛选并保存 plan_info.csv (可能需要一些时间)...")
    try:
        # 定义输出路径
        train_plan_info_path = os.path.join(train_dir, "plan_info.csv")
        test_plan_info_path = os.path.join(test_dir, "plan_info.csv")

        # 初始化输出文件并写入表头
        header_written_train = False
        header_written_test = False

        # 使用 chunksize 迭代读取大文件
        chunk_iterator = pd.read_csv(source_plan_info_path, delimiter=';', chunksize=100000)
        
        for i, chunk in enumerate(chunk_iterator):
            print(f"  处理 plan_info 块 {i+1}...")
            # 筛选训练数据
            train_chunk = chunk[chunk['query_id'].isin(train_ids)]
            if not train_chunk.empty:
                train_chunk.to_csv(
                    train_plan_info_path, 
                    mode='a', 
                    sep=';', 
                    index=False, 
                    header=not header_written_train
                )
                header_written_train = True

            # 筛选测试数据
            test_chunk = chunk[chunk['query_id'].isin(test_ids)]
            if not test_chunk.empty:
                test_chunk.to_csv(
                    test_plan_info_path, 
                    mode='a', 
                    sep=';', 
                    index=False, 
                    header=not header_written_test
                )
                header_written_test = True
        
        print("  - plan_info.csv 已保存。")
        
    except Exception as e:
        print(f"处理 plan_info.csv 时出错: {e}")
        sys.exit(1)
        
    print(f"\n--- 数据集划分成功！---")

# --- 脚本入口 ---
if __name__ == "__main__":
    # 在这里指定要划分的数据集名称
    target_dataset = 'tpcds_100g_new'
    create_dataset_split(dataset_name=target_dataset, train_ratio=0.8)