# coding:utf-8 
import os
import subprocess
import shutil
import csv

import pandas as pd

def setup_environment_variable(source_dir):
    """自动处理所有动态环境变量，通过读取 gauss_env.sh 文件加载所有的环境变量到当前 Python 环境中。"""
    with open(source_dir, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('export'):
                    # 解析 export 语句
                    key_value = line.replace('export', '', 1).strip()
                    if '=' in key_value:
                        key, value = key_value.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 剥离注释
                        if '#' in value:
                            value = value.split('#', 1)[0].strip()
                        # 解析依赖于其他环境变量的值
                        value = os.path.expandvars(value)
                        # 设置到环境变量中
                        os.environ[key] = value

def execute_command(cmd, output_file='output.log', error_file='error.log'):
    """执行命令，并将输出追加到文件"""
    # 确保文件存在，如果不存在则创建
    if not os.path.exists(output_file):
        with open(output_file, 'w') as out_file:
            out_file.write("")  # 创建空文件
    if not os.path.exists(error_file):
        with open(error_file, 'w') as err_file:
            err_file.write("")  # 创建空文件

    # 以追加模式打开文件
    with open(output_file, 'a') as out_file, open(error_file, 'a') as err_file:
        result = subprocess.run(cmd, shell=True, env=os.environ, stdout=out_file, stderr=err_file)
    return result.returncode  # 返回命令的返回码（0 表示成功）


def update_query_id_counter(data_dir, query_id):
    """更新 query_id_counter 文件中的数字"""
    query_id_file = os.path.join(data_dir, 'query_id_counter')
    if os.path.exists(query_id_file):
        with open(query_id_file, 'w') as f:
            f.write(str(query_id))  # 更新为当前的 query_id
    else:
        # 如果文件不存在，创建并写入当前的 query_id
        with open(query_id_file, 'w') as f:
            f.write(str(query_id))

def reset_query_id_counter(database_dir):
    """重置 query_id_counter 文件中的数字"""
    query_id_file = os.path.join(database_dir, 'query_id_counter')
    if os.path.exists(query_id_file):
        with open(query_id_file, 'w') as f:
            f.write('1')  # 重置为 1

def clear_data_file(data_dir):
    """清空 data_file 目录中的 query_info.csv 和 plan_info.csv 文件"""
    query_info_path = os.path.join(data_dir, 'query_info.csv')
    plan_info_path = os.path.join(data_dir, 'plan_info.csv')

    if os.path.exists(query_info_path):
        with open(query_info_path, 'w') as f:
            f.truncate(0)  # 清空文件
    if os.path.exists(plan_info_path):
        with open(plan_info_path, 'w') as f:
            f.truncate(0)  # 清空文件

def get_last_processed_query(data_dir):
    """读取 query_info.csv 文件并返回上一个处理的 query_id 和 dop"""
    query_info_path = os.path.join(data_dir, 'query_info.csv')
    last_query_id = 0
    last_dop = 0

    if os.path.exists(query_info_path) and os.path.getsize(query_info_path) > 0:
        # 使用 pandas 读取 CSV 文件，指定分隔符为分号，并跳过第一行列名
        df = pd.read_csv(query_info_path, delimiter=';', header=0)

        if not df.empty:
            # 获取最后一行的 query_id 和 dop
            last_query_id = int(df.iloc[-1, 0])  # 假设 query_id 在第一列
            last_dop = int(df.iloc[-1, 1])  # 假设 dop 在第二列

    return last_query_id, last_dop

def generate_query_file(input_file, query_id, output_file):
    # 读取 CSV 文件为 DataFrame
    df = pd.read_csv(input_file)

    # 过滤指定的 query_id（假设第一列是 query_id）
    df_filtered = df[df.iloc[:, 0] == query_id]

    # 选择需要保留的列（注意列名必须和 CSV 文件一致）
    columns_to_keep = ["plan_id", "operator_type", "width", "dop", "left_child", "parent_child"]
    df_filtered = df_filtered[columns_to_keep]

    # 写入输出文件
    df_filtered.to_csv(output_file, index=False)
                
def main():
    # 定义参数和路径
    total_querys = 22
    instance_mem_values = [150]  # 示例 INSTANCE_MEM 值
    dop_values = [64]  # 示例 DOP 值
    # sql_dir = "/mnt/zhy/tpcds-kit-master/tpcds_query_100"
    sql_dir = "/mnt/nvme2/zhy/AP_DATA/TPCH/TPCH-og/SQL/SQL"
    source_dir = "/home/zhy/gauss_env.sh"
    gauss_dir = "/mnt/nvme2/zhy/gaussdata"  # 替换为实际数据目录
    data_dir = "/mnt/nvme2/zhy/data_file"  # 替换为实际数据目录
    databases = ["tpch_100g"]
    skip_queries = {2, 9}
    # total_querys = 200
    # instance_mem_values = [500]  # 示例 INSTANCE_MEM 值
    # dop_values = [8, 16, 32, 64, 96]  # 示例 DOP 值
    # sql_dir = "/mnt/nvme0/zhy/zhy/tpcds-kit-master/tpcds_query_200"
    # source_dir = "/mnt/nvme0/zhy/zhy/gauss_env.sh"
    # gauss_dir = "/mnt/nvme0/zhy/gaussdata"  # 替换为实际数据目录
    # data_dir = "/mnt/nvme0/zhy/data_file"  # 替换为实际数据目录
    # databases = ["tpcds_100g"]
    # skip_queries = {1,2,3,5,9,14,16,17,21,23,24,30,32,33,37,40,42,54,55,56,57,59,60,63,64,76,77,78,80,81,92,95}
    # skip_queries = {1,2,5,9,14,16,17,21,23,24,30,32,33,40,47,42,52,54,56,57,59,60,63,64,77,80,81,92,95}
    # 使用环境配置文件加载环境变量
    setup_environment_variable(source_dir)

    # 获取上次处理的 query_id 和 dop
    last_query_id, last_dop = get_last_processed_query(data_dir)

    # 为每个数据库设置输出目录并执行任务
    for database in databases:
        # 创建每个数据库的输出目录
        database_output_dir = os.path.join(data_dir, f"{database}_MCI")
        os.makedirs(database_output_dir, exist_ok=True)

        opts = '-p 4321 -d ' + database + ' -U zhy'
        for instance_mem in instance_mem_values:
            for dop in dop_values:
                # 重置 query_id_counter 文件中的数字
                reset_query_id_counter(data_dir)
                print(f"Reset query_id_counter for database {database}")
                # 如果已经处理过当前的 dop 和 query_id，跳过
                if last_dop > dop:
                    continue
                if  last_dop == dop and last_query_id > 0:
                    start_query_id = last_query_id + 1
                    update_query_id_counter(data_dir, start_query_id)
                else:
                    start_query_id = 1

                for query_id in range(start_query_id, total_querys + 1):
                    if query_id in skip_queries:
                        update_query_id_counter(data_dir, query_id + 1)
                        continue
                    sql_file = f"{sql_dir}/{query_id}.sql"
                    if not os.path.exists(sql_file):
                        continue  # 如果 SQL 文件不存在则跳过
                    # 计算参数
                    shared_buffers = int((instance_mem - 1024) / 16)
                    cstore_buffers = int((instance_mem - 1512) / 6)
                    work_mem = 51200

                    # 使用 gs_guc 设置参数
                    # execute_command(f"gs_guc set -D {gauss_dir} -c \"shared_buffers={shared_buffers}MB\"")
                    # execute_command(f"gs_guc set -D {gauss_dir} -c \"cstore_buffers={cstore_buffers}MB\"")
                    # execute_command(f"gs_guc set -D {gauss_dir} -c \"work_mem={work_mem}MB\"")
                    generate_query_file("/mnt/nvme2/zhy/json/operators_optimized.csv", query_id, "/mnt/nvme2/zhy/json/query.txt")
                    # 重启数据库
                    execute_command(f"gs_ctl restart -D {gauss_dir} -Z single_node -l {gauss_dir}/logfile")
                    sql_content = open(sql_file, 'r').read()
                    sql_content = sql_content.replace('"', '\\"')
                    gsql_commands = f"""
                    SET query_dop = {dop};
                    {sql_content};
                    """
                    print(f"Running SQL from {sql_file} with INSTANCE_MEM={instance_mem}, DOP={dop}, Shared Buffers={shared_buffers}, CStore Buffers={cstore_buffers}, Work Mem={work_mem}")
                    execute_command(f"gsql {opts} -q -o tmp_result -c \"{gsql_commands}\"")

        # 将 query_info.csv 和 plan_info.csv 复制到对应数据库的输出目录
        query_info_file = os.path.join(data_dir, 'query_info.csv')
        plan_info_file = os.path.join(data_dir, 'plan_info.csv')
        if os.path.exists(query_info_file):
            shutil.copy(query_info_file, os.path.join(database_output_dir, 'query_info.csv'))
        if os.path.exists(plan_info_file):
            shutil.copy(plan_info_file, os.path.join(database_output_dir, 'plan_info.csv'))

        # 清空 data_file 目录中的 query_info.csv 和 plan_info.csv 文件
        clear_data_file(data_dir)

if __name__ == "__main__":
    main()



