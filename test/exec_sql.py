import os
import subprocess
import shutil

def setup_environment_variable(source_dir):
    """
    自动处理所有动态环境变量，通过读取 gauss_env.sh 文件加载所有的环境变量到当前 Python 环境中。
    """
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

def main():
    # 定义参数和路径
    instance_mem_values = [16384]  # 示例 INSTANCE_MEM 值
    dop_values = [1, 2, 4, 8]  # 示例 DOP 值
    sql_dir = "/home/zhy/opengauss/tools/TPCH-og/SQL/SQL/"
    sql_files = [f"{sql_dir}/{i}.sql" for i in range(1, 23)]  # TPCH SQL 脚本路径
    source_dir = "/home/zhy/gauss_env.sh"
    gauss_dir = "/home/zhy/opengauss/GaussData"  # 替换为实际数据目录
    data_dir = "/home/zhy/opengauss/data_file"  # 替换为实际数据目录
    databases = ["tpch_1", "tpch_5", "tpch_8", "tpch_10"]

    # 使用环境配置文件加载环境变量
    setup_environment_variable(source_dir)

    # 为每个数据库设置输出目录并执行任务
    for database in databases:
        # 创建每个数据库的输出目录
        database_output_dir = os.path.join(data_dir, f"{database}_output")
        os.makedirs(database_output_dir, exist_ok=True)

        opts = '-p 5432 -d ' + database + ' -U zhy'
        for instance_mem in instance_mem_values:
            for dop in dop_values:
                # 计算参数
                shared_buffers = int((instance_mem - 1024) / 16)
                cstore_buffers = int((instance_mem - 1512) / 4)
                work_mem = int((instance_mem - 1512) / 16)

                # 使用 gs_guc 设置参数
                execute_command(f"gs_guc set -D {gauss_dir} -c \"shared_buffers={shared_buffers}MB\"")
                execute_command(f"gs_guc set -D {gauss_dir} -c \"cstore_buffers={cstore_buffers}MB\"")
                execute_command(f"gs_guc set -D {gauss_dir} -c \"work_mem={work_mem}MB\"")

                for sql_file in sql_files:
                    with open(sql_file, 'r') as f:
                        sql_content = f.read()
                    # 重启数据库
                    execute_command(f"gs_ctl restart -D {gauss_dir} -Z single_node -l {gauss_dir}/logfile")
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

        # 重置 query_id_counter 文件中的数字
        reset_query_id_counter(data_dir)
        print(f"Reset query_id_counter for database {database}")

if __name__ == "__main__":
    main()
