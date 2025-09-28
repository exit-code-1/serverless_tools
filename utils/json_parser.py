import csv
import json
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Operator:
    plan_id: int
    width: int
    operator_type: str
    dop: int  # 线程块的 optimal_dop
    left_child: int  # 左子节点的 plan_id，没有则为 None
    parent_child: int

def parse_json_to_operator_map(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    operator_map = defaultdict(dict)

    with open(output_file, 'w') as f:
        f.write("query_id,plan_id,operator_type,width,dop,left_child,parent_child\n")  # CSV 头部
        for query in data["queries"]:
            query_id = int(query["query_id"])  # 转换为整数
            # 再处理 operator_map
            for thread_block in query["thread_blocks"]:
                optimal_dop = thread_block["optimal_dop"]  # 线程块的 dop
                for operator in thread_block["operators"]:
                    plan_id = operator["plan_id"]
                    operator_type = operator["operator_type"]
                    width = operator["width"]
                    left_child = operator["left_child"]
                    parent_child = operator["parent_child"]

                    operator_map[query_id][plan_id] = Operator(
                        plan_id=plan_id,
                        operator_type=operator_type,
                        width=width,
                        dop=optimal_dop,
                        left_child=left_child,
                        parent_child=parent_child
                    )
                    # 按 query_id 分组写入文件
                    f.write(f"{query_id},{plan_id},{operator_type},{width},{optimal_dop},{left_child},{parent_child}\n")

    return operator_map

def extract_dop_from_json(json_path, csv_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    result = [['query_id', 'optimal_dop']]

    for query in data.get('queries', []):
        query_id = query.get('query_id')
        max_dop = query.get('max_dop')
        result.append([query_id, max_dop])

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(result)

    print(f'Data successfully written to {csv_path}')

def generate_query_file(input_file, query_id, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        f.write("plan_id,operator_type,width,dop,left_child,parent_child\n")  # 只保留这几列
        for line in lines[1:]:  # 跳过标题行
            parts = line.strip().split(',')
            if int(parts[0]) == query_id:  # 过滤指定 query_id
                f.write(f"{parts[1]},{parts[2]},{parts[3]},{parts[4]},{parts[5]},{parts[6]}\n")
                

# # 示例用法：
# json_file = "/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/query_details.json"
# output_file = "json/operators.txt"
# operator_map = parse_json_to_operator_map(json_file, output_file)
# generate_query_file(output_file, 17
#                     , "/home/zhy/opengauss/json/query.txt")
# # extract_dop_from_json(json_file, 'no_dop/dop_result/query_dop_grid.csv')


