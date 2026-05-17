# Serverless Predictor 可视化控制台

UI 模块由两部分组成：

- `ui/backend`：FastAPI 服务，复用 `scripts/optimize.py` 等现有模块，封装数据集浏览、资源预测、openGauss 执行调度、结果回读、历史记录等接口。
- `ui/frontend`：Vite + React + TypeScript 前端，提供「选择查询 → 资源预测 → 触发执行 → 查看结果 → 指标面板 → 历史记录」六个步骤的可视化界面。

## 1. 后端

### 1.1 依赖

```bash
cd ui/backend
pip install -r requirements.txt
```

后端复用工程根目录下的 `scripts/optimize.py`、`utils/*`、`config/main_config.py` 等模块，因此请在已经能跑 `python scripts/main.py` 的同一 Python 环境中安装上述依赖（本机为 conda env `zhy_env`）。

### 1.2 配置

编辑 `ui/backend/config.yaml`：

| key                  | 作用                                                    |
| -------------------- | ------------------------------------------------------- |
| `predictor_root`     | 工程根，影响 `sys.path` 与缓存目录                     |
| `query_txt_path`     | 写出的 `query.txt` 位置（openGauss 端读取此文件）        |
| `gauss_env_sh`       | bash 环境变量脚本（提供 `GAUSSHOME`/`PATH` 等）         |
| `gauss_data_dir`     | `gs_ctl -D` 使用的数据目录                              |
| `data_file_dir`      | openGauss 执行后写入 `plan_info.csv`/`query_info.csv`   |
| `databases.*`        | 不同 dataset 对应的逻辑库名                              |
| `sql_dirs.*`         | 各 dataset 的 `<query_id>.sql` 所在目录                  |
| `gsql_port` / `gsql_user` / `gsql_extra_opts` | gsql 连接参数                          |
| `optimize_defaults`  | `run_pipeline_optimization` 默认参数                    |
| `runs_dir`           | 单次预测产物（payload JSON）的缓存目录                  |
| `history_db_path`    | 历史记录 SQLite 文件路径                                |

> 也可以通过环境变量 `PREDICTOR_UI_CONFIG=/abs/path/to/your.yaml` 覆盖配置文件位置。

### 1.3 启动

最简单的方式：

```bash
cd <repo>
PYTHON=/home/zhy/miniconda3/envs/zhy_env/bin/python ui/run_backend.sh
# 等价于：
# python -m uvicorn ui.backend.main:app --host 0.0.0.0 --port 8000
```

启动脚本默认监听 `0.0.0.0:8000`，可用 `HOST` / `PORT` / `LOG_LEVEL` / `PYTHON` 环境变量覆盖。

启动后：

- `GET /api/health` 健康检查
- `GET /docs` Swagger 文档
- `GET /api/datasets`、`/api/datasets/{ds}/queries`、`/api/datasets/{ds}/queries/{qid}`
- `POST /api/optimize`、`GET /api/optimize/{task_id}`、`GET /api/optimize/{task_id}/result`
- `POST /api/execute`、`GET /api/execute/{task_id}`、`GET /api/execute/{task_id}/logs`
- `GET /api/results/{ds}/{qid}` 读取 openGauss 的实际指标
- `GET/POST/DELETE /api/history`
- 任意任务实时日志流：`GET /api/tasks/{task_id}/stream` (SSE)

如果 `ui/frontend/dist` 已经构建过，FastAPI 会把它作为静态站点挂在 `/`。

## 2. 前端

### 2.1 在你的本地机器上运行

前端代码使用 Vite + React + TypeScript，需要 Node.js 18+。**可以在任意机器上跑，不必和后端在同一台机器。**

```bash
cd ui/frontend
npm install
# 后端不在本机时，把代理指向后端实际地址：
VITE_BACKEND_URL=http://<vm-ip>:8000 npm run dev
```

开发服务器默认监听 `http://localhost:5173`，并把 `/api/*` 代理到 `VITE_BACKEND_URL`。

### 2.2 生产构建

```bash
npm run build
```

构建产物在 `ui/frontend/dist`。如果你把 `dist/` 同步回服务器的 `ui/frontend/dist`，FastAPI 启动时会自动把它挂在 `/`，单端口即可对外。

## 3. 端到端流程

1. **选择查询**：从 `data_kunpeng/<dataset>/<test_dir>/query_info.csv` 中挑选 `query_id`，前端展示 Plan Tree 和 SQL 文本。
2. **资源预测**：UI 调用 `POST /api/optimize`，后台运行 `run_pipeline_optimization`（或 `baseline` / `query_level` / `ppm`），完成后从 `output/<dataset>/optimization_results/pipeline_optimization.json` 中按 `query_id` 抽取 thread blocks，缓存到 `ui/runs/<task_id>/`。
3. **执行查询**：UI 调用 `POST /api/execute`，后台先 `generate_query_file()` 写出 `query.txt`，再依次执行 `gs_ctl restart` 和 `gsql -p ... -d <db> -c "SET query_dop=N; <sql>"`。stdout/stderr 通过 SSE 实时推到前端。
4. **执行结果**：UI 调用 `GET /api/results/{dataset}/{qid}`，后台从 `data_file_dir/{plan_info,query_info}.csv` 拉取对应 query 的最新行，渲染 KPI / Plan Tree / 算子表格。
5. **指标面板**：在第 4 步基础上做汇总——KPI 卡片、DOP 分布、Top 耗时算子等。
6. **历史记录**：可以把当前预测 + 实际结果一键保存到 SQLite，并在历史页面勾选多条进行横向对比。

## 4. 已有的安全栏

- `gs_ctl restart` 会重启整库，前端按钮要求二次勾选「我已确认可以执行」。
- 同一时刻最多只允许一个执行任务通过 `gauss_runner` 内部的全局锁串行化。
- 预测任务同样持有全局锁，避免多次并发污染 `output/<dataset>/optimization_results/`。
