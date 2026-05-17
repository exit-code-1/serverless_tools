# -*- coding: utf-8 -*-
"""Reproduces the user's batch execution helpers: env loading, gs_ctl restart,
gsql query execution. All steps stream their stdout/stderr through the task
manager so subscribers see live logs."""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from ..settings import Settings, get_settings
from .task_manager import TaskManager


_EXECUTE_LOCK = threading.Lock()


def load_env_file(path: str, env: Dict[str, str]) -> Dict[str, str]:
    """Source the bash-style export file into a dict. Mirrors the user's
    `setup_environment_variable` helper."""
    if not path or not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export"):
                stripped = stripped[len("export"):].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if "#" in value and not (value.startswith('"') or value.startswith("'")):
                value = value.split("#", 1)[0].strip()
            value = value.strip('"').strip("'")
            value = os.path.expandvars(_expand_with_env(value, env))
            env[key] = value
    return env


def _expand_with_env(text: str, env: Dict[str, str]) -> str:
    """Allow $VAR substitutions using a custom env mapping before falling back
    to the process environment."""
    if "$" not in text:
        return text
    result: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "$":
            result.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(text):
            result.append("$")
            break
        brace = text[i] == "{"
        if brace:
            i += 1
            end = text.find("}", i)
            if end == -1:
                result.append("$")
                break
            name = text[i:end]
            i = end + 1
        else:
            end = i
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            name = text[i:end]
            i = end
        if not name:
            result.append("$")
            continue
        value = env.get(name, os.environ.get(name, ""))
        result.append(value)
    return "".join(result)


def run_execution(
    *,
    task_manager: TaskManager,
    task_id: str,
    dataset: str,
    query_id: int,
    optimization_csv_path: str,
    base_dop: int,
    restart_gauss: bool = True,
) -> Dict[str, Any]:
    """Synchronous worker for executing a SQL query against openGauss."""
    settings = get_settings()
    task_manager.mark_started(task_id)

    if dataset not in settings.databases:
        raise KeyError(f"No database mapping for dataset: {dataset}")
    database = settings.databases[dataset]
    sql_dir = settings.sql_dirs.get(dataset)
    if not sql_dir:
        raise KeyError(f"No sql_dir configured for dataset: {dataset}")
    sql_path = os.path.join(sql_dir, f"{query_id}.sql")
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    from .query_writer import generate_query_file

    task_manager.append_log(task_id, f"[execute] writing query.txt for dataset={dataset} query_id={query_id}")
    generate_query_file(optimization_csv_path, query_id, settings.query_txt_path)
    task_manager.append_log(task_id, f"[execute] query.txt written to {settings.query_txt_path}")

    env = dict(os.environ)
    env = load_env_file(settings.gauss_env_sh, env)

    with _EXECUTE_LOCK:
        _truncate_data_files(settings, task_manager, task_id)
        _reset_query_id_counter(settings, task_manager, task_id)

        if restart_gauss:
            log_file = os.path.join(settings.gauss_data_dir, "logfile")
            restart_cmd = [
                "gs_ctl", "restart",
                "-D", settings.gauss_data_dir,
                "-Z", "single_node",
                "-l", log_file,
            ]
            task_manager.append_log(task_id, f"[execute] $ {_format_cmd(restart_cmd)}")
            rc = _stream_command(task_manager, task_id, restart_cmd, env=env)
            if rc != 0:
                raise RuntimeError(f"gs_ctl restart exited with code {rc}")
            _wait_for_gsql_ready(settings, database, env, task_manager, task_id)

        with open(sql_path, "r", encoding="utf-8") as fp:
            sql_content = fp.read().strip()
        if not sql_content.endswith(";"):
            sql_content += ";"
        gsql_payload = f"SET query_dop = {int(base_dop)};\n{sql_content}\n"

        gsql_cmd = [
            "gsql",
            "-p", str(settings.gsql_port),
            "-d", database,
            "-U", settings.gsql_user,
            "-q",
            "-o", "/tmp/gsql_ui_result.txt",
            "-c", gsql_payload,
        ]
        extra = settings.gsql_extra_opts
        if extra:
            gsql_cmd[1:1] = shlex.split(extra)
        task_manager.append_log(
            task_id,
            f"[execute] running query {query_id} against {database} (dop={base_dop})",
        )
        task_manager.append_log(task_id, f"[execute] SQL prelude: SET query_dop = {int(base_dop)};")
        task_manager.append_log(task_id, f"[execute] $ gsql -p {settings.gsql_port} -d {database} -U {settings.gsql_user} -c <SQL>")
        start_ts = time.time()
        rc = _stream_command(task_manager, task_id, gsql_cmd, env=env)
        elapsed = time.time() - start_ts
        task_manager.append_log(task_id, f"[execute] gsql exit={rc} elapsed={elapsed:.2f}s")
        if rc != 0:
            raise RuntimeError(f"gsql exited with code {rc}")

    plan_info_path = os.path.join(settings.data_file_dir, "plan_info.csv")
    query_info_path = os.path.join(settings.data_file_dir, "query_info.csv")
    return {
        "plan_info_path": plan_info_path,
        "query_info_path": query_info_path,
        "query_txt_path": settings.query_txt_path,
        "database": database,
        "sql_path": sql_path,
    }


def _wait_for_gsql_ready(
    settings: Settings,
    database: str,
    env: Dict[str, str],
    task_manager: TaskManager,
    task_id: str,
    *,
    timeout_seconds: int = 30,
) -> None:
    """Poll gsql after restart because gs_ctl can return before the port accepts connections."""
    deadline = time.time() + timeout_seconds
    cmd = [
        "gsql",
        "-p", str(settings.gsql_port),
        "-d", database,
        "-U", settings.gsql_user,
        "-q",
        "-c", "SELECT 1;",
    ]
    extra = settings.gsql_extra_opts
    if extra:
        cmd[1:1] = shlex.split(extra)

    task_manager.append_log(task_id, f"[execute] waiting for gsql on port {settings.gsql_port}")
    last_rc: Optional[int] = None
    last_output = ""
    while time.time() < deadline:
        last_rc, last_output = _run_command_probe(cmd, env=env, timeout_seconds=3)
        if last_rc == 0:
            task_manager.append_log(task_id, "[execute] gsql is ready")
            return
        if last_output:
            task_manager.append_log(task_id, f"[execute] gsql probe failed: {last_output}")
        time.sleep(1)
    raise RuntimeError(
        f"gsql is not ready after {timeout_seconds}s "
        f"(last exit code {last_rc}, last output: {last_output})"
    )


def _run_command_probe(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout_seconds: int = 3,
) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env or os.environ,
            cwd=cwd,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return 124, f"probe timed out after {timeout_seconds}s"
    output = (proc.stdout or "").strip().replace("\n", " ")
    return proc.returncode, output[-500:]


def _stream_command(
    task_manager: TaskManager,
    task_id: str,
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> int:
    """Run command, push stdout/stderr line-by-line to the task log, return exit code."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env or os.environ,
        cwd=cwd,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            task_manager.append_log(task_id, line.rstrip("\n"))
    finally:
        proc.wait()
    return proc.returncode


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _truncate_data_files(settings: Settings, task_manager: TaskManager, task_id: str) -> None:
    for name in ("plan_info.csv", "query_info.csv"):
        target = os.path.join(settings.data_file_dir, name)
        if os.path.exists(target):
            try:
                with open(target, "w", encoding="utf-8") as fp:
                    fp.truncate(0)
                task_manager.append_log(task_id, f"[execute] truncated {target}")
            except OSError as exc:
                task_manager.append_log(task_id, f"[execute][warn] failed to truncate {target}: {exc}")


def _reset_query_id_counter(settings: Settings, task_manager: TaskManager, task_id: str) -> None:
    counter_path = os.path.join(settings.data_file_dir, "query_id_counter")
    try:
        os.makedirs(settings.data_file_dir, exist_ok=True)
        with open(counter_path, "w", encoding="utf-8") as fp:
            fp.write("1")
        task_manager.append_log(task_id, f"[execute] reset query_id_counter at {counter_path}")
    except OSError as exc:
        task_manager.append_log(task_id, f"[execute][warn] failed to reset counter: {exc}")
