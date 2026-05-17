# -*- coding: utf-8 -*-
"""Centralized backend settings loaded from config.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml


_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_BACKEND_DIR, "config.yaml")


class Settings:
    """Typed wrapper around the YAML configuration."""

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw

    @property
    def predictor_root(self) -> str:
        return self._raw["predictor_root"]

    @property
    def query_txt_path(self) -> str:
        return self._raw["query_txt_path"]

    @property
    def gauss_env_sh(self) -> str:
        return self._raw["gauss_env_sh"]

    @property
    def gauss_data_dir(self) -> str:
        return self._raw["gauss_data_dir"]

    @property
    def data_file_dir(self) -> str:
        return self._raw["data_file_dir"]

    @property
    def dataset_data_root(self) -> str:
        return str(self._raw.get("dataset_data_root", ""))

    @property
    def dataset_data_dirs(self) -> Dict[str, str]:
        return dict(self._raw.get("dataset_data_dirs", {}))

    @property
    def databases(self) -> Dict[str, str]:
        return dict(self._raw.get("databases", {}))

    @property
    def sql_dirs(self) -> Dict[str, str]:
        return dict(self._raw.get("sql_dirs", {}))

    @property
    def gsql_port(self) -> int:
        return int(self._raw.get("gsql_port", 4321))

    @property
    def gsql_user(self) -> str:
        return str(self._raw.get("gsql_user", "zhy"))

    @property
    def gsql_extra_opts(self) -> str:
        return str(self._raw.get("gsql_extra_opts", "") or "")

    @property
    def optimize_defaults(self) -> Dict[str, Any]:
        return dict(self._raw.get("optimize_defaults", {}))

    @property
    def runs_dir(self) -> str:
        return self._raw["runs_dir"]

    @property
    def history_db_path(self) -> str:
        return self._raw["history_db_path"]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_path = os.environ.get("PREDICTOR_UI_CONFIG", DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Backend config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    return Settings(raw)
