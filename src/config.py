"""配置加载：单一入口，任何模块都不许自己去读 yaml。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"


def _load(name: str) -> Dict[str, Any]:
    with open(CONF_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def config() -> Dict[str, Any]:
    return _load("config.yaml")


@lru_cache(maxsize=None)
def label_def() -> Dict[str, Any]:
    return _load("label_def.yaml")


@lru_cache(maxsize=None)
def price_decks() -> Dict[str, Any]:
    return _load("price_deck.yaml")["decks"]


@lru_cache(maxsize=None)
def units() -> Dict[str, Any]:
    """SEC 单元层级（仅合成适配器读取；业务代码读 sec_unit / unit_well 表）。"""
    return _load("units.yaml")


@lru_cache(maxsize=None)
def indicators() -> Dict[str, Any]:
    return _load("indicators.yaml")


def label_def_version() -> str:
    return str(label_def()["version"])


def path(key: str) -> Path:
    """conf.paths 里的相对路径统一解析为仓库内绝对路径。"""
    p = ROOT / config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    url = config()["db"]["url"]
    if not url.startswith("sqlite:///"):
        raise ValueError(
            f"当前 db.url={url!r}。本仓库内置 sqlite 实现；"
            "接 PostgreSQL 请在 src/db.py 里换成对应驱动，schema.sql 无需改动。"
        )
    rel = url[len("sqlite:///"):]
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
