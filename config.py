"""config.yaml の読み込みと既定値の補完。

設定値はすべて config.yaml で管理する。ここにある DEFAULT_CONFIG は
「キーが省略されたときの既定値」を一箇所にまとめたもので，config.yaml と
同じ構造を持つ。値の意味・単位の説明は config.yaml のコメントを参照。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "input": {"pcd_path": "./data/map.pcd"},
    "output": {
        "directory": "./output",
        "map_name": "map",
        "save_preview": True,
        "save_log": True,
    },
    "preprocessing": {
        "remove_invalid_points": True,
        "height_crop": {"enabled": False, "min_z": -2.0, "max_z": 5.0},
        "voxel_downsample": {"enabled": True, "voxel_size": 0.02},
        "statistical_outlier": {"enabled": False, "nb_neighbors": 20, "std_ratio": 2.0},
    },
    "map": {
        "resolution": 0.05,
        "padding": 1.0,
        "bounds": {
            "enabled": False,
            "min_x": -20.0,
            "max_x": 20.0,
            "min_y": -20.0,
            "max_y": 20.0,
        },
        "origin_yaw": 0.0,
        "vertical_axis_check": {
            "enabled": True,
            "max_tilt_deg": 5.0,
            "band_height": 0.30,
            "sample_points": 50000,
        },
    },
    "height_filter": {
        "mode": "relative_to_floor",
        "floor": {
            "method": "fixed",
            "fixed_z": 0.0,
            "percentile": 2.0,
            "ransac_distance_threshold": 0.05,
            "ransac_iterations": 1000,
            "ransac_max_points": 200000,
            "ransac_z_range": {"enabled": False, "min_z": -1.0, "max_z": 0.5},
        },
        "obstacle": {"min_height": 0.10, "max_height": 1.80, "min_points_per_cell": 2},
    },
    "free_space": {
        "method": "floor_based",
        "floor_based": {
            "floor_height_tolerance": 0.08,
            "min_floor_points_per_cell": 2,
            "min_floor_coverage_warn": 0.05,
        },
    },
    "postprocessing": {
        "remove_small_obstacles": {"enabled": False, "min_size_cells": 3, "connectivity": 8},
        "fill_small_holes": {"enabled": False, "max_size_cells": 3},
        "fill_small_unknown_in_free": {"enabled": False, "max_size_cells": 3},
        "erosion": {"enabled": False, "iterations": 1},
        "dilation": {"enabled": False, "iterations": 1},
    },
    "nav2": {"mode": "trinary", "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196},
    "visualization": {"enabled": True, "show_grid": False, "show_axes": True, "dpi": 150},
}


def deep_merge(base: dict, override: dict) -> dict:
    """base に override を再帰的に上書きした新しい dict を返す。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML を読み込み，既定値で欠損キーを補完して返す。"""
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError(f"config のトップレベルはマッピングである必要があります: {path}")
    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """値域・選択肢の簡易検証。問題があれば ValueError。"""
    if cfg["map"]["resolution"] <= 0:
        raise ValueError("map.resolution は正の値である必要があります")
    if cfg["map"]["padding"] < 0:
        raise ValueError("map.padding は 0 以上である必要があります")
    b = cfg["map"]["bounds"]
    if b["enabled"] and (b["min_x"] >= b["max_x"] or b["min_y"] >= b["max_y"]):
        raise ValueError("map.bounds は min < max である必要があります")
    if cfg["height_filter"]["mode"] not in ("absolute", "relative_to_floor"):
        raise ValueError("height_filter.mode は 'absolute' か 'relative_to_floor'")
    if cfg["height_filter"]["floor"]["method"] not in ("fixed", "percentile", "ransac"):
        raise ValueError("height_filter.floor.method は 'fixed' / 'percentile' / 'ransac'")
    obs = cfg["height_filter"]["obstacle"]
    if obs["min_height"] >= obs["max_height"]:
        raise ValueError("height_filter.obstacle は min_height < max_height")
    if obs["min_points_per_cell"] < 1:
        raise ValueError("height_filter.obstacle.min_points_per_cell は 1 以上")
    if cfg["free_space"]["method"] not in ("floor_based", "unknown_default", "bounded_free"):
        raise ValueError("free_space.method は 'floor_based' / 'unknown_default' / 'bounded_free'")
    if cfg["postprocessing"]["remove_small_obstacles"]["connectivity"] not in (4, 8):
        raise ValueError("postprocessing.remove_small_obstacles.connectivity は 4 か 8")
    if cfg["nav2"]["mode"] != "trinary":
        raise ValueError("nav2.mode は現状 'trinary' のみ対応（PGM 画素値の設計が trinary 前提）")
