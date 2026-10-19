"""PCD ファイルの読み込み。

Open3D で読み込み，XYZ 座標を (N, 3) の float64 NumPy 配列として返す。
色・強度などの属性は使わない。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


def load_pcd(path: str | Path) -> np.ndarray:
    """PCD（または Open3D が読める点群形式）を読み込み XYZ 配列を返す。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"入力点群が見つかりません: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"点群を読み込めませんでした（点数 0 または形式不正）: {path}")
    log.info("PCD を読み込みました: %s (%d 点)", path, len(points))
    return points


def remove_invalid_points(points: np.ndarray) -> tuple[np.ndarray, int]:
    """NaN / Inf を含む点を削除する。(残った点, 削除点数) を返す。"""
    valid = np.isfinite(points).all(axis=1)
    n_removed = int((~valid).sum())
    if n_removed:
        points = points[valid]
    return points, n_removed
