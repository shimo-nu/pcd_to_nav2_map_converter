"""人工点群（床・壁・天井・家具）の生成。テストとデモの両方で使う。

座標系は GLIM 地図座標系を模しており，Z 軸が鉛直上向き。
"""
from __future__ import annotations

import numpy as np


def plane_xy(x0: float, x1: float, y0: float, y1: float, z: float, step: float) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def wall_x(x: float, y0: float, y1: float, z0: float, z1: float, step: float) -> np.ndarray:
    """x = const の鉛直な壁。"""
    ys = np.arange(y0, y1, step)
    zs = np.arange(z0, z1, step)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack([np.full(gy.size, x), gy.ravel(), gz.ravel()])


def wall_y(y: float, x0: float, x1: float, z0: float, z1: float, step: float) -> np.ndarray:
    """y = const の鉛直な壁。"""
    xs = np.arange(x0, x1, step)
    zs = np.arange(z0, z1, step)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    return np.column_stack([gx.ravel(), np.full(gx.size, y), gz.ravel()])


def box(x0, x1, y0, y1, z0, z1, step) -> np.ndarray:
    """直方体の表面（家具など）。"""
    return np.vstack([
        plane_xy(x0, x1, y0, y1, z1, step),
        wall_x(x0, y0, y1, z0, z1, step), wall_x(x1, y0, y1, z0, z1, step),
        wall_y(y0, x0, x1, z0, z1, step), wall_y(y1, x0, x1, z0, z1, step),
    ])


def make_room(
    x0: float = -5.0, x1: float = 5.0, y0: float = -3.0, y1: float = 3.0,
    floor_z: float = 0.0, ceiling_z: float = 2.5, step: float = 0.02,
    with_ceiling: bool = True, with_table: bool = True, seed: int = 0, noise: float = 0.0,
) -> np.ndarray:
    """床・四方の壁・天井・テーブルを持つ部屋。原点はオフセットさせて非対称にできる。"""
    parts = [plane_xy(x0, x1, y0, y1, floor_z, step)]
    for x in (x0, x1):
        parts.append(wall_x(x, y0, y1, floor_z, ceiling_z, step))
    for y in (y0, y1):
        parts.append(wall_y(y, x0, x1, floor_z, ceiling_z, step))
    if with_ceiling:
        parts.append(plane_xy(x0, x1, y0, y1, ceiling_z, step))
    if with_table:
        # 床から 0.7 m のテーブル（天板のみ：脚は省略）。XY では床と重なる。
        parts.append(plane_xy(x0 + 2.0, x0 + 3.0, y0 + 1.0, y0 + 2.0, floor_z + 0.7, step))
    pts = np.vstack(parts)
    if noise > 0:
        pts = pts + np.random.default_rng(seed).normal(0.0, noise, pts.shape)
    return pts
