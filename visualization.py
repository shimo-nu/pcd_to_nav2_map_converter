"""生成した 2D 地図のプレビュー画像（世界座標軸付き）。"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ディスプレイ不要
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from occupancy_grid import GridSpec, grid_to_pgm  # noqa: E402

log = logging.getLogger(__name__)


def save_preview(path: str | Path, grid: np.ndarray, spec: GridSpec, vis_cfg: dict, stats: dict, title: str = "") -> Path:
    """グリッドを世界座標 [m] の軸で描画して PNG に保存する。

    PGM と同じ画素値（Free=254 白，Occupied=0 黒，Unknown=205 灰）で描く。
    extent を origin から計算するため，軸の目盛りが GLIM 座標系の値になる。
    """
    pgm = grid_to_pgm(grid)  # row 0 = 北
    extent = (spec.origin_x, spec.max_x, spec.origin_y, spec.max_y)

    w_m, h_m = spec.max_x - spec.origin_x, spec.max_y - spec.origin_y
    base = 8.0
    figsize = (base, base * h_m / w_m) if w_m >= h_m else (base * w_m / h_m, base)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(pgm, cmap="gray", vmin=0, vmax=255, extent=extent, origin="upper", interpolation="nearest")

    # map.yaml の origin（左下セルの左下隅）
    ax.plot(spec.origin_x, spec.origin_y, marker="+", color="red", markersize=12, mew=2)
    ax.annotate(f"origin ({spec.origin_x:.2f}, {spec.origin_y:.2f})", (spec.origin_x, spec.origin_y),
                color="red", fontsize=8, xytext=(4, 4), textcoords="offset points")
    # 世界座標の原点 (0,0) が地図内なら印をつける
    if spec.origin_x <= 0 <= spec.max_x and spec.origin_y <= 0 <= spec.max_y:
        ax.plot(0, 0, marker="o", color="tab:blue", markersize=6)
        ax.annotate("(0, 0)", (0, 0), color="tab:blue", fontsize=8, xytext=(4, 4), textcoords="offset points")

    if vis_cfg.get("show_grid", False):
        for x in np.arange(np.ceil(spec.origin_x), spec.max_x, 1.0):
            ax.axvline(x, color="tab:orange", lw=0.3, alpha=0.6)
        for y in np.arange(np.ceil(spec.origin_y), spec.max_y, 1.0):
            ax.axhline(y, color="tab:orange", lw=0.3, alpha=0.6)

    if vis_cfg.get("show_axes", True):
        ax.set_xlabel("x [m] (GLIM map frame)")
        ax.set_ylabel("y [m] (GLIM map frame)")
    else:
        ax.set_axis_off()

    sub = (f"{spec.width}x{spec.height} cells @ {spec.resolution} m | "
           f"free {stats['free_ratio']*100:.1f}%  occ {stats['occupied_ratio']*100:.1f}%  "
           f"unknown {stats['unknown_ratio']*100:.1f}%")
    ax.set_title((title + "\n" if title else "") + sub, fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=int(vis_cfg.get("dpi", 150)))
    plt.close(fig)
    return path
