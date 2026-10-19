#!/usr/bin/env python3
"""GLIM の 3D PCD 地図 → Nav2 用 2D OccupancyGrid (map.pgm / map.yaml) 変換ツール。

使い方:
    python main.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from config import load_config
from free_space import FreeSpaceContext, create_estimator
from map_exporter import build_map_yaml, check_pgm_values_decodable, save_map_yaml, save_pgm
from occupancy_grid import (
    GridSpec,
    build_obstacle_mask,
    compose_grid,
    grid_statistics,
    grid_to_pgm,
    postprocess_grid,
)
from pcd_loader import load_pcd, remove_invalid_points
from pointcloud_processor import (
    check_vertical_axis,
    crop_xy,
    crop_z,
    estimate_floor,
    statistical_outlier_removal,
    voxel_downsample,
)

log = logging.getLogger("pcd_to_nav2_map")


@dataclass
class ConversionResult:
    grid: np.ndarray
    spec: GridSpec
    stats: dict
    n_input_points: int
    n_processed_points: int
    pgm_path: Path | None = None
    yaml_path: Path | None = None
    preview_path: Path | None = None
    log_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def setup_logging(log_path: Path | None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_path is not None:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(fh)


def run_conversion(cfg: dict, base_dir: Path | None = None, write_outputs: bool = True) -> ConversionResult:
    """設定 dict に従って変換を実行する。base_dir は相対パスの基準（config.yaml の場所）。"""
    base_dir = Path(base_dir or ".")
    t0 = time.perf_counter()
    warnings: list[str] = []

    out_dir = (base_dir / cfg["output"]["directory"]).resolve()
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "conversion.log" if (write_outputs and cfg["output"]["save_log"]) else None
    setup_logging(log_path)
    log.info("使用する設定:\n%s", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    # ---- 1. 読み込みと前処理 -------------------------------------------------
    pcd_path = base_dir / cfg["input"]["pcd_path"]
    points = load_pcd(pcd_path)
    n_input = len(points)

    pre = cfg["preprocessing"]
    if pre["remove_invalid_points"]:
        points, n_removed = remove_invalid_points(points)
        if n_removed:
            log.info("無効点 (NaN/Inf) を %d 点削除", n_removed)
    if pre["height_crop"]["enabled"]:
        points = crop_z(points, pre["height_crop"]["min_z"], pre["height_crop"]["max_z"])
        log.info("絶対 Z [%.2f, %.2f] で切り出し → %d 点", pre["height_crop"]["min_z"], pre["height_crop"]["max_z"], len(points))
    mcfg = cfg["map"]
    if mcfg["bounds"]["enabled"]:
        b = mcfg["bounds"]
        points = crop_xy(points, b["min_x"], b["max_x"], b["min_y"], b["max_y"])
        log.info("地図範囲 X[%.2f, %.2f) Y[%.2f, %.2f) で切り出し → %d 点", b["min_x"], b["max_x"], b["min_y"], b["max_y"], len(points))
    if pre["voxel_downsample"]["enabled"]:
        points = voxel_downsample(points, pre["voxel_downsample"]["voxel_size"])
        log.info("Voxel Downsampling (%.3f m) → %d 点", pre["voxel_downsample"]["voxel_size"], len(points))
    if pre["statistical_outlier"]["enabled"]:
        so = pre["statistical_outlier"]
        points = statistical_outlier_removal(points, so["nb_neighbors"], so["std_ratio"])
        log.info("Statistical Outlier Removal → %d 点", len(points))
    if len(points) == 0:
        raise ValueError("前処理後の点数が 0 です。切り出し範囲や入力ファイルを確認してください。")
    n_processed = len(points)

    # ---- 2. グリッド範囲 ---------------------------------------------------------
    if mcfg["bounds"]["enabled"]:
        b = mcfg["bounds"]
        spec = GridSpec.from_bounds(b["min_x"], b["max_x"], b["min_y"], b["max_y"], mcfg["resolution"])
    else:
        spec = GridSpec.from_points(points, mcfg["resolution"], mcfg["padding"])
    if float(mcfg["origin_yaw"]) != 0.0:
        warnings.append(
            f"map.origin_yaw = {mcfg['origin_yaw']} が設定されていますが，グリッドは GLIM 座標軸に平行なまま"
            "回転されません。map.yaml の origin に yaw がそのまま書き出されるため，通常は 0.0 にしてください。"
        )

    # ---- 3. 床面推定と高さ ----------------------------------------------------------
    hf = cfg["height_filter"]
    floor = estimate_floor(points, hf["floor"])
    warnings += floor.warnings
    warnings += check_vertical_axis(points, floor, mcfg["vertical_axis_check"])
    floor_heights = floor.signed_height(points)
    if hf["mode"] == "relative_to_floor":
        obstacle_heights = floor_heights
        basis = "床面からの相対高さ"
    else:
        obstacle_heights = points[:, 2]
        basis = "GLIM 座標系の絶対 Z"
    ob = hf["obstacle"]
    log.info("障害物判定: %s %.2f〜%.2f m，min_points_per_cell = %d", basis, ob["min_height"], ob["max_height"], ob["min_points_per_cell"])

    # ---- 4. Occupied / Free / Unknown ------------------------------------------
    obstacle_mask, _ = build_obstacle_mask(
        spec, points, obstacle_heights, ob["min_height"], ob["max_height"], ob["min_points_per_cell"]
    )
    estimator = create_estimator(cfg["free_space"]["method"])
    fs = estimator.estimate(
        FreeSpaceContext(spec, points, floor_heights, obstacle_mask, floor, cfg["free_space"])
    )
    warnings += fs.warnings
    if fs.info:
        log.info("Free 推定 (%s): %s", estimator.name, fs.info)
    grid = compose_grid(obstacle_mask, fs.free_mask)

    # ---- 5. 後処理 -------------------------------------------------------------------
    grid, post_stats, post_warn = postprocess_grid(grid, fs.free_mask, cfg["postprocessing"])
    warnings += post_warn
    if post_stats:
        log.info("後処理: %s", post_stats)

    warnings += check_pgm_values_decodable(cfg["nav2"])
    stats = grid_statistics(grid)
    result = ConversionResult(grid, spec, stats, n_input, n_processed, warnings=warnings, log_path=log_path)

    # ---- 6. 書き出し -----------------------------------------------------------------
    if write_outputs:
        name = cfg["output"]["map_name"]
        result.pgm_path = save_pgm(out_dir / f"{name}.pgm", grid_to_pgm(grid))
        meta = build_map_yaml(f"{name}.pgm", spec, cfg["nav2"], mcfg["origin_yaw"])
        result.yaml_path = save_map_yaml(out_dir / f"{name}.yaml", meta)
        if cfg["output"]["save_preview"] and cfg["visualization"]["enabled"]:
            from visualization import save_preview  # matplotlib は必要なときだけ読み込む

            result.preview_path = save_preview(
                out_dir / "preview.png", grid, spec, cfg["visualization"], stats, title=str(pcd_path.name)
            )

    print_summary(result, time.perf_counter() - t0)
    return result


def print_summary(r: ConversionResult, elapsed: float) -> None:
    s, g = r.stats, r.spec
    lines = [
        "=" * 70,
        f"入力点数            : {r.n_input_points:,}",
        f"前処理後の点数      : {r.n_processed_points:,}",
        f"地図範囲 X          : [{g.origin_x:.3f}, {g.max_x:.3f}] m",
        f"地図範囲 Y          : [{g.origin_y:.3f}, {g.max_y:.3f}] m",
        f"map.yaml origin     : [{g.origin_x:.3f}, {g.origin_y:.3f}, 0.0]",
        f"画像サイズ          : {g.width} x {g.height} px (幅 x 高さ)",
        f"解像度              : {g.resolution} m/cell",
        f"Free                : {s['free']:>10,} cells ({s['free_ratio']*100:6.2f}%)",
        f"Occupied            : {s['occupied']:>10,} cells ({s['occupied_ratio']*100:6.2f}%)",
        f"Unknown             : {s['unknown']:>10,} cells ({s['unknown_ratio']*100:6.2f}%)",
    ]
    if r.pgm_path:
        lines += [
            f"出力 PGM            : {r.pgm_path}",
            f"出力 YAML           : {r.yaml_path}",
        ]
        if r.preview_path:
            lines.append(f"出力 preview        : {r.preview_path}")
        if r.log_path:
            lines.append(f"出力 log            : {r.log_path}")
    lines.append(f"処理時間            : {elapsed:.2f} s")
    lines.append("=" * 70)
    for ln in lines:
        log.info(ln)
    for w in r.warnings:
        log.warning(w)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GLIM PCD → Nav2 OccupancyGrid 変換")
    parser.add_argument("--config", required=True, help="config.yaml のパス")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    try:
        run_conversion(cfg, base_dir=cfg_path.parent)
    except Exception as e:  # noqa: BLE001
        logging.getLogger().error("変換に失敗しました: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
