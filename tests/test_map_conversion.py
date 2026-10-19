"""pcd_to_nav2_map の変換テスト。

人工点群を PCD に書き出し，main.run_conversion で変換して検証する。
ROS 2 / Nav2 の実行は含まない（Nav2 Humble の trinary デコード仕様は
map_exporter.nav2_trinary_decode で再現し，それに対する検証のみ行う）。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

import synthetic
from config import DEFAULT_CONFIG, deep_merge, validate_config
from main import run_conversion
from map_exporter import load_map_yaml, load_pgm, nav2_trinary_decode
from occupancy_grid import (
    FREE,
    OCCUPIED,
    PGM_FREE,
    PGM_OCCUPIED,
    PGM_UNKNOWN,
    UNKNOWN,
    GridSpec,
    grid_to_pgm,
    pgm_to_grid,
)
from pointcloud_processor import FloorModel, estimate_floor

RES = 0.05


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------
def write_pcd(path, points: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    assert o3d.io.write_point_cloud(str(path), pcd)


def make_cfg(tmp_path, **overrides) -> dict:
    cfg = deep_merge(
        DEFAULT_CONFIG,
        {
            "input": {"pcd_path": str(tmp_path / "map.pcd")},
            "output": {"directory": str(tmp_path / "output"), "save_log": False},
            "preprocessing": {"voxel_downsample": {"enabled": True, "voxel_size": 0.02}},
            "map": {"resolution": RES, "padding": 0.5},
            "visualization": {"enabled": False},
        },
    )
    cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def convert(tmp_path, points: np.ndarray, **overrides):
    write_pcd(tmp_path / "map.pcd", points)
    cfg = make_cfg(tmp_path, **overrides)
    return run_conversion(cfg, base_dir=tmp_path)


def cell_value(result, x: float, y: float) -> int:
    """世界座標 (x, y) を含むセルのグリッド値。"""
    col, row, valid = result.spec.world_to_cell(np.array([[x, y]]))
    assert valid[0], f"({x}, {y}) は地図範囲外"
    return int(result.grid[row[0], col[0]])


def near_occupied(result, x: float, y: float, tol: float = RES) -> bool:
    """(x, y) の周囲 ±tol のセルのいずれかが Occupied か（境界上の壁の判定用）。"""
    xs = np.array([x - tol, x, x + tol])
    ys = np.array([y - tol, y, y + tol])
    gx, gy = np.meshgrid(xs, ys)
    col, row, valid = result.spec.world_to_cell(np.column_stack([gx.ravel(), gy.ravel()]))
    return bool(np.any(result.grid[row[valid], col[valid]] == OCCUPIED))


def floor_with_pillar(px: float, py: float, size: float = 0.3) -> np.ndarray:
    """10 m x 6 m の床（z=0）と，(px, py) を中心とする高さ 1.2 m の柱。"""
    floor = synthetic.plane_xy(-5.0, 5.0, -3.0, 3.0, 0.0, 0.02)
    h = size / 2
    pillar = synthetic.box(px - h, px + h, py - h, py + h, 0.0, 1.2, 0.02)
    return np.vstack([floor, pillar])


# ---------------------------------------------------------------------------
# 1. 人工 PCD の障害物が意図した位置に置かれる
# ---------------------------------------------------------------------------
def test_obstacle_placed_at_expected_position(tmp_path):
    r = convert(tmp_path, floor_with_pillar(2.0, 1.0))

    # 柱の中心と縁は Occupied
    assert cell_value(r, 2.0, 1.0) == OCCUPIED
    assert cell_value(r, 2.0 - 0.14, 1.0) == OCCUPIED
    # 柱から離れた床は Free，床のない余白は Unknown
    assert cell_value(r, 0.0, 0.0) == FREE
    assert cell_value(r, -2.0, 1.0) == FREE
    assert cell_value(r, 5.3, 0.0) == UNKNOWN

    # Occupied セルの重心が柱の中心に一致する
    rows, cols = np.nonzero(r.grid == OCCUPIED)
    cx, cy = r.spec.cell_to_world(cols, rows)
    assert abs(cx.mean() - 2.0) < RES and abs(cy.mean() - 1.0) < RES
    # Occupied セルは柱の周辺 (±0.2 m) に限られる
    assert np.all(np.abs(cx - 2.0) < 0.2 + RES) and np.all(np.abs(cy - 1.0) < 0.2 + RES)


# ---------------------------------------------------------------------------
# 2. PGM の上下反転で XY が反転しない
# ---------------------------------------------------------------------------
def test_pgm_flip_preserves_world_orientation(tmp_path):
    r = convert(tmp_path, floor_with_pillar(3.0, 2.0))
    pgm = grid_to_pgm(r.grid)
    H, W = pgm.shape

    # グリッド配列（row 0 = 南）では柱は上側（row 大），PGM（row 0 = 北）では上部（row 小）
    g_rows, g_cols = np.nonzero(r.grid == OCCUPIED)
    p_rows, p_cols = np.nonzero(pgm == PGM_OCCUPIED)
    assert g_rows.mean() > H / 2 and p_rows.mean() < H / 2
    assert g_cols.mean() > W / 2 and p_cols.mean() > W / 2  # x は反転しない
    np.testing.assert_array_equal(np.sort(p_cols), np.sort(g_cols))
    np.testing.assert_array_equal(np.sort(H - 1 - p_rows), np.sort(g_rows))

    # PGM 画素 → 世界座標の復元で柱の位置に戻る
    wx, wy = r.spec.image_to_world(p_cols, p_rows)
    assert abs(wx.mean() - 3.0) < RES and abs(wy.mean() - 2.0) < RES

    # 往復変換が恒等
    np.testing.assert_array_equal(pgm_to_grid(pgm), r.grid)


def test_grid_to_pgm_values():
    grid = np.array([[FREE, OCCUPIED], [UNKNOWN, FREE]], dtype=np.int8)
    pgm = grid_to_pgm(grid)
    # 上下反転: 元の row 1 が画像 row 0 に来る
    np.testing.assert_array_equal(pgm, [[PGM_UNKNOWN, PGM_FREE], [PGM_FREE, PGM_OCCUPIED]])
    assert pgm.dtype == np.uint8


# ---------------------------------------------------------------------------
# 3. map.yaml の origin と解像度から元の座標を復元できる
# ---------------------------------------------------------------------------
def test_yaml_origin_and_resolution_restore_coordinates(tmp_path):
    r = convert(tmp_path, floor_with_pillar(-1.5, -2.0))
    meta = load_map_yaml(r.yaml_path)
    pgm = load_pgm(r.pgm_path)

    # origin は点群の XY 最小値 - padding
    assert meta["origin"][0] == pytest.approx(-5.0 - 0.5, abs=1e-9)
    assert meta["origin"][1] == pytest.approx(-3.0 - 0.5, abs=1e-9)
    assert meta["origin"][2] == 0.0
    assert meta["resolution"] == RES
    assert meta["image"] == "map.pgm"
    assert pgm.shape == (r.spec.height, r.spec.width)

    # Nav2 と同じ規則: 画像 (col, row) → x = ox + (col+0.5)*res, y = oy + (H-1-row+0.5)*res
    ox, oy, _ = meta["origin"]
    res = meta["resolution"]
    H = pgm.shape[0]
    rows, cols = np.nonzero(pgm == PGM_OCCUPIED)
    x = ox + (cols + 0.5) * res
    y = oy + (H - 1 - rows + 0.5) * res
    assert abs(x.mean() + 1.5) < res and abs(y.mean() + 2.0) < res
    assert x.min() >= -1.5 - 0.15 - res and x.max() <= -1.5 + 0.15 + res
    assert y.min() >= -2.0 - 0.15 - res and y.max() <= -2.0 + 0.15 + res


def test_explicit_bounds_define_origin(tmp_path):
    r = convert(
        tmp_path,
        floor_with_pillar(0.0, 0.0),
        map={"bounds": {"enabled": True, "min_x": -2.0, "max_x": 3.0, "min_y": -1.0, "max_y": 1.0}},
    )
    meta = load_map_yaml(r.yaml_path)
    assert meta["origin"][:2] == [-2.0, -1.0]
    assert (r.spec.width, r.spec.height) == (100, 40)
    assert cell_value(r, 0.0, 0.0) == OCCUPIED
    assert cell_value(r, 2.9, 0.9) == FREE


# ---------------------------------------------------------------------------
# 4. Free / Occupied / Unknown の判定
# ---------------------------------------------------------------------------
def test_three_state_classification(tmp_path):
    # 床 (x<0 側のみ) + 柱 + 床のない場所に浮いた天井片
    floor = synthetic.plane_xy(-4.0, 0.0, -2.0, 2.0, 0.0, 0.02)
    pillar = synthetic.box(-2.15, -1.85, -0.15, 0.15, 0.0, 1.2, 0.02)
    ceiling_only = synthetic.plane_xy(2.0, 3.0, -1.0, 1.0, 2.5, 0.02)
    pts = np.vstack([floor, pillar, ceiling_only])

    r = convert(tmp_path, pts)  # floor_based
    assert cell_value(r, -2.0, 0.0) == OCCUPIED          # 柱
    assert cell_value(r, -3.0, 1.0) == FREE              # 床あり・障害物なし
    assert cell_value(r, 2.5, 0.0) == UNKNOWN            # 天井のみ: 障害物でも Free でもない
    assert cell_value(r, 1.0, 0.0) == UNKNOWN            # 何もない
    assert r.stats["unknown"] > 0 and r.stats["free"] > 0

    r = convert(tmp_path, pts, free_space={"method": "unknown_default"})
    assert r.stats["free"] == 0
    assert cell_value(r, -2.0, 0.0) == OCCUPIED
    assert cell_value(r, -3.0, 1.0) == UNKNOWN

    r = convert(tmp_path, pts, free_space={"method": "bounded_free"})
    assert r.stats["unknown"] == 0
    assert cell_value(r, -2.0, 0.0) == OCCUPIED
    assert cell_value(r, 1.0, 0.0) == FREE               # 未観測でも Free になる（簡易モード）
    assert any("bounded_free" in w for w in r.warnings)


def test_occupied_has_priority_over_free(tmp_path):
    # 柱のセルには床点もあるが Occupied が優先される
    r = convert(tmp_path, floor_with_pillar(1.0, 1.0))
    assert cell_value(r, 1.0, 1.0) == OCCUPIED


def test_min_points_per_cell(tmp_path):
    # 1 点だけの障害物は min_points_per_cell=2 では無視され，=1 では Occupied
    pts = np.vstack([synthetic.plane_xy(-2.0, 2.0, -2.0, 2.0, 0.0, 0.02), [[1.0, 1.0, 0.8]]])
    r = convert(tmp_path, pts, preprocessing={"voxel_downsample": {"enabled": False}})
    assert cell_value(r, 1.0, 1.0) == FREE
    r = convert(
        tmp_path, pts,
        preprocessing={"voxel_downsample": {"enabled": False}},
        height_filter={"obstacle": {"min_points_per_cell": 1}},
    )
    assert cell_value(r, 1.0, 1.0) == OCCUPIED


def test_floor_based_warns_when_floor_missing(tmp_path):
    # fixed_z が実際の床 (z=0) と合っていない → 床点が無く Free が生成されない + 警告
    r = convert(tmp_path, floor_with_pillar(0.0, 0.0), height_filter={"floor": {"fixed_z": 1.0}})
    assert r.stats["free"] == 0
    # 柱の側面 (z≈1.0) がわずかに「床」と判定されるため，点数 0 ではなく被覆率不足の警告になる
    assert any(("床面付近" in w) or ("床観測セル" in w) for w in r.warnings)


# ---------------------------------------------------------------------------
# 5. 床・壁・天井を含む点群での高さフィルタリング
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("floor_z", [0.0, 1.3])
def test_height_filter_floor_wall_ceiling(tmp_path, floor_z):
    pts = synthetic.make_room(floor_z=floor_z, ceiling_z=floor_z + 2.5, with_ceiling=True, with_table=True)
    r = convert(tmp_path, pts, height_filter={"floor": {"fixed_z": floor_z}})

    assert near_occupied(r, -5.0, 0.0)                  # 壁 (x=-5)
    assert near_occupied(r, 0.0, 3.0)                   # 壁 (y=3)
    assert cell_value(r, -2.5, -1.5) == OCCUPIED       # テーブル天板 (床+0.7 m)
    assert cell_value(r, 0.0, 0.0) == FREE             # 床 + 天井のみ → 天井は障害物にならない
    assert cell_value(r, 3.0, -2.0) == FREE
    assert cell_value(r, 5.3, 0.0) == UNKNOWN          # 部屋の外（余白）

    # 天井を除いても同じ結果になる（天井は判定に影響しない）
    pts_no_ceiling = synthetic.make_room(floor_z=floor_z, ceiling_z=floor_z + 2.5, with_ceiling=False)
    r2 = convert(tmp_path, pts_no_ceiling, height_filter={"floor": {"fixed_z": floor_z}})
    np.testing.assert_array_equal(r.grid, r2.grid)


def test_absolute_mode_uses_global_z(tmp_path):
    floor_z = 1.3
    pts = synthetic.make_room(floor_z=floor_z, ceiling_z=floor_z + 2.5)
    # absolute: 障害物範囲は GLIM 座標系の絶対 Z で指定する
    r_abs = convert(
        tmp_path, pts,
        height_filter={
            "mode": "absolute",
            "floor": {"fixed_z": floor_z},
            "obstacle": {"min_height": floor_z + 0.10, "max_height": floor_z + 1.80},
        },
    )
    r_rel = convert(tmp_path, pts, height_filter={"mode": "relative_to_floor", "floor": {"fixed_z": floor_z}})
    np.testing.assert_array_equal(r_abs.grid, r_rel.grid)
    assert cell_value(r_abs, -2.5, -1.5) == OCCUPIED
    assert cell_value(r_abs, 0.0, 0.0) == FREE


def test_ransac_and_percentile_floor_estimation():
    floor_z = 0.42
    pts = synthetic.make_room(floor_z=floor_z, ceiling_z=floor_z + 2.5, noise=0.005)
    cfg = DEFAULT_CONFIG["height_filter"]["floor"]

    ransac = estimate_floor(pts, deep_merge(cfg, {"method": "ransac"}))
    assert ransac.method == "ransac"
    assert ransac.tilt_deg < 1.0
    assert abs(ransac.z_at(0.0, 0.0) - floor_z) < 0.02
    h = ransac.signed_height(np.array([[0.0, 0.0, floor_z + 1.0]]))
    assert abs(h[0] - 1.0) < 0.02

    perc = estimate_floor(pts, deep_merge(cfg, {"method": "percentile", "percentile": 2.0}))
    assert abs(perc.z_at(0.0, 0.0) - floor_z) < 0.03


def test_ransac_floor_in_pipeline(tmp_path):
    floor_z = 0.42
    pts = synthetic.make_room(floor_z=floor_z, ceiling_z=floor_z + 2.5)
    r = convert(tmp_path, pts, height_filter={"floor": {"method": "ransac"}})
    assert cell_value(r, 0.0, 0.0) == FREE
    assert cell_value(r, -2.5, -1.5) == OCCUPIED
    assert near_occupied(r, -5.0, 0.0)


def test_vertical_axis_warning_on_tilted_map(tmp_path):
    # 地図全体を X 軸まわりに 10 deg 回転 → Z が鉛直でない
    pts = synthetic.make_room(with_ceiling=False, with_table=False)
    th = np.deg2rad(10.0)
    rot = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    tilted = pts @ rot.T
    r = convert(tmp_path, tilted, height_filter={"floor": {"method": "ransac"}})
    assert any("傾いています" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# 6. Nav2 Humble map_server (trinary) での読み込み整合性
#    実機の map_server は使わず map_io.cpp の閾値規則を再現して確認する。
# ---------------------------------------------------------------------------
def test_nav2_trinary_thresholds_keep_three_states():
    nav2 = DEFAULT_CONFIG["nav2"]
    px = np.array([[PGM_FREE, PGM_OCCUPIED, PGM_UNKNOWN]], dtype=np.uint8)
    decoded = nav2_trinary_decode(px, nav2["negate"], nav2["occupied_thresh"], nav2["free_thresh"])
    np.testing.assert_array_equal(decoded, [[FREE, OCCUPIED, UNKNOWN]])


def test_saved_files_decode_like_nav2(tmp_path):
    r = convert(tmp_path, synthetic.make_room())
    meta = load_map_yaml(r.yaml_path)
    pgm = load_pgm(r.pgm_path)
    assert set(np.unique(pgm)) <= {PGM_FREE, PGM_OCCUPIED, PGM_UNKNOWN}
    with open(r.pgm_path, "rb") as f:
        assert f.read(2) == b"P5"  # 8bit バイナリ PGM

    decoded = nav2_trinary_decode(pgm, meta["negate"], meta["occupied_thresh"], meta["free_thresh"])
    # map_server は画像を上下反転して data に格納する → 本ツールのグリッドと一致
    np.testing.assert_array_equal(np.flipud(decoded), r.grid)
    assert set(np.unique(decoded)) == {FREE, OCCUPIED, UNKNOWN}


# ---------------------------------------------------------------------------
# 後処理
# ---------------------------------------------------------------------------
def test_remove_small_obstacles_restores_free_or_unknown(tmp_path):
    # 床上のノイズ柱（1 セル）と床のない場所の孤立点
    floor = synthetic.plane_xy(-2.0, 2.0, -2.0, 2.0, 0.0, 0.02)
    noise_on_floor = np.array([[1.01, 1.01, 0.5], [1.02, 1.02, 0.6]])
    noise_outside = np.array([[3.01, 0.01, 0.5], [3.02, 0.02, 0.6]])
    big = synthetic.box(-1.3, -0.7, -1.3, -0.7, 0.0, 1.0, 0.02)
    pts = np.vstack([floor, noise_on_floor, noise_outside, big])
    over = {"preprocessing": {"voxel_downsample": {"enabled": False}}, "map": {"padding": 1.5}}

    r = convert(tmp_path, pts, **over)
    assert cell_value(r, 1.01, 1.01) == OCCUPIED and cell_value(r, 3.01, 0.01) == OCCUPIED

    r = convert(tmp_path, pts, **over, postprocessing={"remove_small_obstacles": {"enabled": True, "min_size_cells": 3}})
    assert cell_value(r, 1.01, 1.01) == FREE       # 床観測あり → Free に戻る
    assert cell_value(r, 3.01, 0.01) == UNKNOWN    # 床観測なし → Unknown（Free にはしない）
    assert cell_value(r, -1.0, -1.0) == OCCUPIED   # 大きな障害物は残る


def test_dilation_expands_obstacles(tmp_path):
    r0 = convert(tmp_path, floor_with_pillar(0.0, 0.0))
    r1 = convert(tmp_path, floor_with_pillar(0.0, 0.0), postprocessing={"dilation": {"enabled": True, "iterations": 2}})
    assert r1.stats["occupied"] > r0.stats["occupied"]
    assert cell_value(r1, 0.15 + 2 * RES - 0.01, 0.0) == OCCUPIED
    assert any("膨張" in w for w in r1.warnings)


# ---------------------------------------------------------------------------
# 単体: GridSpec
# ---------------------------------------------------------------------------
def test_gridspec_mapping():
    spec = GridSpec(origin_x=-1.0, origin_y=-2.0, resolution=0.5, width=4, height=6)
    col, row, valid = spec.world_to_cell(np.array([[-1.0, -2.0], [0.99, 0.99], [1.0, 0.0], [-1.01, 0.0]]))
    np.testing.assert_array_equal(col, [0, 3, 4, -1])
    np.testing.assert_array_equal(row, [0, 5, 4, 4])
    np.testing.assert_array_equal(valid, [True, True, False, False])
    x, y = spec.cell_to_world(0, 0)
    assert (x, y) == (-0.75, -1.75)
    x, y = spec.image_to_world(0, 0)  # 画像左上 = 北西のセル
    assert (x, y) == (-0.75, 0.75)


def test_floor_model_horizontal():
    fm = FloorModel.horizontal(1.5, "fixed")
    np.testing.assert_allclose(fm.signed_height(np.array([[0, 0, 2.0], [3, 4, 1.0]])), [0.5, -0.5])
    assert fm.tilt_deg == 0.0
