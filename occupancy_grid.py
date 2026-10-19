"""2D OccupancyGrid の生成，座標変換，後処理。

座標系の約束
-----------
* 世界座標 (x, y) は GLIM の地図座標系 [m]。
* グリッド配列 ``grid[row, col]`` は row 0 が **南（y 最小）**，col 0 が **西（x 最小）**。
  これは Nav2 の ``OccupancyGrid.data``（row-major，先頭が origin セル）と同じ並び。
* PGM 画像は row 0 が **上（y 最大）**なので，書き出し時に上下反転する
  （``grid_to_pgm`` / ``pgm_to_grid``）。Nav2 の map_server も同じ反転を行って読み込む。
* セル (col, row) が覆う世界座標は
  ``[origin_x + col*res, origin_x + (col+1)*res) × [origin_y + row*res, origin_y + (row+1)*res)``。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

# OccupancyGrid の値
FREE = 0
OCCUPIED = 100
UNKNOWN = -1

# PGM 画素値（Nav2 trinary モードの既定閾値で上記 3 値に分類される）
PGM_FREE = 254
PGM_OCCUPIED = 0
PGM_UNKNOWN = 205


# ---------------------------------------------------------------------------
# グリッド仕様
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """グリッドの原点（左下セルの左下隅の世界座標），解像度，サイズ。"""

    origin_x: float
    origin_y: float
    resolution: float
    width: int   # [cells] x 方向
    height: int  # [cells] y 方向

    @classmethod
    def from_points(cls, points: np.ndarray, resolution: float, padding: float) -> "GridSpec":
        """点群の XY 範囲 + 余白から自動決定する。"""
        min_xy = points[:, :2].min(axis=0) - padding
        max_xy = points[:, :2].max(axis=0) + padding
        return cls._from_extent(min_xy[0], min_xy[1], max_xy[0], max_xy[1], resolution)

    @classmethod
    def from_bounds(cls, min_x: float, max_x: float, min_y: float, max_y: float, resolution: float) -> "GridSpec":
        """明示した地図範囲から決定する。"""
        return cls._from_extent(min_x, min_y, max_x, max_y, resolution)

    @classmethod
    def _from_extent(cls, min_x, min_y, max_x, max_y, resolution) -> "GridSpec":
        # origin は 1 µm に丸めて map.yaml を読みやすくする（セル幅と比べ無視できる）
        min_x, min_y = round(float(min_x), 6), round(float(min_y), 6)
        width = max(1, int(np.ceil((max_x - min_x) / resolution - 1e-9)))
        height = max(1, int(np.ceil((max_y - min_y) / resolution - 1e-9)))
        return cls(min_x, min_y, float(resolution), width, height)

    @property
    def max_x(self) -> float:
        return self.origin_x + self.width * self.resolution

    @property
    def max_y(self) -> float:
        return self.origin_y + self.height * self.resolution

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def world_to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """世界座標 (N,2) → (col, row, valid)。範囲外は valid=False。"""
        col = np.floor((xy[:, 0] - self.origin_x) / self.resolution).astype(np.int64)
        row = np.floor((xy[:, 1] - self.origin_y) / self.resolution).astype(np.int64)
        valid = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return col, row, valid

    def cell_to_world(self, col, row) -> tuple[np.ndarray, np.ndarray]:
        """セル (col, row) の中心の世界座標。"""
        x = self.origin_x + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin_y + (np.asarray(row) + 0.5) * self.resolution
        return x, y

    def image_to_world(self, px_col, px_row) -> tuple[np.ndarray, np.ndarray]:
        """PGM 画像座標（左上原点，px_row は下向き）→ セル中心の世界座標。"""
        row = self.height - 1 - np.asarray(px_row)
        return self.cell_to_world(px_col, row)


# ---------------------------------------------------------------------------
# セル単位の集計
# ---------------------------------------------------------------------------
def count_points_per_cell(spec: GridSpec, points: np.ndarray) -> np.ndarray:
    """各セルに落ちる点の数 (height, width) を np.bincount で高速に集計する。"""
    col, row, valid = spec.world_to_cell(points[:, :2])
    flat = row[valid] * spec.width + col[valid]
    counts = np.bincount(flat, minlength=spec.width * spec.height)
    return counts.reshape(spec.shape)


def build_obstacle_mask(
    spec: GridSpec,
    points: np.ndarray,
    heights: np.ndarray,
    min_height: float,
    max_height: float,
    min_points_per_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """高さ範囲 [min_height, max_height] 内の点が閾値以上あるセルを Occupied とする。

    heights は点ごとの判定用高さ（absolute なら Z，relative_to_floor なら床からの高さ）。
    戻り値: (occupied_mask (H,W) bool, 範囲内点数 (H,W) int)
    """
    in_range = (heights >= min_height) & (heights <= max_height)
    counts = count_points_per_cell(spec, points[in_range])
    return counts >= int(min_points_per_cell), counts


def compose_grid(occupied_mask: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    """Occupied / Free マスクから 3 値グリッドを作る。Occupied を優先。"""
    grid = np.full(occupied_mask.shape, UNKNOWN, dtype=np.int8)
    grid[free_mask] = FREE
    grid[occupied_mask] = OCCUPIED
    return grid


# ---------------------------------------------------------------------------
# PGM 画像との相互変換
# ---------------------------------------------------------------------------
def grid_to_pgm(grid: np.ndarray) -> np.ndarray:
    """3 値グリッド（row 0 = 南）→ PGM 画素配列 uint8（row 0 = 北，上下反転）。"""
    pgm = np.full(grid.shape, PGM_UNKNOWN, dtype=np.uint8)
    pgm[grid == FREE] = PGM_FREE
    pgm[grid == OCCUPIED] = PGM_OCCUPIED
    return np.ascontiguousarray(np.flipud(pgm))


def pgm_to_grid(pgm: np.ndarray) -> np.ndarray:
    """grid_to_pgm の逆変換（テスト・検証用）。"""
    img = np.flipud(pgm)
    grid = np.full(img.shape, UNKNOWN, dtype=np.int8)
    grid[img == PGM_FREE] = FREE
    grid[img == PGM_OCCUPIED] = OCCUPIED
    return np.ascontiguousarray(grid)


# ---------------------------------------------------------------------------
# 後処理
# ---------------------------------------------------------------------------
def _structure(connectivity: int) -> np.ndarray:
    return ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)


def _remove_small_components(mask: np.ndarray, min_size: int, connectivity: int) -> tuple[np.ndarray, int]:
    labels, n = ndimage.label(mask, structure=_structure(connectivity))
    if n == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    small = sizes < min_size
    removed = int(sizes[small].sum())
    return mask & ~small[labels], removed


def _enclosed_small_components(mask: np.ndarray, max_size: int, connectivity: int) -> np.ndarray:
    """mask の連結成分のうち，画像の縁に接せず size <= max_size のものを返す。"""
    labels, n = ndimage.label(mask, structure=_structure(connectivity))
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = (sizes <= max_size)
    keep[border] = False
    keep[0] = False
    return keep[labels]


def postprocess_grid(grid: np.ndarray, free_mask: np.ndarray, post_cfg: dict) -> tuple[np.ndarray, dict, list[str]]:
    """config の postprocessing に従って形態学的処理を行う。

    Occupied でなくなったセルは free_mask（Free 推定結果）があれば Free，なければ Unknown に戻す。
    Unknown を無条件に Free へ変える処理はない。
    戻り値: (grid, 統計 dict, 警告リスト)
    """
    stats: dict[str, int] = {}
    warnings: list[str] = []
    occ = grid == OCCUPIED

    c = post_cfg["remove_small_obstacles"]
    if c["enabled"]:
        occ, removed = _remove_small_components(occ, int(c["min_size_cells"]), int(c["connectivity"]))
        stats["removed_small_obstacle_cells"] = removed

    c = post_cfg["fill_small_holes"]
    if c["enabled"]:
        holes = _enclosed_small_components(~occ, int(c["max_size_cells"]), 4)
        stats["filled_hole_cells"] = int(holes.sum())
        occ = occ | holes

    c = post_cfg["erosion"]
    if c["enabled"] and int(c["iterations"]) > 0:
        before = int(occ.sum())
        occ = ndimage.binary_erosion(occ, structure=_structure(8), iterations=int(c["iterations"]))
        stats["eroded_cells"] = before - int(occ.sum())

    c = post_cfg["dilation"]
    if c["enabled"] and int(c["iterations"]) > 0:
        before = int(occ.sum())
        occ = ndimage.binary_dilation(occ, structure=_structure(8), iterations=int(c["iterations"]))
        stats["dilated_cells"] = int(occ.sum()) - before
        warnings.append(
            "障害物の膨張 (postprocessing.dilation) が有効です。Nav2 Costmap の Inflation Layer と"
            "二重に膨張するため，通常は無効を推奨します。"
        )

    grid = compose_grid(occ, free_mask)

    c = post_cfg["fill_small_unknown_in_free"]
    if c["enabled"]:
        unknown = grid == UNKNOWN
        cand = _enclosed_small_components(unknown, int(c["max_size_cells"]), 4)
        # 周囲が Free のみ（Occupied に接していない）成分だけを埋める
        touching_occ = ndimage.binary_dilation(grid == OCCUPIED, structure=_structure(8))
        labels, _ = ndimage.label(cand, structure=_structure(4))
        bad = np.unique(labels[touching_occ & cand])
        fill = cand & ~np.isin(labels, bad[bad > 0])
        grid[fill] = FREE
        stats["filled_unknown_in_free_cells"] = int(fill.sum())
        warnings.append(
            f"Free に囲まれた小さな Unknown {int(fill.sum())} セルを Free に変換しました"
            "（観測に基づかない推測です: postprocessing.fill_small_unknown_in_free）。"
        )

    return grid, stats, warnings


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def grid_statistics(grid: np.ndarray) -> dict[str, float]:
    total = grid.size
    n_free = int((grid == FREE).sum())
    n_occ = int((grid == OCCUPIED).sum())
    n_unk = int((grid == UNKNOWN).sum())
    return {
        "total": total,
        "free": n_free,
        "occupied": n_occ,
        "unknown": n_unk,
        "free_ratio": n_free / total,
        "occupied_ratio": n_occ / total,
        "unknown_ratio": n_unk / total,
    }
