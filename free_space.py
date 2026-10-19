"""Free Space（走行可能領域）の推定モジュール。

PCD は点の集合であり，LiDAR の各観測位置・観測方向を含まないため，
「点がないセル = 観測済みの空き空間」とは言えない。本モジュールは Free を
推定する方法を差し替え可能にし，将来 GLIM の最適化済み軌跡と各スキャンを使った
Ray Casting 推定器（``RayCastingEstimator``）を追加できるようにしている。

各推定器は ``FreeSpaceContext`` を受け取り，``FreeSpaceResult``（Free マスクと警告）を返す。
Occupied との重複は呼び出し側（``compose_grid``）で Occupied 優先に解決される。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from occupancy_grid import GridSpec, count_points_per_cell
from pointcloud_processor import FloorModel


@dataclass
class FreeSpaceContext:
    spec: GridSpec
    points: np.ndarray            # 前処理後の点群 (N,3)
    floor_heights: np.ndarray     # 各点の床面からの符号付き高さ (N,)
    obstacle_mask: np.ndarray     # Occupied マスク (H,W)
    floor: FloorModel
    config: dict                  # config.yaml の free_space セクション


@dataclass
class FreeSpaceResult:
    free_mask: np.ndarray
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


class FreeSpaceEstimator(ABC):
    name: str = "base"

    @abstractmethod
    def estimate(self, ctx: FreeSpaceContext) -> FreeSpaceResult:
        ...


class FloorBasedEstimator(FreeSpaceEstimator):
    """床面を観測した点が十分あり，かつ障害物がないセルを Free とする（既定）。

    床の点が存在する = そのセルの床面まで LiDAR が到達した，という間接的な
    観測証拠に基づく。床が観測されていないセル（遮蔽・LiDAR 下方死角・ガラス等）は
    Unknown のまま残る。
    """

    name = "floor_based"

    def estimate(self, ctx: FreeSpaceContext) -> FreeSpaceResult:
        cfg = ctx.config["floor_based"]
        tol = float(cfg["floor_height_tolerance"])
        is_floor = np.abs(ctx.floor_heights) <= tol
        floor_counts = count_points_per_cell(ctx.spec, ctx.points[is_floor])
        floor_cells = floor_counts >= int(cfg["min_floor_points_per_cell"])
        # Occupied との重複は compose_grid 側で Occupied 優先に解決するため，ここでは
        # 「床を観測したセル」をそのまま返す（後処理で障害物が消えた際に Free へ戻せる）。
        free_mask = floor_cells

        warnings = [
            "Free は『床を観測した点があるセル』のみです。床が観測されていない領域は Unknown のまま残り，"
            "未観測空間を Free として補完はしません（Ray Casting ではない簡易推定）。"
        ]
        n_floor_pts = int(is_floor.sum())
        coverage = float(floor_cells.sum()) / floor_cells.size
        if n_floor_pts == 0:
            warnings.append(
                "床面付近 (±%.2f m) の点が 1 点もありません。height_filter.floor の設定"
                "（fixed_z など）が実際の床高さと合っているか確認してください。Free は生成されません。" % tol
            )
        elif coverage < float(cfg["min_floor_coverage_warn"]):
            warnings.append(
                f"床観測セルが地図全体の {coverage*100:.1f}% しかありません。床点群が不十分な場合，"
                "Free 領域が極端に少ない地図になります。floor_height_tolerance / fixed_z / "
                "voxel_size を見直すか，free_space.method を再検討してください。"
            )
        return FreeSpaceResult(
            free_mask=free_mask,
            warnings=warnings,
            info={"floor_points": n_floor_pts, "floor_cells": int(floor_cells.sum()), "floor_coverage": coverage},
        )


class UnknownDefaultEstimator(FreeSpaceEstimator):
    """Free を推定しない。障害物以外はすべて Unknown。"""

    name = "unknown_default"

    def estimate(self, ctx: FreeSpaceContext) -> FreeSpaceResult:
        return FreeSpaceResult(
            free_mask=np.zeros(ctx.spec.shape, dtype=bool),
            warnings=[
                "free_space.method=unknown_default: Free セルは生成されません。Nav2 でこの地図を使うには"
                "Costmap 側で Unknown を通行可能に扱う設定（track_unknown_space 等）が必要です。"
            ],
        )


class BoundedFreeEstimator(FreeSpaceEstimator):
    """地図範囲内の障害物以外をすべて Free とする簡易モード。"""

    name = "bounded_free"

    def estimate(self, ctx: FreeSpaceContext) -> FreeSpaceResult:
        return FreeSpaceResult(
            free_mask=np.ones(ctx.spec.shape, dtype=bool),
            warnings=[
                "free_space.method=bounded_free: 地図範囲内の非 Occupied セルをすべて Free にしました。"
                "LiDAR が観測していない領域（壁の裏・部屋の外・余白）も Free になるため，"
                "実際には通れない場所へ経路が計画される危険があります。Unknown セルは 0 になります。"
            ],
        )


class RayCastingEstimator(FreeSpaceEstimator):
    """（未実装）GLIM の最適化済み軌跡 + 各 LiDAR スキャンによる Ray Casting。

    各スキャンのセンサ位置から各点までの線分を 2D グリッド上でトレースし，
    通過セルを Free とする。PCD 単体では実現できないため，軌跡とスキャン列の
    入力インタフェースを設計したうえで実装する。
    """

    name = "ray_casting"

    def estimate(self, ctx: FreeSpaceContext) -> FreeSpaceResult:
        raise NotImplementedError("ray_casting は未実装です（GLIM の軌跡とスキャン列が必要）")


_ESTIMATORS: dict[str, type[FreeSpaceEstimator]] = {
    FloorBasedEstimator.name: FloorBasedEstimator,
    UnknownDefaultEstimator.name: UnknownDefaultEstimator,
    BoundedFreeEstimator.name: BoundedFreeEstimator,
    RayCastingEstimator.name: RayCastingEstimator,
}


def create_estimator(method: str) -> FreeSpaceEstimator:
    try:
        return _ESTIMATORS[method]()
    except KeyError:
        raise ValueError(f"未知の free_space.method: {method}（選択肢: {sorted(_ESTIMATORS)}）") from None
