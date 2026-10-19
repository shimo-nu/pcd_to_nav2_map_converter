"""点群の前処理と床面推定。

すべて NumPy のベクトル演算または Open3D の組み込み処理で行い，
全点に対する Python ループは使わない。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 前処理
# ---------------------------------------------------------------------------
def _to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Open3D の Voxel Downsampling。"""
    if voxel_size <= 0:
        raise ValueError("voxel_size は正の値である必要があります")
    down = _to_o3d(points).voxel_down_sample(voxel_size)
    return np.asarray(down.points, dtype=np.float64)


def statistical_outlier_removal(points: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    """Open3D の Statistical Outlier Removal。"""
    filtered, _ = _to_o3d(points).remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
    )
    return np.asarray(filtered.points, dtype=np.float64)


def crop_xy(points: np.ndarray, min_x: float, max_x: float, min_y: float, max_y: float) -> np.ndarray:
    """XY 範囲 [min, max) で切り出す（上端は半開区間，グリッド境界と整合）。"""
    m = (
        (points[:, 0] >= min_x)
        & (points[:, 0] < max_x)
        & (points[:, 1] >= min_y)
        & (points[:, 1] < max_y)
    )
    return points[m]


def crop_z(points: np.ndarray, min_z: float, max_z: float) -> np.ndarray:
    """絶対 Z 範囲 [min_z, max_z] で切り出す。"""
    m = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    return points[m]


# ---------------------------------------------------------------------------
# 床面モデル
# ---------------------------------------------------------------------------
@dataclass
class FloorModel:
    """床平面 normal·p + offset = 0（normal は単位ベクトルで z 成分 > 0）。

    signed_height() は点の床面からの符号付き高さ（上向きが正）を返す。
    fixed / percentile では normal = (0, 0, 1)，offset = -floor_z。
    """

    method: str
    normal: np.ndarray
    offset: float
    n_inliers: int = 0
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def horizontal(cls, floor_z: float, method: str) -> "FloorModel":
        return cls(method=method, normal=np.array([0.0, 0.0, 1.0]), offset=-float(floor_z))

    def signed_height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset

    def z_at(self, x: float, y: float) -> float:
        """座標 (x, y) における床面の Z。"""
        a, b, c = self.normal
        return float(-(a * x + b * y + self.offset) / c)

    @property
    def tilt_deg(self) -> float:
        """法線の Z 軸からの傾き [deg]。"""
        return math.degrees(math.acos(float(np.clip(self.normal[2], -1.0, 1.0))))

    @property
    def is_horizontal(self) -> bool:
        return self.method in ("fixed", "percentile")


def _subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points and len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=int(max_points), replace=False)
        return points[idx]
    return points


def estimate_floor(points: np.ndarray, floor_cfg: dict) -> FloorModel:
    """config の height_filter.floor に従って床面を推定する。"""
    method = floor_cfg["method"]
    if method == "fixed":
        model = FloorModel.horizontal(floor_cfg["fixed_z"], method)
        log.info("床面: fixed, z = %.3f m（GLIM 座標系の絶対 Z）", floor_cfg["fixed_z"])
        return model

    if method == "percentile":
        p = float(floor_cfg["percentile"])
        floor_z = float(np.percentile(points[:, 2], p))
        model = FloorModel.horizontal(floor_z, method)
        log.info("床面: percentile(%.1f%%), z = %.3f m", p, floor_z)
        model.warnings.append(
            "床面を Z 値のパーセンタイルで推定しました。床より低い外れ点や段差があると"
            "床高さがずれるため，preview.png とログの床 Z を確認してください。"
        )
        return model

    if method == "ransac":
        return _estimate_floor_ransac(points, floor_cfg)

    raise ValueError(f"未知の床面推定方法: {method}")


def _estimate_floor_ransac(points: np.ndarray, floor_cfg: dict) -> FloorModel:
    cand = points
    zr = floor_cfg["ransac_z_range"]
    if zr["enabled"]:
        cand = crop_z(cand, zr["min_z"], zr["max_z"])
    cand = _subsample(cand, int(floor_cfg["ransac_max_points"]))
    if len(cand) < 3:
        raise ValueError("RANSAC 床面推定に使える点が不足しています（ransac_z_range を確認）")

    plane, inliers = _to_o3d(cand).segment_plane(
        distance_threshold=float(floor_cfg["ransac_distance_threshold"]),
        ransac_n=3,
        num_iterations=int(floor_cfg["ransac_iterations"]),
    )
    a, b, c, d = plane
    normal = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(normal)
    normal /= norm
    d /= norm
    if normal[2] < 0:  # 上向きに揃える
        normal, d = -normal, -d
    model = FloorModel(method="ransac", normal=normal, offset=float(d), n_inliers=len(inliers))

    cx, cy = float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))
    log.info(
        "床面: ransac, inliers = %d / %d, 傾き = %.2f deg, 地図中心 (%.2f, %.2f) での床 Z = %.3f m",
        len(inliers), len(cand), model.tilt_deg, cx, cy, model.z_at(cx, cy),
    )
    if model.tilt_deg > 30.0:
        model.warnings.append(
            f"RANSAC で推定した平面の傾きが {model.tilt_deg:.1f} deg と大きく，壁や斜面を"
            "床と誤認している可能性があります。ransac_z_range で候補を絞るか method=fixed を検討してください。"
        )
    return model


# ---------------------------------------------------------------------------
# 鉛直軸の検証
# ---------------------------------------------------------------------------
def check_vertical_axis(points: np.ndarray, floor: FloorModel, check_cfg: dict) -> list[str]:
    """床面付近の点を平面近似し，法線が Z 軸から傾いていないかを検証する。

    戻り値は警告メッセージのリスト（空なら問題なし）。
    fixed/percentile の床推定に対しては「床付近に点が十分あるか」の確認にもなる。
    """
    warnings: list[str] = []
    if not check_cfg["enabled"]:
        return warnings

    band = float(check_cfg["band_height"])
    h = floor.signed_height(points)
    near = points[np.abs(h) <= band]
    if len(near) < 100:
        warnings.append(
            f"床面推定値の ±{band:.2f} m 以内に点が {len(near)} 点しかなく，鉛直軸の検証ができません。"
            "床 Z (fixed_z 等) が実際の床と合っていない可能性があります。"
        )
        return warnings

    near = _subsample(near, int(check_cfg["sample_points"]), seed=1)
    centered = near - near.mean(axis=0)
    # 共分散の最小固有ベクトル = 床面の法線（PCA 平面近似）
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    tilt = math.degrees(math.acos(float(np.clip(abs(normal[2]), -1.0, 1.0))))
    log.info("鉛直軸検証: 床付近 %d 点の平面法線の Z 軸からの傾き = %.2f deg", len(near), tilt)
    if tilt > float(check_cfg["max_tilt_deg"]):
        warnings.append(
            f"床付近の点群から推定した法線が Z 軸から {tilt:.1f} deg 傾いています"
            f"（許容 {check_cfg['max_tilt_deg']} deg）。GLIM 地図の Z 軸が鉛直でない，"
            "または床面推定が誤っている可能性があります。XY 投影した 2D 地図が歪むため，"
            "GLIM 側で重力方向を合わせるか height_filter.floor.method=ransac を検討してください。"
        )
    return warnings
