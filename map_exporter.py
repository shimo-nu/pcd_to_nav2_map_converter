"""map.pgm / map.yaml の書き出しと，Nav2 Humble trinary モードのデコード仕様。"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from occupancy_grid import FREE, OCCUPIED, UNKNOWN, GridSpec

log = logging.getLogger(__name__)


def save_pgm(path: str | Path, pgm: np.ndarray) -> Path:
    """uint8 (H,W) 配列を 8bit グレースケール PGM (P5) として保存する。"""
    path = Path(path)
    if pgm.dtype != np.uint8 or pgm.ndim != 2:
        raise ValueError("PGM 配列は uint8 の 2 次元配列である必要があります")
    Image.fromarray(pgm).save(path, format="PPM")  # 拡張子 .pgm + mode L → P5 で書かれる
    return path


def load_pgm(path: str | Path) -> np.ndarray:
    """PGM を uint8 (H,W) 配列として読み込む（検証・テスト用）。"""
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        return np.asarray(im, dtype=np.uint8).copy()


def build_map_yaml(image_name: str, spec: GridSpec, nav2_cfg: dict, origin_yaw: float = 0.0) -> dict:
    """Nav2 map_server 用メタデータ dict を作る。origin / resolution はグリッドから計算。"""
    return {
        "image": image_name,
        "mode": str(nav2_cfg["mode"]),
        "resolution": float(spec.resolution),
        "origin": [float(spec.origin_x), float(spec.origin_y), float(origin_yaw)],
        "negate": int(nav2_cfg["negate"]),
        "occupied_thresh": float(nav2_cfg["occupied_thresh"]),
        "free_thresh": float(nav2_cfg["free_thresh"]),
    }


def save_map_yaml(path: str | Path, meta: dict) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        # Nav2 の例と同じキー順で書く（sort_keys=False），origin はフロースタイル
        f.write(f"image: {meta['image']}\n")
        f.write(f"mode: {meta['mode']}\n")
        f.write(f"resolution: {meta['resolution']}\n")
        f.write(f"origin: [{meta['origin'][0]}, {meta['origin'][1]}, {meta['origin'][2]}]\n")
        f.write(f"negate: {meta['negate']}\n")
        f.write(f"occupied_thresh: {meta['occupied_thresh']}\n")
        f.write(f"free_thresh: {meta['free_thresh']}\n")
    return path


def load_map_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def nav2_trinary_decode(pgm: np.ndarray, negate: int, occupied_thresh: float, free_thresh: float) -> np.ndarray:
    """Nav2 Humble ``nav2_map_server/map_io.cpp`` の trinary モードを再現する。

    occ = pixel/255 (negate) または (255 - pixel)/255 (negate=0)。
    occ > occupied_thresh → 100，occ < free_thresh → 0，その他 → -1。
    PGM 画像の並び（row 0 = 上）で返す。実機の map_server の代替ではなく，
    閾値設定の整合性を確認するための再実装である。
    """
    px = pgm.astype(np.float64)
    occ = px / 255.0 if negate else (255.0 - px) / 255.0
    out = np.full(pgm.shape, UNKNOWN, dtype=np.int8)
    out[occ > occupied_thresh] = OCCUPIED
    out[occ < free_thresh] = FREE
    return out


def check_pgm_values_decodable(nav2_cfg: dict) -> list[str]:
    """Free=254 / Occupied=0 / Unknown=205 が設定した閾値で意図通りに分類されるか確認する。"""
    from occupancy_grid import PGM_FREE, PGM_OCCUPIED, PGM_UNKNOWN

    px = np.array([PGM_FREE, PGM_OCCUPIED, PGM_UNKNOWN], dtype=np.uint8)
    decoded = nav2_trinary_decode(px, nav2_cfg["negate"], nav2_cfg["occupied_thresh"], nav2_cfg["free_thresh"])
    expected = np.array([FREE, OCCUPIED, UNKNOWN], dtype=np.int8)
    problems = []
    for p, d, e in zip(px, decoded, expected):
        if d != e:
            problems.append(
                f"nav2 の閾値設定では画素値 {int(p)} が {int(d)} と解釈されます（期待: {int(e)}）。"
                "negate / occupied_thresh / free_thresh を確認してください。"
            )
    return problems
