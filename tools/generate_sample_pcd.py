#!/usr/bin/env python3
"""動作確認用の人工 PCD（床・壁・天井・テーブル・柱）を生成する。

    python tools/generate_sample_pcd.py --out data/map.pcd

生成される部屋は x∈[-5,5], y∈[-3,3], 床 z=0, 天井 z=2.5（GLIM 座標系を模した座標）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/map.pcd")
    ap.add_argument("--floor-z", type=float, default=0.0, help="床の Z [m]")
    ap.add_argument("--step", type=float, default=0.03, help="点の間隔 [m]")
    ap.add_argument("--noise", type=float, default=0.005, help="ガウスノイズの標準偏差 [m]")
    args = ap.parse_args()

    pts = synthetic.make_room(floor_z=args.floor_z, ceiling_z=args.floor_z + 2.5, step=args.step, noise=args.noise)
    # 床から立つ柱 2 本（床の観測は柱の周囲に残る）
    pts = np.vstack([
        pts,
        synthetic.box(1.85, 2.15, -0.15, 0.15, args.floor_z, args.floor_z + 2.5, args.step),
        synthetic.box(-0.15, 0.15, 1.35, 1.65, args.floor_z, args.floor_z + 1.0, args.step),
    ])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"wrote {out} ({len(pts)} points)")


if __name__ == "__main__":
    main()
