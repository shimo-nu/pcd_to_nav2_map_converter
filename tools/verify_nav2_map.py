#!/usr/bin/env python3
"""生成した map.pgm / map.yaml が Nav2 Humble の map_server で意図通りに読まれるかを確認する。

2 つのモードがある。

(1) オフライン検証（ROS 2 不要・常に実行可能）
    python tools/verify_nav2_map.py output/map.yaml

    nav2_map_server の map_io.cpp (Humble) の trinary 変換規則を Python で再現し，
    PGM の画素が Free(0) / Occupied(100) / Unknown(-1) の 3 値だけに分類されること，
    画素値ごとの分類結果，画像サイズ・origin・resolution を表示する。
    ※ 実機 map_server の代替ではない。閾値設定の整合性チェックである。

(2) ROS 2 による実機検証（ROS 2 Humble + nav2_map_server が必要）
    別端末で map_server を起動してから:
        ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=$PWD/output/map.yaml
        ros2 lifecycle set /map_server configure && ros2 lifecycle set /map_server activate
    本スクリプトを --ros 付きで実行:
        python tools/verify_nav2_map.py output/map.yaml --ros

    /map (nav_msgs/OccupancyGrid) を 1 回受信し，data 配列・解像度・origin・サイズが
    本ツールの出力（オフライン検証結果）と完全一致するかを判定する。
    rclpy がインポートできない環境では実行せず，その旨を表示して終了する。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_exporter import load_map_yaml, load_pgm, nav2_trinary_decode  # noqa: E402
from occupancy_grid import FREE, OCCUPIED, UNKNOWN  # noqa: E402

LABEL = {FREE: "Free(0)", OCCUPIED: "Occupied(100)", UNKNOWN: "Unknown(-1)"}


def offline_decode(yaml_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta = load_map_yaml(yaml_path)
    pgm = load_pgm(yaml_path.parent / meta["image"])
    if meta.get("mode", "trinary") != "trinary":
        raise SystemExit(f"mode={meta.get('mode')} は未対応（trinary のみ）")
    decoded_img = nav2_trinary_decode(pgm, int(meta["negate"]), float(meta["occupied_thresh"]), float(meta["free_thresh"]))
    # map_server は画像を上下反転して data[row*width + col]（row 0 = origin 側）に格納する
    data = np.flipud(decoded_img)
    return meta, pgm, data


def report_offline(meta: dict, pgm: np.ndarray, data: np.ndarray) -> bool:
    h, w = pgm.shape
    print(f"image      : {meta['image']}  ({w} x {h} px)")
    print(f"resolution : {meta['resolution']} m/cell")
    print(f"origin     : {meta['origin']}")
    print(f"thresholds : negate={meta['negate']} occupied_thresh={meta['occupied_thresh']} free_thresh={meta['free_thresh']}")
    print("画素値ごとの trinary 分類 (Humble map_io.cpp 再現):")
    ok = True
    for px in np.unique(pgm):
        val = int(nav2_trinary_decode(np.array([px], dtype=np.uint8), int(meta["negate"]),
                                      float(meta["occupied_thresh"]), float(meta["free_thresh"]))[0])
        n = int((pgm == px).sum())
        print(f"  pixel {int(px):3d} -> {LABEL[val]:14s} ({n} cells)")
    expected = {254: FREE, 0: OCCUPIED, 205: UNKNOWN}
    for px, exp in expected.items():
        if (pgm == px).any():
            val = int(nav2_trinary_decode(np.array([px], dtype=np.uint8), int(meta["negate"]),
                                          float(meta["occupied_thresh"]), float(meta["free_thresh"]))[0])
            if val != exp:
                print(f"  NG: pixel {px} は {LABEL[exp]} であるべきですが {LABEL[val]} と解釈されます")
                ok = False
    extra = set(np.unique(pgm).tolist()) - set(expected)
    if extra:
        print(f"  注意: 想定外の画素値 {sorted(extra)} が含まれています")
        ok = False
    total = data.size
    for v in (FREE, OCCUPIED, UNKNOWN):
        n = int((data == v).sum())
        print(f"{LABEL[v]:14s}: {n:>10d} cells ({100.0*n/total:6.2f}%)")
    print("オフライン検証:", "OK" if ok else "NG")
    return ok


def verify_with_ros(meta: dict, expected_data: np.ndarray, topic: str, timeout_s: float) -> bool:
    try:
        import rclpy
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError:
        print("rclpy / nav_msgs をインポートできません。ROS 2 Humble 環境を source してから再実行してください。")
        print("（この環境では ROS 検証は実行されていません）")
        return False

    rclpy.init()
    node = rclpy.create_node("pcd_to_nav2_map_verifier")
    received: list = []
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(OccupancyGrid, topic, lambda m: received.append(m), qos)
    print(f"{topic} を待機中 (最大 {timeout_s} s) ...")
    end = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
    while rclpy.ok() and not received and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if not received:
        print("OccupancyGrid を受信できませんでした。map_server が activate されているか確認してください。")
        return False

    msg = received[0]
    info = msg.info
    h, w = expected_data.shape
    ok = True
    def check(name, got, exp, tol=1e-6):
        nonlocal ok
        good = abs(got - exp) <= tol
        ok &= good
        print(f"  {name:12s}: ros={got}  expected={exp}  {'OK' if good else 'NG'}")
    check("width", info.width, w, 0)
    check("height", info.height, h, 0)
    check("resolution", info.resolution, float(meta["resolution"]), 1e-6)
    check("origin.x", info.origin.position.x, float(meta["origin"][0]), 1e-6)
    check("origin.y", info.origin.position.y, float(meta["origin"][1]), 1e-6)
    if info.width == w and info.height == h:
        ros_data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        mismatch = int((ros_data != expected_data).sum())
        print(f"  data mismatch: {mismatch} cells  {'OK' if mismatch == 0 else 'NG'}")
        for v in (FREE, OCCUPIED, UNKNOWN):
            print(f"    {LABEL[v]:14s}: ros={int((ros_data == v).sum()):>10d}  expected={int((expected_data == v).sum()):>10d}")
        ok &= mismatch == 0
    print("ROS 検証:", "OK" if ok else "NG")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_yaml", help="map.yaml のパス")
    ap.add_argument("--ros", action="store_true", help="map_server が publish する /map と比較する")
    ap.add_argument("--topic", default="/map")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    meta, pgm, data = offline_decode(Path(args.map_yaml).resolve())
    ok = report_offline(meta, pgm, data)
    if args.ros:
        ok &= verify_with_ros(meta, data, args.topic, args.timeout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
