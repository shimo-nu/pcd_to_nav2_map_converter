"""pytest 用: pcd_to_nav2_map/ をインポートパスに追加する。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
