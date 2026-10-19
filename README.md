# pcd_to_nav2_map

GLIM（3D LiDAR SLAM）が出力した 3 次元点群地図（`.pcd`）から，ROS 2 Humble の Nav2 で
そのまま読み込める 2 次元 OccupancyGrid 地図（`map.pgm` / `map.yaml`）を生成するオフラインツールです。
ROS 2 の起動は不要で，すべてのパラメータは `config.yaml` で管理します。

```
map.pcd (GLIM, 3D) ──► 前処理 ──► 床面推定・高さフィルタ ──► Occupied / Free / Unknown ──► map.pgm + map.yaml
```

設計上の最重要点は **GLIM の地図座標系と Nav2 の地図座標系を一致させること** です。
PCD 上の点 `(x, y)` は，生成された OccupancyGrid 上でも同じ世界座標 `(x, y)` に対応します
（手動の位置合わせは不要）。

---

## 目次

1. [インストール](#1-インストール)
2. [依存ライブラリ](#2-依存ライブラリ)
3. [PCD ファイルの準備](#3-pcd-ファイルの準備)
4. [実行方法](#4-実行方法)
5. [config.yaml の各パラメータ](#5-configyaml-の各パラメータ)
6. [出力地図の確認方法](#6-出力地図の確認方法)
7. [Nav2 Humble の map_server での読み込み](#7-nav2-humble-の-map_server-での読み込み)
8. [GLIM の Localization と組み合わせる際の座標系の注意点](#8-glim-の-localization-と組み合わせる際の座標系の注意点)
9. [PCD のみから Free Space を推定することの限界](#9-pcd-のみから-free-space-を推定することの限界)
10. [テストと検証](#10-テストと検証)
11. [プログラム構成](#11-プログラム構成)

---

## 1. インストール

Python 3.9 以上（開発・検証は 3.11 で実施）。

```bash
cd pcd_to_nav2_map
python3 -m venv .venv            # もしくは: uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # もしくは: uv pip install -r requirements.txt
```

ROS 2 は不要です。ROS 2 環境を `source` したシェルでも動作しますが，Open3D と
ROS 2 の Python が競合する場合は別の仮想環境を使ってください。

## 2. 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| Open3D | PCD の読み込み，Voxel Downsampling，Statistical Outlier Removal，RANSAC 平面推定 |
| NumPy | 座標計算，セル集計（`np.bincount`），グリッド生成 |
| SciPy | 後処理（連結成分，膨張・収縮） |
| Pillow | 8bit PGM (P5) の書き出し |
| PyYAML | `config.yaml` の読み込み，`map.yaml` の書き出し |
| Matplotlib | `preview.png` の生成 |
| pytest | テスト |

## 3. PCD ファイルの準備

- GLIM が保存した **グローバル座標系（地図座標系）** の点群を用意します
  （GLIM のビューワの Save や `glim_rosbag` 実行後に保存される `map.pcd` など）。
- 前提: **Z 軸が鉛直上向き**であること。GLIM は IMU を使う構成では重力方向に整列した
  地図を出力しますが，整列していない場合はツールが床面の傾きから警告を出します
  （`map.vertical_axis_check`）。地図そのものは回転しないので，傾いている場合は GLIM 側で
  重力整列するか，傾きを補正した PCD を用意してください。
- XYZ 以外の属性（強度・色など）は無視されます。ASCII / バイナリどちらの PCD も読めます。
- 動作確認用の人工データが必要なら:

  ```bash
  python tools/generate_sample_pcd.py --out data/map.pcd
  ```

  （10 m × 6 m の部屋。床 z=0，天井 z=2.5，テーブル・柱つき）

## 4. 実行方法

```bash
python main.py --config config.yaml
```

`config.yaml` 内の相対パス（`input.pcd_path`，`output.directory`）は **config.yaml の置き場所** を
基準に解決されます。実行すると以下が表示・記録されます。

- 入力点群数と前処理後の点数
- 地図範囲（X/Y）と `map.yaml` の origin
- 生成画像のサイズ（px）と解像度
- Free / Occupied / Unknown のセル数と割合
- 出力ファイルのパス
- Free 推定・床面推定・座標系に関する警告（`WARNING`）

出力（`output.directory`）:

| ファイル | 内容 |
|---|---|
| `map.pgm` | 8bit グレースケール PGM（Free=254，Occupied=0，Unknown=205） |
| `map.yaml` | Nav2 map_server 用メタデータ |
| `preview.png` | 世界座標の軸付き確認画像（任意） |
| `conversion.log` | 使用した設定全文と処理ログ（任意） |

## 5. config.yaml の各パラメータ

`config.yaml` 本体にすべてのパラメータの役割と単位をコメントで記載しています。
ここでは要点のみまとめます。省略したキーは `config.py` の既定値で補完されます。

### input / output

| キー | 説明 |
|---|---|
| `input.pcd_path` | 入力 PCD |
| `output.directory` / `map_name` | 出力先と `<map_name>.pgm` / `.yaml` のベース名 |
| `output.save_preview` / `save_log` | `preview.png` / `conversion.log` の保存 |

### preprocessing（処理順に適用）

| キー | 説明 |
|---|---|
| `remove_invalid_points` | NaN / Inf 点の削除 |
| `height_crop` | 絶対 Z [m] での粗い切り出し（地下の外れ点・高所ノイズ除去） |
| `voxel_downsample.voxel_size` | Voxel Downsampling [m]。**`map.resolution` の半分以下を推奨**。解像度と同じにすると 1 セルあたり約 1 点になり `min_points_per_cell` で虫食いになる |
| `statistical_outlier` | Statistical Outlier Removal（`nb_neighbors`，`std_ratio`） |

> 「地面の推定と除去」は `height_filter` が担います。床付近の点は障害物にはならず
> （`obstacle.min_height` 未満），Free 推定のための床観測として別途使われます。

### map

| キー | 説明 |
|---|---|
| `resolution` | セル一辺 [m]（既定 0.05）。`map.yaml` の `resolution` にそのまま書かれる |
| `padding` | 自動範囲決定時の余白 [m]。origin = (min_x − padding, min_y − padding) |
| `bounds` | 地図範囲の明示指定（GLIM 座標系 [m]）。有効時は範囲外の点を捨て，origin = (min_x, min_y) |
| `origin_yaw` | `map.yaml` origin の yaw [rad]。グリッドは回転しないので通常 0.0（0 以外は警告） |
| `vertical_axis_check` | 床付近の点から法線の傾きを評価し `max_tilt_deg` を超えたら警告 |

### height_filter

| キー | 説明 |
|---|---|
| `mode` | `absolute`: 障害物高さを GLIM 座標系の絶対 Z で指定 / `relative_to_floor`: 推定床面からの相対高さで指定 |
| `floor.method` | `fixed`（水平床，`fixed_z` を床 Z とする）/ `percentile`（水平床，Z の下位パーセンタイル）/ `ransac`（Open3D RANSAC 平面，傾き可） |
| `floor.fixed_z` | **GLIM 座標系における床の絶対 Z [m]**。相対高さではない |
| `floor.ransac_*` | RANSAC の距離閾値・反復回数・使用点数上限・候補 Z 範囲 |
| `obstacle.min_height` / `max_height` | 障害物とみなす高さ範囲 [m]（基準は `mode`）。床のノイズより上，ロボット全高程度まで。天井はこの範囲外にする |
| `obstacle.min_points_per_cell` | 範囲内の点がこの数以上あるセルを Occupied |

`absolute` モードでも床面推定は実行され，`free_space.floor_based` の床観測判定に使われます。

### free_space

| 方法 | 動作 | 注意 |
|---|---|---|
| `floor_based`（既定） | 床面 ±`floor_height_tolerance` の点が `min_floor_points_per_cell` 以上あり，障害物でないセルを Free | 床が観測されていないセルは Unknown のまま。床観測セルが `min_floor_coverage_warn` 未満なら警告 |
| `unknown_default` | Free を生成しない（障害物以外はすべて Unknown） | Nav2 側で Unknown の扱いを設定する必要あり |
| `bounded_free` | 地図範囲内の非障害物セルをすべて Free | **未観測領域も Free になる**。壁の裏や部屋の外に経路が引かれ得る |

Occupied と Free の条件が重複した場合は常に Occupied が優先されます。

### postprocessing

| キー | 説明 |
|---|---|
| `remove_small_obstacles` | `min_size_cells` 未満の孤立 Occupied 成分を除去。除去後は床観測があれば Free，なければ Unknown |
| `fill_small_holes` | 障害物に囲まれた `max_size_cells` 以下の穴を Occupied で埋める |
| `fill_small_unknown_in_free` | Free に囲まれた小さな Unknown を Free にする（観測に基づかないため既定 OFF，ON 時は警告） |
| `erosion` / `dilation` | Occupied の収縮 / 膨張（3×3 カーネル × `iterations`）。**膨張は Nav2 の Inflation Layer と重複するため既定 OFF** |

Unknown を無条件に Free に変える処理はありません。

### nav2

`map.yaml` に書き出す `mode` / `negate` / `occupied_thresh` / `free_thresh`。
既定値（trinary，0，0.65，0.196）で画素値 254 / 0 / 205 が Free / Occupied / Unknown に
分類されることをツールが起動時に検証し，崩れていれば警告します。

### visualization

`preview.png` の生成可否，1 m グリッド線，軸表示，dpi。

## 6. 出力地図の確認方法

1. **`preview.png`** — 軸が GLIM 座標系 [m] になっています。赤 `+` が `map.yaml` の origin
   （左下セルの左下隅），青 `●` が世界座標 (0, 0)。壁や柱の位置が PCD と同じ座標に
   出ているか，上下左右が反転していないかを確認してください。
2. **`conversion.log`** — 使用した設定全文，床面推定結果（床 Z・傾き），床観測セルの被覆率，
   警告がすべて残ります。
3. **閾値の整合性チェック（ROS 不要）**

   ```bash
   python tools/verify_nav2_map.py output/map.yaml
   ```

   Nav2 Humble `map_io.cpp` の trinary 変換規則を再現し，PGM の各画素値がどの状態に
   解釈されるか，サイズ・origin・resolution を表示します。
4. **PCD と重ねて確認したい場合** — Open3D などで PCD を XY 平面に投影し，`preview.png`
   の軸と同じ範囲で描画すれば一致を目視できます（画像座標ではなく世界座標で比較すること）。

### PGM 画像座標と世界座標の関係

Nav2 と同じ規則です。画像の左上が (col=0, row=0)，row は下向きに増えます。

```
x = origin_x + (col + 0.5) * resolution
y = origin_y + (height - 1 - row + 0.5) * resolution
```

本ツール内部のグリッド配列は row 0 = 南（y 最小）で保持し，PGM 書き出し時に上下反転します。
map_server は読み込み時に同じ反転を行うため，`OccupancyGrid.data[0]` が origin のセルになります。

## 7. Nav2 Humble の map_server での読み込み

```bash
source /opt/ros/humble/setup.bash

# 単体起動
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=$PWD/output/map.yaml
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate

# 確認
ros2 topic echo /map --once --field info      # width / height / resolution / origin
```

Nav2 の bringup から使う場合は `map:=/path/to/output/map.yaml` を渡します。

```bash
ros2 launch nav2_bringup bringup_launch.py map:=$PWD/output/map.yaml use_sim_time:=false
```

### 各占有状態が保持されていることの確認

`map_server` が publish した `/map` と本ツールの出力を比較するスクリプトを用意しています。

```bash
# map_server を activate した状態で
python tools/verify_nav2_map.py output/map.yaml --ros
```

`/map` を 1 回受信し，`width` / `height` / `resolution` / `origin` と `data` 配列
（0 / 100 / −1 のセル数と全セルの一致）を判定します。

> **注記**: 本リポジトリの開発環境には ROS 2 がインストールされていないため，
> `--ros` モードおよび実機 map_server での読み込みは **未実行** です。
> 実行済みなのはオフライン検証（trinary 規則の再現）と pytest のみです。

Free の無い地図（`unknown_default`）を使う場合は，Costmap 側で
`track_unknown_space` / `unknown_cost_value` 等の設定が必要です。

## 8. GLIM の Localization と組み合わせる際の座標系の注意点

- 本ツールは点群を **回転・平行移動しません**。2D 地図の `(x, y)` は入力 PCD の `(x, y)` と
  同一で，`map.yaml` の `origin` は点群の XY 最小値 − `padding`（または `map.bounds`）です。
  したがって，GLIM の Localization が出力する地図座標系での姿勢を，そのまま Nav2 の
  `map` フレームの姿勢として使えます。
- Nav2 側では `map → odom → base_link` の TF が必要です。GLIM の Localization（例: glim の
  `localization` 構成や `glim_ros` の出力）が `map → base_link` のみを出す場合は，
  オドメトリと組み合わせて `map → odom` に変換して publish してください（AMCL は使わない）。
- **同じ PCD を Localization の参照地図に使う**ことが前提です。GLIM で地図を再最適化・
  再保存した場合は，2D 地図も必ず再生成してください（原点やドリフトが変わります）。
- `map.yaml` の origin yaw は 0 のままにしてください。`map.origin_yaw` を変えても
  グリッドは回転されません。
- Z 軸が鉛直でない地図（重力整列されていない GLIM 地図）では XY 投影が歪み，
  床面からの高さ判定も破綻します。ツールが `vertical_axis_check` で警告しますが，
  補正はしません。GLIM 側で重力整列してから使ってください。
- 2D 地図は `obstacle.max_height` 以下の構造しか含みません。Nav2 の障害物層
  （Obstacle / Voxel Layer）で実センサの点群から動的障害物と高さ方向の情報を補うことを前提にしてください。

## 9. PCD のみから Free Space を推定することの限界

PCD は **点の集合**であり，各点を観測した LiDAR の位置・方向を含みません。
そのため，OccupancyGrid の本来の作り方である **Ray Casting（センサから点までの線分を
Free にする）ができません**。「点が無いセル」は「空いているセル」ではなく，
「観測していないセル」かもしれません（壁の裏，部屋の外，ガラス越し，LiDAR の死角など）。

本ツールの `floor_based` は「床面の点が観測されている = そのセルまで LiDAR が届いた」
という間接的な証拠から Free を推定します。制約:

- LiDAR の取り付け高さや FoV によっては足元の床が観測されず，走行した場所でも Unknown が残る。
- 反射率の低い床，水濡れ，ガラス床などは床点が得られない。
- `voxel_size` が大きい，`min_floor_points_per_cell` が大きいと床観測セルが減る。
- 段差やスロープがある環境では水平床の仮定（`fixed` / `percentile`）が破綻する。
  RANSAC も単一平面しか扱わない。
- 床と同じ高さにある低い障害物（`min_height` 未満）は Free と判定される。

ツールは **未観測空間を Free として補完しません**。Unknown が多すぎて困る場合は，
Nav2 側で Unknown の扱いを設定するか，`bounded_free`（未観測も Free になる危険を理解した上で）
を選ぶか，将来の Ray Casting 推定器（GLIM の最適化済み軌跡と各スキャンが必要）を
`free_space.py` の `RayCastingEstimator` として実装してください。Free 推定部分は
`FreeSpaceEstimator` インタフェースで差し替え可能になっています。

## 10. テストと検証

```bash
python -m pytest tests -q
```

実装済みテスト（`tests/test_map_conversion.py`，21 件，実行済み・全件成功）:

| # | 指示書の項目 | テスト |
|---|---|---|
| 1 | 人工 PCD の障害物が意図した位置に置かれる | `test_obstacle_placed_at_expected_position`，`test_explicit_bounds_define_origin` |
| 2 | PGM の上下反転で XY が反転しない | `test_pgm_flip_preserves_world_orientation`，`test_grid_to_pgm_values`，`test_gridspec_mapping` |
| 3 | `map.yaml` の origin と解像度から元座標を復元できる | `test_yaml_origin_and_resolution_restore_coordinates` |
| 4 | Free / Occupied / Unknown の判定 | `test_three_state_classification`，`test_occupied_has_priority_over_free`，`test_min_points_per_cell`，`test_floor_based_warns_when_floor_missing` |
| 5 | 床・壁・天井を含む点群での高さフィルタ | `test_height_filter_floor_wall_ceiling`（床 z=0 / 1.3），`test_absolute_mode_uses_global_z`，`test_ransac_*`，`test_vertical_axis_warning_on_tilted_map` |
| 6 | Nav2 Humble map_server での状態保持 | `test_nav2_trinary_thresholds_keep_three_states`，`test_saved_files_decode_like_nav2`（trinary 規則の再現による検証） |
| – | 後処理 | `test_remove_small_obstacles_restores_free_or_unknown`，`test_dilation_expands_obstacles` |

**実行していない検証**: 実機の Nav2 Humble `map_server` での読み込み（ROS 2 未導入のため）。
手順は [§7](#7-nav2-humble-の-map_server-での読み込み) を参照してください。

## 11. プログラム構成

```
pcd_to_nav2_map/
├── config.yaml              # 全パラメータ（コメント付き）
├── config.py                # 読み込み・既定値補完・検証
├── main.py                  # エントリポイント（python main.py --config config.yaml）
├── pcd_loader.py            # PCD 読み込み，無効点除去
├── pointcloud_processor.py  # Voxel / SOR / 切り出し / 床面推定 / 鉛直軸検証
├── occupancy_grid.py        # GridSpec（座標変換），セル集計，3 値グリッド，PGM 変換，後処理
├── free_space.py            # Free Space 推定器（floor_based / unknown_default / bounded_free / 将来の ray_casting）
├── map_exporter.py          # map.pgm / map.yaml 書き出し，Nav2 trinary 規則の再現
├── visualization.py         # preview.png
├── synthetic.py             # 人工点群生成（テスト・サンプル共用）
├── tools/
│   ├── generate_sample_pcd.py   # サンプル PCD 生成
│   └── verify_nav2_map.py       # オフライン検証 / --ros で map_server と比較
├── tests/
│   ├── conftest.py
│   └── test_map_conversion.py
├── requirements.txt
└── README.md
```
