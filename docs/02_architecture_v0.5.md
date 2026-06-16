# 02_architecture_v0.5.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | gl_widget.py追加・base_mode設計明確化・OpenGL統一案採用・ライブラリ追記 | 承認済み |
| v0.3 | F-09操作ガイドをQLabel（main_window.py管理）で対応・PyQt6-Qt6をライブラリ表から削除 | 承認済み |
| v0.4 | draw()シグネチャをdraw(frame, results, width, height)に修正（gl_context引数を削除） | 承認済み |
| v0.5 | Mode2/3 マネキン・動画再生/書出・トレイル・時系列グラフ・ControlPanel 群分離・AppSettings / PoseStream による Model 層導入・内部 FBO 二段化・ffmpeg 音声 mux を反映 | 要承認 |

---

# システム構造設計：AIスケルトン体験デモ

## ディレクトリ構成

```
skeleton_opencanvas/
│
├── main.py                    # エントリーポイント
├── config.yaml                # システム設定（カメラ番号・ポーズ・モデルパス等）
├── user_settings.json         # ユーザー調整値の永続化（git 管理外、起動時 load／終了時 save）
├── requirements.txt           # 依存パッケージ一覧
│
├── app/
│   ├── __init__.py
│   ├── main_window.py         # メインウィンドウ・モード切替・キー操作・ウィジェット配置
│   ├── gl_widget.py           # QOpenGLWidget。内部 FBO 1280x720 二段描画
│   ├── capture_worker.py      # QThread。カメラ取得スレッドと推定スレッドを分離
│   ├── camera.py              # カメラ／動画ファイルの統一インターフェース
│   ├── pose_estimator.py      # MediaPipe Pose Landmarker。EMA 平滑化、num_poses 動的変更
│   ├── pose_constants.py      # ランドマーク定数・POSE_CONNECTIONS
│   ├── config_loader.py       # config.yaml の読み込み
│   ├── user_settings.py       # user_settings.json の load/save
│   ├── app_settings.py        # AppSettings(QObject) — Model 層、各値の変更 signal
│   ├── pose_stream.py         # PoseStream / PoseFrame — フレームブローカ
│   ├── camera_overlay.py      # Mode2/3 の「実写半透明オーバーレイ」描画
│   ├── scene_objects.py       # Mode3 の 3D 背景オブジェクト（床／柱／壁）
│   ├── t_pose.py              # T ポーズ固定ランドマーク（造形確認用）
│   ├── trails.py              # TrailBuffer — 両手両足の軌跡
│   ├── video_export.py        # VideoExporter — 入力動画 → Mode2 で MP4 書出 + 音声 mux
│   │
│   ├── modes/
│   │   ├── __init__.py
│   │   ├── base_mode.py       # BaseMode 抽象基底クラス
│   │   ├── mode1_overlay.py   # モード1：オーバーレイ
│   │   ├── mode2_mannequin.py # モード2：背景画像 + マネキン正射影 + 実写オーバーレイ
│   │   └── mode3_3d.py        # モード3：3D 空間 + マネキン透視投影（視点回転）
│   │
│   ├── mannequin/
│   │   ├── __init__.py
│   │   ├── primitives.py      # 球・カプセル・テーパード円柱・角丸プリズム描画
│   │   ├── mannequin_renderer.py  # マネキン本体描画 + トレイル管理 + 可視性制御
│   │   ├── gltf_loader.py     # GLTF モデル読込（mesh スタイル用）
│   │   └── skinning.py        # スキニング計算（mesh スタイル用、参考実装）
│   │
│   └── ui/
│       ├── __init__.py
│       └── control_panels.py  # 9 個の ControlPanel + TimeSeriesGraph + 共有スタイル定数
│
├── assets/
│   ├── backgrounds/           # モード2 用背景画像
│   ├── characters/            # モード3 用キャラクター素材（GLTF）
│   └── models/                # MediaPipe Pose Landmarker モデル（git 管理外）
│
├── sample_vid/                # 動作確認用サンプル動画（git 管理外）
│
└── docs/                      # 設計ドキュメント
```

## モジュール責務（層別）

### Model 層（パラメータと到達フレームの集約）

| モジュール | 責務 |
|-----------|------|
| `app/app_settings.py` | `AppSettings(QObject)`。スライダー値・UI トグルなど全パラメータの一元管理。値ごとに `pyqtSignal` を持ち、変更時に自動 emit。`user_settings.load()` を初期値ソースに使い、`save()` でディスクへ。 |
| `app/pose_stream.py` | `PoseStream(QObject)` + `PoseFrame` dataclass。`CaptureWorker.frame_ready` を購読し、シーク世代弾き／`t_video` 計算をしてから `frame_arrived(PoseFrame)` で配る薄いブローカ。ライブも動画書出も同じ push 経路を共有できる構造。 |
| `app/user_settings.py` | `user_settings.json` の load/save 関数。`DEFAULTS` 辞書を保持。`AppSettings` から呼ばれる。 |

### Controller 層（イベント／配線）

| モジュール | 責務 |
|-----------|------|
| `main.py` | アプリ起動・例外ハンドリング・logging 設定。 |
| `app/main_window.py` | ウィンドウ生成・モード切替・キー入力・各 ControlPanel と Renderer の wiring・オーバーレイ位置計算。 |
| `app/capture_worker.py` | カメラ取得スレッドと推定スレッドを分離。`frame_ready(np.ndarray, list, float, int, int)` シグナルで（frame, results, fps, frame_idx, seek_gen）を emit。emit を 30fps に上限制限。`seek()` / `switch_source()` で `_seek_gen` を +1 する。 |

### View / Rendering 層

| モジュール | 責務 |
|-----------|------|
| `app/gl_widget.py` | `QOpenGLWidget`。`paintGL` を二段化（Stage1: 内部 FBO 1280x720 にモード描画、Stage2: 画面サイズの quad に拡大表示）。B キーのボーンオーバーレイは `draw_bone_overlay` で外部からも呼べる。 |
| `app/modes/mode1_overlay.py` | カメラ映像テクスチャ + ボーン線オーバーレイ。 |
| `app/modes/mode2_mannequin.py` | 背景画像クロップ + 実写半透明オーバーレイ + MannequinRenderer の正射影描画。 |
| `app/modes/mode3_3d.py` | 3D 空間（床/柱）+ 実写半透明オーバーレイ + MannequinRenderer の透視投影描画（視点回転）。 |
| `app/mannequin/mannequin_renderer.py` | プリミティブ／メッシュ両スタイル切替、トレイル描画（GL_LINE_STRIP + アンチエイリアス）、`_mannequin_visible` と `_trail_visible` を独立制御。`draw_ortho` / `draw_perspective` で trail buffer 更新と描画も担当。 |
| `app/mannequin/primitives.py` | 球・カプセル・テーパード円柱・角丸プリズム。`gluNewQuadric()` はモジュール単一インスタンスを共有（リソースリーク防止）。 |
| `app/camera_overlay.py` | カメラフレームを半透明テクスチャとして描画。Mode2/3 共用。 |
| `app/trails.py` | `TrailBuffer`。両手両足の軌跡を deque で保持。認識失敗時は buffer をクリアして「点も線も同時に消える」一元設計。 |
| `app/ui/control_panels.py` | `TimeSeriesGraph`（pyqtgraph）+ 9 個の ControlPanel。共有スタイル定数（VCTL_STYLE/VCTL_BAR_STYLE/BTN_STYLE）と `SPEED_CYCLE` も保持。 |
| `app/video_export.py` | `VideoExporter`。元動画を 1 フレームずつ読み、`Mode2.draw` をオフスクリーン FBO に描画、cv2.VideoWriter で出力、ffmpeg で元音声を mux。グラフ widget は `QGraphicsView.render` で右上に焼き込み。 |

### Common

| モジュール | 責務 |
|-----------|------|
| `app/camera.py` | カメラ／動画ファイル両対応の統一インターフェース。`is_video_file`・`source_fps`・`frame_pos`・`seek`・`loop`・`speed`・`switch_source` をサポート。プラットフォーム別バックエンド（Windows: DSHOW、macOS: AVFOUNDATION）。 |
| `app/pose_estimator.py` | MediaPipe Pose Landmarker。`set_num_poses` で動的に Landmarker を再生成。EMA 平滑化（`smoothing_alpha`）、world_landmarks の合わせて返却。`reset_timestamp` でシーク時の単調増加制約を緊急回避。 |
| `app/config_loader.py` | `config.yaml` の load と `get` / `set`。 |
| `app/pose_constants.py` | `PoseLandmark` 定数と `POSE_CONNECTIONS`。 |
| `app/t_pose.py` | `t_pose_result()`：T ポーズ固定の `PoseLandmarkResult` を返す（M/T キーの造形確認用）。 |
| `app/scene_objects.py` | Mode3 の 3D 背景（床・柱・壁）を描画。 |

## レイヤー間のデータフロー

```
[Camera ／ VideoFile]
        │ read_frame
        ▼
[CaptureWorker._camera_loop] (Thread A)        ← latest_frame / latest_frame_idx を保存
        ▼ (lock-protected buffer)
[CaptureWorker.run] (Thread B)                 ← MediaPipe estimate、EMA 平滑化
        ▼ frame_ready signal (30fps cap)
        │   (frame, results, fps, frame_idx, seek_gen)
        ▼
[PoseStream.push] (Main thread)                ← seek_gen 弾き + t_video 計算
        ▼ frame_arrived(PoseFrame)
        ▼
[MainWindow._on_frame_arrived]
        ├─ GLWidget.update_frame() → paintGL → 内部 FBO で Mode.draw → 画面に拡大
        ├─ TimeSeriesGraph.append(x_values, y_values, t_override=pf.t_video)
        ├─ VideoControlPanel.update_state(...)
        ├─ Mode3ControlPanel.update_state(...)
        └─ DebugLabel.setText(...)
```

`MainWindow` は Controller として `wiring` に専念し、計算（seek_gen 判定／`t_video`）は PoseStream、状態保持は各 View／AppSettings が担う。

### スライダー操作のデータフロー

```
[QSlider valueChanged]
        ▼
[ControlPanel._on_slider_changed]
        ▼ size_changed / alpha_changed 等の signal
        ▼
[MainWindow._on_xxx_changed handler]
        ├─ AppSettings.set("key", v)
        │       ▼ "key"_changed signal
        │       （現状は MainWindow が renderer 操作を継続。将来は Renderer が直接 subscribe する余地）
        └─ Renderer.set_xxx(v) / Estimator.set_xxx(v) / etc.
```

### 動画書出のデータフロー

```
[VideoExporter.run]                                     ← メインスレッドで実行
  loop:
    cv2.VideoCapture.read()
    PoseEstimator.estimate(frame)
    graph_update_cb(results, t_video)                   ← MainWindow._append_graphs_for_export
    gl_widget.makeCurrent()
    fbo.bind()
      Mode2.draw(frame, results, width, height)
      gl_widget.draw_bone_overlay(...)                  ← B キー有効時
    fbo.release()
    fbo.toImage() → numpy(BGR)
    overlay_graphs(arr, graph_widgets)                  ← TimeSeriesGraph を画面右上に貼り込み
    writer.write(arr)
    QCoreApplication.processEvents()                    ← UI 応答性
  writer.release()
  _mux_audio_from_source(out_path, in_path)             ← ffmpeg で元音声を mux
```

`gl_widget.setUpdatesEnabled(False)` でライブ描画を停止、グラフ widget だけ updates 復活させて `QGraphicsView.render()` でアルファチャネル含めて取得する。

## 描画の二段化（GLWidget）

```
[GLWidget.paintGL]
  ┌── Stage 1 ──────────────────────────────────┐
  │ self._internal_fbo.bind()                    │
  │ glViewport(0, 0, 1280, 720)                  │
  │ active_mode.draw(frame, results, 1280, 720)  │ ← モードは常に固定解像度で描く
  │ if show_bones: draw_bone_overlay(...)        │
  │ fbo.release()                                │
  └──────────────────────────────────────────────┘
  ┌── Stage 2 ──────────────────────────────────┐
  │ glViewport(0, 0, win_w, win_h)               │
  │ FBO のカラーテクスチャを 1 quad で拡大表示    │ ← ウィンドウサイズが大きくても塗りが軽い
  └──────────────────────────────────────────────┘
```

ウィンドウサイズに比例して塗りつぶしコストが上がらないため、ウィンドウ拡大時もメインスレッドが詰まらない。書出時は別 FBO（元動画サイズ）を使うので影響なし。

## モード状態のリセット連動

シーク／ソース切替時に時系列・連続性に依存する状態をまとめてリセットする：

| 対象 | 何が壊れるか | リセット手段 |
|-----|------------|-------------|
| Graph (X / Y) | 時間軸が連続のまま跳ぶと過去データが残る | `TimeSeriesGraph.reset()` |
| Trail Buffer | 旧位置と新位置を線で結んでしまう | `Renderer.reset_trail()` |
| PoseEstimator | VIDEO モードはタイムスタンプ単調増加が前提、EMA は前回値依存 | `reset_timestamp()` |
| PoseStream / Worker | キューに残る旧 emit を弾く | `seek_gen += 1` → `PoseStream.set_seek_gen` |

これらは `MainWindow._on_seek` で一括して呼ばれる。

## ライブラリ構成

| ライブラリ | 用途 |
|-----------|------|
| `mediapipe` | Pose Landmarker による骨格推定 |
| `opencv-python` | カメラ／動画キャプチャ・出力（VideoCapture / VideoWriter） |
| `PyQt6` | GUI ウィンドウ・イベント管理・QOpenGLWidget |
| `PyOpenGL` | OpenGL 描画 |
| `PyOpenGL-accelerate` | PyOpenGL の高速化 |
| `numpy` | フレーム・ランドマーク・グラフバッファの数値処理 |
| `PyYAML` | `config.yaml` の読み込み |
| `Pillow` | 背景画像の OpenGL テクスチャ変換 |
| `pyqtgraph` | 時系列グラフ描画（QGraphicsView 上に PlotItem） |
| `ffmpeg`（外部実行ファイル） | 動画書出時の音声 mux（PATH または Shotcut/Krita/OBS 同梱版を自動検出） |

## 設計原則

- **1 ファイル = 1 責務**。Model（AppSettings/PoseStream）・View（ControlPanel/Graph/GLWidget）・Controller（MainWindow）の層分離を意識する。
- スライダー値などのパラメータは **AppSettings に一元集約**。直接 dict や個別属性を MainWindow に持たない。
- フレーム到着の **判定・計算は PoseStream に集約**。MainWindow に判断ロジックを書かない。
- 全モードは単一の `QOpenGLWidget` 上に統一し、内部 FBO 二段化でウィンドウサイズと描画コストを分離する。
- ライブと書出の処理を **同じパイプラインに乗せる**（PoseStream.push + Mode.draw + Renderer + Graph append + 必要なら overlay）。
- モード追加・キャラクター差し替え・背景差し替えは `app/modes/`・`assets/` のファイル変更と `config.yaml` の編集のみで完結させる。
- ユーザー操作で変わる値は `user_settings.json` に自動永続化、システム設定は `config.yaml`（git 管理）。
