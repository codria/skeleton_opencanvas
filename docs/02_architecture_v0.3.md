# 02_architecture_v0.3.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | gl_widget.py追加・base_mode設計明確化・OpenGL統一案採用・ライブラリ追記 | 承認済み |
| v0.3 | F-09操作ガイドをQLabel（main_window.py管理）で対応・PyQt6-Qt6をライブラリ表から削除 | 要修正 |
| v0.4 | draw()シグネチャをdraw(frame, results, width, height)に修正（gl_context引数を削除） | 承認済み |
| v1.0 | 人間承認・メジャーバージョン確定 | 承認済み |

---

# システム構造設計：AIスケルトン体験デモ

## ディレクトリ構成

```
skeleton_opencanvas/
│
├── main.py                    # エントリーポイント
├── config.yaml                # 設定ファイル（カメラ番号・背景画像パス等）
├── requirements.txt           # 依存パッケージ一覧
│
├── app/
│   ├── __init__.py
│   ├── main_window.py         # PyQt6 メインウィンドウ・モード切り替え制御・操作ガイドQLabel管理
│   ├── gl_widget.py           # QOpenGLWidget継承クラス・全モード共通の描画ループ
│   ├── capture_worker.py      # カメラキャプチャ・骨格推定のバックグラウンドスレッド
│   ├── camera.py              # カメラキャプチャ管理・起動時疎通確認
│   ├── pose_estimator.py      # MediaPipe Pose Landmarker 初期化・推定・結果正規化
│   ├── config_loader.py       # config.yaml の読み込みと値の提供
│   ├── pose_constants.py      # ランドマークインデックス定数・POSE_CONNECTIONS定義
│   │
│   └── modes/
│       ├── __init__.py
│       ├── base_mode.py       # 各モードの基底クラス（OpenGL描画インターフェース定義）
│       ├── mode1_overlay.py   # モード1：カメラ映像テクスチャ＋骨格線オーバーレイ
│       ├── mode2_mannequin.py # モード2：背景画像テクスチャ＋3Dマネキン正射影描画
│       └── mode3_3d.py        # モード3：3D空間＋キャラクター描画
│
├── app/mannequin/
│   ├── __init__.py
│   ├── mannequin_renderer.py  # 3Dマネキン描画クラス（モード2・3共用）
│   └── primitives.py          # 球・カプセルなどのOpenGL3Dプリミティブ描画関数
│
├── assets/
│   ├── backgrounds/           # モード2用背景画像置き場
│   ├── characters/            # モード3用キャラクターデザイン差し替え素材置き場
│   └── models/                # MediaPipe Pose Landmarker モデルファイル置き場
│
└── docs/                      # 設計ドキュメント置き場
```

## モジュール分割と責務

| モジュール | 責務 |
|-----------|------|
| `main.py` | アプリ起動・PyQt6 イベントループ開始・起動時エラー（CameraNotFoundError・ConfigLoadError）のハンドリング |
| `app/main_window.py` | GUIレイアウト・モード切り替え制御・ファイル選択ダイアログ・各種QLabelのオーバーレイ管理（モード名・ガイド・デバッグ・ボタンパネル） |
| `app/gl_widget.py` | QOpenGLWidget継承・ウィンドウ全体を占有・paintGL()でアクティブモードのdraw()を呼び出す |
| `app/capture_worker.py` | QThreadによるバックグラウンドスレッド・カメラ取得→骨格推定→シグナル送信のループ・FPS計測 |
| `app/camera.py` | カメラデバイス接続・解像度およびFPS設定・フレーム取得・起動時疎通確認 |
| `app/pose_estimator.py` | MediaPipe Pose Landmarker の初期化・骨格推定・結果の正規化・リソース解放（release） |
| `app/config_loader.py` | config.yaml の読み込みと値の提供・ランタイム上書き（set） |
| `app/pose_constants.py` | ランドマークインデックス定数・POSE_CONNECTIONS の定義（全モード共通参照） |
| `app/modes/base_mode.py` | 全モード共通の抽象基底クラス。`draw(frame, results, width, height)` を抽象メソッドとして定義 |
| `app/modes/mode1_overlay.py` | OpenGLテクスチャにカメラ映像を貼り、骨格線をプリミティブで重畳描画 |
| `app/modes/mode2_mannequin.py` | 背景画像をクロップ表示・MannequinRendererで3Dマネキンを正射影描画 |
| `app/modes/mode3_3d.py` | OpenGLで3D空間を生成しMannequinRendererで透視投影描画。視点をQTimerで自動回転 |
| `app/mannequin/mannequin_renderer.py` | 3Dマネキン描画クラス（モード2・3共用）。正射影/透視投影を切り替え可能 |
| `app/mannequin/primitives.py` | 球・カプセルなどのOpenGL3Dプリミティブ描画関数 |

## 依存関係

```
main.py
  └── main_window.py
        ├── capture_worker.py（QThread）
        │     ├── camera.py
        │     └── pose_estimator.py
        ├── gl_widget.py
        │     └── modes/
        │           ├── base_mode.py（基底）
        │           ├── mode1_overlay.py
        │           ├── mode2_mannequin.py
        │           └── mode3_3d.py
        └── config_loader.py
```

- `capture_worker.py` はバックグラウンドスレッド（QThread）でカメラ取得・骨格推定を実行する
- `capture_worker.py` は `frame_ready` シグナルで `main_window.py` にフレーム・推定結果・FPSを通知する
- `main_window.py` はシグナルを受け取り `gl_widget.update_frame()` を呼ぶ

## 描画方式の統一

全モードを単一の `QOpenGLWidget`（gl_widget.py）に統一する。
カメラ取得・骨格推定はバックグラウンドスレッド（CaptureWorker）で実行し、
メインスレッドの描画ループをブロックしない。

```
[CaptureWorker スレッド]
  → camera.py からフレーム取得
  → pose_estimator.py で骨格推定
  → frame_ready シグナル送信（frame, results, fps）

[メインスレッド]
  → main_window._on_frame_ready() でシグナル受信
  → gl_widget.update_frame() 呼び出し → 再描画トリガー
  → gl_widget.paintGL()
        → active_mode.draw(frame, results, width, height)
              モード1: カメラ映像テクスチャ（アスペクト比維持）→ 骨格線プリミティブ
              モード2: 背景画像テクスチャ（クロップ・画面全体）→ MannequinRenderer.draw_ortho()（正射影）
              モード3: 3D空間生成 → MannequinRenderer.draw_perspective()（透視投影・視点回転）
  → main_window がデバッグQLabelのテキストを更新（FPS・人数・解像度）

[オーバーレイQLabelたち（gl_widgetの子ウィジェット）]
  → モード名QL abel（左上）：常時表示、Hキーで非表示
  → ガイドラベル（右下・ボタンパネルの上）：常時表示、Hキーで非表示
  → デバッグラベル（モード名の下）：Fキーで表示、Hキーで非表示
  → ボタンパネル（下部オーバーレイ）：常時表示
```

- 全モードは `QOpenGLWidget` の `paintGL()` コンテキスト内で描画する
- `gl_widget.py` がアクティブモードの `draw()` を呼び出す唯一の窓口となる
- モード切り替えは `main_window.py` がアクティブモードを差し替えるだけでよい
- `gl_widget.py` はウィンドウ全体を占有するセントラルウィジェットとして配置する
- ボタンパネル・モード名・ガイド・デバッグ情報は全て `gl_widget.py` の子ウィジェット（QLabel・QWidget）としてオーバーレイ表示する
- FPS・デバッグ情報は `main_window.py` が管理するQLabelで表示し、OpenGLコンテキスト外で完結させる
- HキーでモードQL abel・ガイド・デバッグのみ非表示（ボタンパネルは常時表示）
- モード同士は互いに依存しない
- `camera.py` と `pose_estimator.py` はモードに依存しない
- モード切り替え時は `on_mode_enter()` でOpenGL状態（GL_LIGHTING・GL_DEPTH_TEST等）をリセットし、前モードの状態汚染を防ぐ
- `on_mode_enter()` はOpenGLコンテキスト確立後（`initializeGL()` 後）のみ呼ぶ

## ライブラリ構成

| ライブラリ | 用途 |
|-----------|------|
| `mediapipe` | Pose Landmarker による骨格推定 |
| `opencv-python` | カメラキャプチャ・フレーム取得 |
| `PyQt6` | GUIウィンドウ・イベント管理 |
| `PyOpenGL` | OpenGL描画（全モード共通） |
| `PyOpenGL-accelerate` | PyOpenGL の高速化 |
| `numpy` | フレーム・ランドマークデータの数値処理 |
| `PyYAML` | config.yaml の読み込み |
| `Pillow` | 背景画像のOpenGLテクスチャ変換（モード2） |

## 設計原則

- 1ファイル = 1責務
- 全モードを単一の QOpenGLWidget に統一し、モード切り替えをシームレスにする
- 操作ガイド表示はOpenGLコンテキスト外（QLabel）で管理し、描画ループと分離する
- モードの追加・削除は `modes/` 以下にファイルを追加するだけで対応できる構造とする
- キャラクター・背景のデザイン差し替えは `assets/` 以下のファイル変更と
  `config.yaml` の編集のみで完結させる
