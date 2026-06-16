# 05_runbook_v0.3.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | セクション1-6をdownload_model.py→手動ダウンロード手順に変更・assets/models/の配置手順を明記 | 承認済み |
| v0.3 | キー操作・動画再生／書出・ffmpeg 自動検出・user_settings.json・スライダー操作・現状の依存パッケージを追記 | 要承認 |

---

# 実行手順書：AIスケルトン体験デモ

## 対象環境

| 項目 | 内容 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.10 以上 |
| エディタ | VSCode |
| カメラ | USB カメラまたは内蔵カメラ |
| 動画再生時の音声 mux | ffmpeg（PATH または Shotcut/Krita/OBS 同梱版） |

---

## 1. 環境構築（初回のみ）

### 1-1. Python のインストール確認

```bash
python --version
# Python 3.10.x 以上であることを確認
```

3.10 未満の場合は https://www.python.org/ から最新版をインストールする。

### 1-2. リポジトリの配置

```
C:\Users\<ユーザー名>\projects\skeleton_opencanvas\
```

### 1-3. VSCode でプロジェクトを開く

`File → Open Folder → skeleton_opencanvas` フォルダを選択。

### 1-4. 仮想環境の作成と有効化

VSCode のターミナル（Ctrl+`@`）で：

```bash
python -m venv .venv
.venv\Scripts\activate
# (.venv) が表示されれば OK
```

### 1-5. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主なパッケージ：

| パッケージ | 用途 |
|-----------|------|
| `mediapipe` | 骨格推定 |
| `opencv-python` | カメラ・動画 I/O |
| `PyQt6` | GUI ウィンドウ |
| `PyOpenGL` | OpenGL 描画 |
| `PyOpenGL-accelerate` | PyOpenGL 高速化 |
| `numpy` | 数値処理 |
| `PyYAML` | システム設定の読込 |
| `Pillow` | 背景画像変換 |
| `pyqtgraph` | 時系列グラフ |

### 1-6. MediaPipe モデルファイルのダウンロード

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

を以下に配置：

```
skeleton_opencanvas/assets/models/pose_landmarker.task
```

配置後、`config.yaml` のモデルパスを確認：

```yaml
pose:
  model_path: "assets/models/pose_landmarker.task"
```

### 1-7. ffmpeg（動画書出の音声 mux 用、任意）

動画書出で **音声付き** の MP4 を出したい場合は ffmpeg が必要。

**自動検出される場所**：
1. PATH に `ffmpeg` が通っている（一番素直）
2. `C:\Program Files\Shotcut\ffmpeg.exe`（Shotcut 同梱版でも OK）
3. `C:\Program Files\Krita (x64)\bin\ffmpeg.exe`
4. `C:\Program Files\obs-studio\bin\64bit\ffmpeg.exe`

**インストール手段（推奨順）**：

```powershell
winget install ffmpeg
# または
choco install ffmpeg
# または
scoop install ffmpeg
```

いずれも入れずに Shotcut だけ入れてあっても動作する。検出できない場合は無音 MP4 で書き出される（処理は完了する）。

---

## 2. 設定ファイルの編集

### 2-1. システム設定（`config.yaml`、git 管理）

```yaml
camera:
  device_index: 0    # 認識しない場合は 1 や 2 を試す
  width: 1280
  height: 720
  fps: 30

display:
  width: 1280
  height: 720

pose:
  num_poses: 1                # ユーザー設定が優先される
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  smoothing_alpha: 0.25
  model_path: "assets/models/pose_landmarker.task"

mode2:
  background_image: "assets/backgrounds/default.jpg"
```

### 2-2. ユーザー設定（`user_settings.json`、git 管理外）

スライダー操作で自動更新・終了時に自動保存される。手動で削除すればデフォルト値に戻る。

```json
{
  "num_poses": 1,
  "smoothing_alpha": 0.25,
  "graph_scale": 1.0,
  "graph_visible": true,
  "show_bones": false,
  "trail_point_size": 6.0,
  "trail_line_width": 3.0,
  "trail_max_points": 32,
  "overlay_alpha": 1.0,
  "mode2_size_scale": 0.28,
  "mode3_speed": 30.0,
  "mode3_angle": 0.0
}
```

---

## 3. アプリの起動

```bash
.venv\Scripts\activate
python main.py
```

### 起動確認
- PyQt6 ウィンドウが開くこと
- カメラ映像が表示されること
- 画面上に操作ガイド・デバッグ情報が表示されること

### 起動時エラーの対処

| エラー | 原因 | 対処 |
|--------|------|------|
| `CameraNotFoundError` | カメラ未接続／認識されない | カメラを接続し直す・`device_index` を変更 |
| `ConfigLoadError` | `config.yaml` の書式エラー | 初期値に戻す |
| `ModuleNotFoundError` | パッケージ未インストール | `pip install -r requirements.txt` を再実行 |
| `pose_landmarker.task` が見つからない | モデル未配置 | セクション 1-6 を参照 |

---

## 4. 操作方法

### 4-1. キーボード

| キー | 動作 |
|------|------|
| `1` | モード1（オーバーレイ） |
| `2` | モード2（マネキン） |
| `3` | モード3（3D キャラクター） |
| `B` | ボーン線オーバーレイ表示／非表示 |
| `M` | マネキンスタイル切替（primitive / mesh） |
| `T` | T ポーズ固定（造形確認用） |
| `V` | 動画ファイル選択ダイアログ |
| `C` | カメラ入力に復帰 |
| `Space` | 動画 再生／一時停止 |
| `L` | 動画ループ on/off |
| `G` | 時系列グラフ単独 ON/OFF |
| `H` | モード名・ガイド・デバッグ・スライダー群を一括 ON/OFF（グラフは独立） |
| `+` / `-` | マネキンサイズ調整 |
| `Q` / `Esc` | アプリ終了（設定が自動保存される） |

### 4-2. GUI ボタン

下部パネル：
- `1: オーバーレイ` / `2: マネキン` / `3: 3D キャラ` — モード切替
- `動画選択` — ファイル選択ダイアログ
- `カメラ` — ライブカメラに復帰
- `背景選択` — Mode2 の背景画像差し替え

動画再生中だけ表示されるコントロールバー（画面下部）：
- `停止 / 再生` ボタン
- シークバー
- 時刻表示（mm:ss / mm:ss）
- `ループ ON/OFF`
- `1.0x` — 再生速度循環（0.5/1/2/3/5/10）
- `動画書出` — 全フレーム書出
- `サンプル書出` — 先頭 300 フレームだけ書出（確認用）

### 4-3. 左カラムのスライダー（常時表示、`H` キーで非表示）

| スライダー | 役割 |
|-----------|------|
| 検出人数 | num_poses（1〜3）。変更時 MediaPipe を再生成（1〜2 秒固まる） |
| 平滑化α | EMA 係数。小さいほど滑らか（追従遅れ大） |
| グラフ | グラフサイズ係数 0.3〜3.0（軸目盛・ラベルも連動拡大） |
| 軌跡 点 | トレイル点サイズ（0〜50px） |
| 軌跡 線 | トレイル線太さ（0〜25px） |
| 軌跡 長 | 保持点数（8〜256） |
| 実写透過 | Mode2/3 の実写オーバーレイ透過度 |
| 太さ | Mode2 マネキン太さ |
| 速度／角度（Mode3） | 回転速度（°/s）／視点角度（°） |

---

## 5. デモ当日の運用手順

1. PC を起動し、カメラを接続する
2. VSCode でプロジェクトフォルダを開く
3. ターミナルで仮想環境を有効化：`.venv\Scripts\activate`
4. アプリを起動：`python main.py`
5. モード1 が表示されることを確認
6. 高校生を案内し、カメラ前の床のテープ内に立ってもらう
7. `1`/`2`/`3` キー（または GUI ボタン）でモードを切り替えながらデモを行う
8. 来客が入れ替わる際は特別な操作は不要
9. デモ終了時は `Q` / `Esc` キー（設定は自動保存される）

---

## 6. 動画素材を使う場合

### 6-1. 動画ファイル再生

1. `V` キー or `動画選択` ボタンで mp4 / mov / avi / mkv / webm を選択
2. 自動的に再生開始。下部コントロールバーで操作できる
3. シーク・ループ・速度変更可能
4. `C` キー or `カメラ` ボタンでライブカメラに戻る

### 6-2. 動画 MP4 書出

1. 動画再生中にモード 2 を表示
2. 必要に応じて B キーでボーン線、グラフサイズ、トレイル設定を調整
3. `サンプル書出` ボタンを押して **先頭 300 フレーム** だけで見た目を確認
4. 出力ファイル名を指定（デフォルト：元動画名 + `_sample.mp4`）
5. 確認した結果が問題なければ `動画書出` で全フレーム書出
6. 元動画の音声が ffmpeg で自動 mux される

**処理時間の目安**：1 分の動画 ≒ 数分の書出時間（MediaPipe 推定が重いため）。

---

## 7. カスタマイズ

### 背景画像の差し替え（Mode2）

- `assets/backgrounds/` に `.jpg` / `.png` を置く
- デモ中は `背景選択` ボタンから切替可能
- デフォルト変更は `config.yaml`：
  ```yaml
  mode2:
    background_image: "assets/backgrounds/任意のファイル名.jpg"
  ```

### キャラクター差し替え（Mode3 mesh モード）

- `assets/characters/` に GLTF を配置
- `config.yaml` で参照

### スライダー初期値の差し替え

- `user_settings.json` を直接編集（次回起動時に反映）
- またはアプリ内で調整して `Q` で終了 → 自動保存

---

## 8. 異常終了時の対応

1. VSCode のターミナルで `Ctrl + C`
2. ログでエラー原因を確認
3. カメラ接続を確認
4. `python main.py` で再起動

書出中に異常終了した場合、未完成の `*.mp4` ファイルが残る可能性あり。手動で削除する。

---

## 9. 仮想環境の終了

```bash
deactivate
```

---

## 10. `.gitignore` の設定

```
# 仮想環境
.venv/

# MediaPipe モデルファイル（大容量のため git 管理対象外）
assets/models/

# サンプル動画・書き出した動画（大容量のため git 管理対象外）
sample_vid/

# 端末ごとのユーザー設定（スライダー値の永続化）
user_settings.json

# Python キャッシュ
__pycache__/
*.pyc
*.pyo

# VSCode 設定（任意）
.vscode/
```

- `assets/models/` は数百 MB
- `sample_vid/` は撮影素材で容量大（GB 単位）
- `user_settings.json` は端末固有値（共有不要）
