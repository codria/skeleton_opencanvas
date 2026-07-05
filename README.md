# skeleton_opencanvas

MediaPipe + PyQt6 + OpenGL によるリアルタイム骨格推定デモアプリ。
カメラまたは動画ファイルを入力に、3 つの表示モード（オーバーレイ／マネキン／3D キャラクター）を
切り替えながら骨格推定結果を可視化する。動画ファイル再生時は MP4 への書き出しと
時系列グラフ／トレイル付きの解析動画生成にも対応する。

詳細は [`docs/`](docs/) 配下の各設計ドキュメント参照。

---

## 動作環境

| 項目 | 内容 |
|------|------|
| OS | Windows 10 / 11（macOS / Linux は要動作確認） |
| Python | 3.10 以上 |
| カメラ | USB カメラまたは内蔵カメラ |
| GPU | 不要（CPU で MediaPipe heavy が動く前提） |
| ffmpeg | 任意（動画書出時の音声 mux を行う場合のみ。Shotcut 同梱版でも可） |

---

## 初期セットアップ

### 1. リポジトリの取得

```powershell
git clone https://github.com/codria/skeleton_opencanvas.git
cd skeleton_opencanvas
```

### 2. Python 環境の作成（どちらか一方）

#### A) conda を使う場合（推奨）

```powershell
conda create -n env_skeleton_opencanvas python=3.10 -y
conda activate env_skeleton_opencanvas
pip install -r requirements.txt
```

#### B) venv を使う場合

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

主要パッケージ：`mediapipe` / `opencv-python` / `PyQt6` / `PyOpenGL` / `numpy` /
`PyYAML` / `Pillow` / `pyqtgraph` / `pygltflib`。

### 3. MediaPipe モデルファイルの配置

`assets/models/` ディレクトリは git 管理外（モデルが大きいため）。
以下から **heavy モデル** をダウンロードして配置する：

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

配置先：

```
skeleton_opencanvas/
└── assets/
    └── models/
        └── pose_landmarker_heavy.task
```

`config.yaml` の `pose.model_path` が `pose_landmarker.task` を指していても、
ファイルがなければ `pose_landmarker_heavy.task` を自動で探すので動く
（[main.py](main.py) のフォールバック解決）。

### 4. ffmpeg（動画書出に音声を付ける場合のみ）

任意。インストール例（どれか一つ）：

```powershell
winget install ffmpeg
# または
choco install ffmpeg
# または Shotcut/Krita/OBS をインストール済みなら自動検出されるので追加作業不要
```

入っていなければ書出される MP4 は **無音** になるだけで、動画書出処理自体は成功する。

### 5. （任意）`config.yaml` の編集

カメラデバイス番号が `0` で認識されない場合のみ。

```yaml
camera:
  device_index: 1   # 認識できる番号を試す
```

---

## 起動

```powershell
# 環境を有効化
conda activate env_skeleton_opencanvas   # または .venv\Scripts\activate

python main.py
```

起動成功すると、カメラ映像と骨格線オーバーレイ（モード 1）が表示される。

スライダーで調整した値は終了時に `user_settings.json` に自動保存され、次回起動時に復元される
（このファイルは git 管理外）。

---

## 基本操作

### キーボード

| キー | 動作 |
|------|------|
| `1` / `2` / `3` | モード切替（オーバーレイ／マネキン／3D キャラクター） |
| `B` | ボーン線オーバーレイ ON/OFF |
| `M` | マネキンスタイル切替（primitive / mesh） |
| `T` | T ポーズ固定（造形確認用） |
| `V` | 動画ファイル選択 |
| `C` | カメラ入力に復帰 |
| `Space` | 動画再生／一時停止 |
| `L` | 動画ループ ON/OFF |
| `G` | 時系列グラフ ON/OFF |
| `H` | UI（モード名・ガイド・デバッグ・スライダー）一括 ON/OFF |
| `+` / `-` | マネキンサイズ調整 |
| `Q` / `Esc` | アプリ終了（設定が自動保存される） |

### GUI ボタン

- 下部：モード切替ボタン、動画選択、カメラ復帰、背景画像差し替え
- 動画再生中のみ：再生／一時停止、シーク、時刻表示、ループ、速度（0.5×〜10×）、動画書出、サンプル書出（先頭 300 フレーム）

---

## 動画書出について

動画ファイルを再生中に「**動画書出**」ボタンを押すと、入力動画と同じ解像度・FPS で
Mode2 のレンダリング結果（マネキン＋実写オーバーレイ＋ボーン＋時系列グラフ）を
MP4 として書き出せる。元動画の音声トラックは ffmpeg で自動 mux される。

**処理時間の目安**：1 分の動画につき数分〜十数分（MediaPipe 推定が重いため）。
書出前に **サンプル書出**（先頭 300 フレームのみ）で見た目を確認するのを推奨。

---

## CLI ツール（GUI 不要）

### 動画 → ランドマーク CSV 抽出

`app/tools/extract_csv.py` は mp4 動画から MediaPipe Pose のランドマーク座標を
CSV に書き出す CLI ツール。GUI は起動しない。

```powershell
# 1 人・Long フォーマット（pandas 向け、default）
python -m app.tools.extract_csv input.mp4 output.csv

# 素人 / Excel で見るとき（Wide、1 行 1 フレーム、列がランドマーク別）
python -m app.tools.extract_csv input.mp4 output.csv --format wide

# 複数人
python -m app.tools.extract_csv input.mp4 output.csv --num-poses 2
```

出力は **MediaPipe の生値**（EMA 平滑化なし）。画像座標系（正規化 [0,1]）と
World 座標系（腰原点メートル）を同じ CSV に併記する。平滑化・フィルタは
後処理側で好きに適用できる。

### Long → Wide フォーマット変換

extract_csv デフォルトの Long 形式を Excel 向けの Wide 形式に変換する。

```powershell
python -m app.tools.long_to_wide input.csv              # in-place 上書き
python -m app.tools.long_to_wide input.csv output.csv   # 別ファイルに出力
```

### 動画書き出し（ヘッドレス）

GUI を起動せずに Mode2 の動画書き出しを行う。GUI で調整した見た目
（マネキンスタイル・トレイル・ボーン・グラフサイズ等）はそのまま
`user_settings.json` から流用される。

```powershell
# フル書き出し（<input>_export.mp4 を生成）
python -m app.tools.export_video input.mp4

# サンプル書き出し（先頭 300 フレームのみ、<input>_sample.mp4）
python -m app.tools.export_video input.mp4 --sample

# 出力先を指定
python -m app.tools.export_video input.mp4 -o my_output.mp4

# 任意のフレーム数
python -m app.tools.export_video input.mp4 --frames 500

# 音声 mux を省略（無音出力）
python -m app.tools.export_video input.mp4 --no-audio

# フォルダ内の全 mp4 を一括処理
for f in *.mp4; do python -m app.tools.export_video "$f"; done
```

---

## ドキュメント

| ファイル | 内容 |
|---------|------|
| [docs/00_overview_v0.4.md](docs/00_overview_v0.4.md) | プロジェクト概要・ユースケース |
| [docs/01_requirements_v0.6.md](docs/01_requirements_v0.6.md) | 機能要件 |
| [docs/02_architecture_v0.5.md](docs/02_architecture_v0.5.md) | システム構造（最重要） |
| [docs/03_interfaces_v0.5.md](docs/03_interfaces_v0.5.md) | 主要クラスの API |
| [docs/04_roadmap_v0.3.md](docs/04_roadmap_v0.3.md) | 実装ロードマップ |
| [docs/05_runbook_v0.3.md](docs/05_runbook_v0.3.md) | 実行手順書（詳細版） |

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `CameraNotFoundError` | カメラ接続確認・`config.yaml` の `device_index` を変更 |
| `ModuleNotFoundError` | `pip install -r requirements.txt` を再実行 |
| `pose_landmarker.task` が見つからない | セクション 3 を参照（モデルを `assets/models/` に配置） |
| 動画書出 MP4 が無音 | ffmpeg が見つからない → セクション 4 で導入、または Shotcut 等をインストール |
| ウィンドウサイズを大きくすると応答が悪い | 内部 FBO 二段化済みなので応答性は維持されるはず。pyqtgraph グラフを `G` で OFF にすると更に軽くなる |
| 両足が重なるシーンで推定がブレる | MediaPipe（単眼 2D 入力）の構造的限界。撮影アングル・距離で対応 |

詳細は [docs/05_runbook_v0.3.md](docs/05_runbook_v0.3.md) のセクション 3「起動時エラーの対処」参照。
