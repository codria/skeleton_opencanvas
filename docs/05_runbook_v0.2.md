# 05_runbook_v0.2.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | セクション1-6をdownload_model.py→手動ダウンロード手順に変更・assets/models/の配置手順を明記 | 承認済み |
| v1.0 | 人間承認・メジャーバージョン確定 | 承認済み |

---

# 実行手順書：AIスケルトン体験デモ

## 対象環境

| 項目 | 内容 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.10 以上 |
| エディタ | VSCode |
| カメラ | USBカメラまたは内蔵カメラ |

---

## 1. 環境構築（初回のみ）

### 1-1. Python のインストール確認

```bash
python --version
# Python 3.10.x 以上であることを確認
```

3.10未満の場合は https://www.python.org/ から最新版をインストールする。

### 1-2. リポジトリの配置

プロジェクトフォルダを任意の場所に配置する。

```
C:\Users\<ユーザー名>\projects\skeleton_opencanvas\
```

### 1-3. VSCode でプロジェクトを開く

```
File → Open Folder → skeleton_opencanvas フォルダを選択
```

### 1-4. 仮想環境の作成と有効化

VSCode のターミナル（Ctrl + @）で実行する。

```bash
# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化（Windows）
.venv\Scripts\activate

# 有効化されると (.venv) がプロンプトに表示される
# (.venv) C:\Users\...>
```

### 1-5. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

インストールされる主なパッケージ：

| パッケージ | 用途 |
|-----------|------|
| mediapipe | 骨格推定 |
| opencv-python | カメラキャプチャ・色変換 |
| PyQt6 | GUIウィンドウ |
| PyOpenGL | OpenGL描画 |
| PyOpenGL-accelerate | PyOpenGL高速化 |
| numpy | 数値処理 |
| PyYAML | 設定ファイル読み込み |
| Pillow | 背景画像変換 |

### 1-6. MediaPipe モデルファイルのダウンロード

以下のURLからモデルファイルを手動でダウンロードする。

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

ダウンロードしたファイルを以下の場所に配置する。

```
skeleton_opencanvas/
└── assets/
    └── models/
        └── pose_landmarker.task   ← ここに配置
```

配置後、`config.yaml` のモデルパスが正しいことを確認する。

```yaml
pose:
  model_path: "assets/models/pose_landmarker.task"
```

---

## 2. 設定ファイルの編集

`config.yaml` を環境に合わせて編集する。

```yaml
camera:
  device_index: 0    # カメラが認識されない場合は 1 や 2 に変更
  width: 1280
  height: 720
  fps: 30
```

### カメラデバイス番号の確認方法

複数カメラが接続されている場合、`device_index` を 0・1・2 と変えて試す。

---

## 3. アプリの起動

```bash
# 仮想環境が有効化されていることを確認してから実行
python main.py
```

### 起動確認

- PyQt6 ウィンドウが開くこと
- カメラ映像が表示されること
- 画面上に操作ガイドが表示されること

### 起動時エラーの対処

| エラー | 原因 | 対処 |
|--------|------|------|
| `CameraNotFoundError` ポップアップ | カメラが未接続・認識されていない | カメラを接続し直す・`device_index` を変更する |
| `ConfigLoadError` ポップアップ | `config.yaml` の書式が壊れている | `config.yaml` を初期値に戻す |
| `ModuleNotFoundError` | パッケージ未インストール | `pip install -r requirements.txt` を再実行する |
| モデルファイルが見つからない | `assets/models/pose_landmarker.task` が未配置 | セクション1-6の手順でモデルファイルを配置する |

---

## 4. 操作方法

| キー / 操作 | 動作 |
|------------|------|
| `1` キー | モード1（オーバーレイ）に切り替え |
| `2` キー | モード2（マネキン）に切り替え |
| `3` キー | モード3（3Dキャラクター）に切り替え |
| `F` キー | FPS・デバッグ情報の表示／非表示切り替え（デフォルト：非表示） |
| `H` キー | モード名・操作ガイド・デバッグ情報の表示／非表示切り替え（ボタンパネルは常時表示） |
| GUIボタン | 各モードに切り替え（キーと同等） |
| 背景選択ボタン | モード2の背景画像をダイアログから選択 |
| `Q` / `Esc` | アプリ終了 |

---

## 5. デモ当日の運用手順

1. PC を起動し、カメラを接続する
2. VSCode を開き、プロジェクトフォルダを開く
3. ターミナルで仮想環境を有効化する
   ```bash
   .venv\Scripts\activate
   ```
4. アプリを起動する
   ```bash
   python main.py
   ```
5. モード1が表示されることを確認する
6. 高校生を案内し、カメラ前の床のテープ内に立ってもらう
7. 1・2・3キーでモードを切り替えながらデモを行う
8. 来客が入れ替わる際は特別な操作は不要（カメラは継続動作）
9. デモ終了時は `Q` キーまたは `Esc` キーでアプリを終了する

---

## 6. カスタマイズ

### 背景画像の差し替え（モード2）

- `assets/backgrounds/` に任意の画像ファイル（`.jpg` / `.png`）を置く
- デモ中はGUIの背景選択ボタンからダイアログで切り替え可能
- デフォルト背景を変更する場合は `config.yaml` を編集する
  ```yaml
  mode2:
    background_image: "assets/backgrounds/任意のファイル名.jpg"
  ```

### キャラクターデザインの差し替え（モード3）

- `assets/characters/` に差し替え素材を配置する
- `config.yaml` を編集する
  ```yaml
  mode3:
    character_model: "assets/characters/任意のフォルダ名"
  ```

---

## 7. 異常終了時の対応

アプリが固まった・クラッシュした場合：

1. VSCode のターミナルで `Ctrl + C` を押す
2. 以下を確認してから再起動する
   - カメラが接続されているか
   - ターミナルにエラーログが出ていないか（logging出力を確認）
3. `python main.py` で再起動する

---

## 8. 仮想環境の終了

デモ終了後、ターミナルを閉じるだけでよい。
明示的に終了する場合は以下を実行する。

```bash
deactivate
```

---

## 9. .gitignore の設定

リポジトリ管理時に以下を `.gitignore` に含めること。

```
# 仮想環境
.venv/

# MediaPipe モデルファイル（大容量のためGit管理対象外）
assets/models/

# Python キャッシュ
__pycache__/
*.pyc
*.pyo

# VSCode 設定（任意）
.vscode/
```

特に `assets/models/` は数百MBになるため必ず除外すること。
モデルファイルはセクション1-6の手順で各自ダウンロードする。
