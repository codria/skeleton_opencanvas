# 04_roadmap_v0.2.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | Step依存関係の明記・動作確認基準の具体化・Step03動作確認修正・Step10にset_character追記・Step11に全モード完了条件追記 | 承認済み |
| v1.0 | 人間承認・メジャーバージョン確定 | 承認済み |

---

# 実装ロードマップ：AIスケルトン体験デモ

## 基本方針

```
1プロンプト = 1タスク
```

各ステップは独立して動作確認できる単位に分割する。
前のステップが完了・動作確認済みになってから次に進む。

---

## ステップ一覧

### Step 01：プロジェクト初期化
- ディレクトリ構成を作成する
- `requirements.txt` を作成する
- `config.yaml` の初期値を作成する
- 動作確認：`pip install -r requirements.txt` が通ること

### Step 02：ConfigLoader の実装
- `config_loader.py` を実装する（`get()` / `set()`）
- `ConfigLoadError` を定義する
- 動作確認：`config.yaml` の値を `get()` で取得できること

### Step 03：Camera の実装
- 前提：Step 02 完了
- `camera.py` を実装する（`start()` / `read_frame()` / `release()`）
- `CameraNotFoundError` を定義する
- 動作確認：`read_frame()` が取得したフレームのサイズ・FPSをターミナルに出力できること
  （OpenCVはカメラキャプチャと色変換のみに使用し、ウィンドウ表示には使わない）

### Step 04：PoseEstimator の実装
- 前提：Step 03 完了
- `pose_estimator.py` を実装する（`estimate()` / `release()`）
- `pose_constants.py` を作成する（`PoseLandmark` / `POSE_CONNECTIONS`）
- 動作確認：カメラフレームからランドマーク座標をターミナルに出力できること

### Step 05：PyQt6 メインウィンドウの骨格実装
- 前提：Step 02 完了
- `main.py` を実装する（起動・エラーハンドリング）
- `main_window.py` の骨格を実装する（ウィンドウ表示のみ）
- `gl_widget.py` の骨格を実装する（QOpenGLWidget継承・黒画面表示）
- 動作確認：PyQt6ウィンドウが起動し、黒いOpenGL領域が表示されること

### Step 06：GLWidget への描画ループ実装・スレッド化
- 前提：Step 03・Step 04・Step 05 完了
- `capture_worker.py` を実装する（QThread・カメラ取得・骨格推定・FPS計測・シグナル送信）
- `main_window.py` を更新する（CaptureWorker起動・frame_readyシグナル受信）
- `gl_widget.py` を更新する（update_frame()実装・FPS表示切り替え toggle_debug()）
- カメラ取得・骨格推定をバックグラウンドスレッドに分離してメインスレッドの描画ループをブロックしない構造にする
- Fキーで FPS・人数・解像度のデバッグ情報をオーバーレイ表示できるようにする
- 動作確認：OpenGL領域にカメラ映像が表示され、Fキーでデバッグ情報が表示されること・24fps以上を達成すること

### Step 07：モード1（オーバーレイ）の実装
- 前提：Step 06 完了
- `base_mode.py` を実装する（抽象基底クラス）
- `mode1_overlay.py` を実装する
- `gl_widget.py` に `set_mode()` を実装する
- 動作確認：カメラ映像に骨格線が重畳表示されること

### Step 08：GUI コントロールパネルの実装
- 前提：Step 07 完了
- `main_window.py` にモード切り替えボタンを追加する
- キーボード（1・2・3・Q/Esc）のイベントを実装する
- 操作ガイドQLabelをgl_widgetの上にオーバーレイ表示する
- 動作確認：ボタン・キーでモードが切り替わり、ガイドが更新されること

### Step 09：モード2（マネキン）の実装
- 前提：Step 08 完了
- `mode2_mannequin.py` を実装する
- 背景画像のOpenGLテクスチャ表示を実装する
- マネキンの2Dプリミティブ描画を実装する
- `Mode2Mannequin.set_background()` を実装する
- `open_background_dialog()` を実装する
- 動作確認：背景画像の上にマネキンが表示され、ダイアログから背景を差し替えられること

### Step 10：モード3（3Dキャラクター）の実装
- 前提：Step 09 完了
- `mode3_3d.py` を実装する
- OpenGLで3D空間（グリッド・背景）を実装する
- シンプルな人型キャラクターを骨格データに連動して描画する
- 視点の自動回転を実装する
- `Mode3D.set_character()` を実装し、`assets/characters/` からの差し替えが機能することを確認する
- 動作確認：3D空間でキャラクターが動きに連動して動作し、キャラクター差し替えができること

### Step 11：複数人対応の検証とフォールバック
- 前提：Step 07〜10（全モード）完了
- MediaPipe Pose Landmarker の `num_poses` で複数人検出を試みる
- 品質・パフォーマンスを検証する（基準：全モードで 24fps 以上・2人以上の同時検出）
- 基準を満たさない場合は `num_poses=1` にフォールバックしてスコープを確定する
- 動作確認：複数人 or 1人専用として全モードが 24fps 以上で安定動作すること

### Step 12：起動時カメラ確認・エラー処理の実装
- 前提：Step 05・Step 03 完了
- `Camera.start()` の `CameraNotFoundError` ハンドリングを実装する
- `MainWindow.show_error()` の `QMessageBox` 表示を実装する
- `closeEvent()` のリソース解放を実装する
- logging 設定を `main.py` に追加する
- 動作確認：カメラ未接続時にポップアップが表示されアプリが終了すること

### Step 13：全体結合テスト・パフォーマンス確認
- 前提：全Step完了
- 全モードを通して動作確認する
- Windows ノートPCでのフレームレートを計測する
- 合格基準：全モードで 24fps 以上を達成すること
- 基準を下回る場合は描画の軽量化を検討する
- 動作確認：全モードが 24fps 以上で安定動作すること

---

## 実装順序の根拠

```
基盤（Step 01-04）
  → 土台となるConfig・Camera・PoseEstimatorを先に固める

GUI骨格（Step 05-06）
  → OpenGLウィンドウとカメラ映像表示を確立してから描画に進む
  → Step 05はStep 02完了後、Step 06はStep 03・04・05完了後に着手する

モード実装（Step 07-10）
  → モード1→GUI→モード2→モード3の順で段階的に積み上げる
  → 各モードは前モードの描画基盤を流用できる

品質・安定化（Step 11-13）
  → 全モード実装完了後に複数人対応・エラー処理・パフォーマンスを仕上げる
  → 合格基準は 24fps 以上で統一する
```
