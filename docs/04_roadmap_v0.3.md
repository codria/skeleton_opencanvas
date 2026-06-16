# 04_roadmap_v0.3.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | Step 依存関係の明記・動作確認基準の具体化・Step03 動作確認修正・Step10 に set_character 追記・Step11 に全モード完了条件追記 | 承認済み |
| v0.3 | Step 01〜13 完了マーク・Step 14〜18 追加（動画再生／トレイル+グラフ／動画書出+音声 mux／設定永続化／AppSettings+PoseStream リファクタ） | 要承認 |

---

# 実装ロードマップ：AIスケルトン体験デモ

## 基本方針

```
1 プロンプト = 1 タスク
```

各ステップは独立して動作確認できる単位に分割する。前のステップが完了・動作確認済みになってから次に進む。

---

## ステップ一覧

### Step 01：プロジェクト初期化 ✅
- ディレクトリ構成・`requirements.txt`・`config.yaml` 初期値を作成
- 動作確認：`pip install -r requirements.txt` が通る

### Step 02：ConfigLoader 実装 ✅
- `config_loader.py`（`get` / `set`）と `ConfigLoadError`
- 動作確認：`config.yaml` の値が取得できる

### Step 03：Camera 実装 ✅
- 前提：Step 02
- `camera.py`（`start` / `read_frame` / `release`）と `CameraNotFoundError`
- 動作確認：フレーム取得とサイズ・FPS のログ出力

### Step 04：PoseEstimator 実装 ✅
- 前提：Step 03
- `pose_estimator.py`（`estimate` / `release`）と `pose_constants.py`
- 動作確認：ランドマーク座標が出力される

### Step 05：PyQt6 メインウィンドウ骨格 ✅
- 前提：Step 02
- `main.py` / `main_window.py` / `gl_widget.py` 骨格
- 動作確認：ウィンドウ起動 + 黒い OpenGL 領域

### Step 06：描画ループ実装・スレッド化 ✅
- 前提：Step 03/04/05
- `capture_worker.py` で QThread 化・FPS 計測
- 動作確認：24fps 以上でカメラ映像表示、F キーでデバッグ（※後にデバッグは常時表示化）

### Step 07：モード1（オーバーレイ） ✅
- 前提：Step 06
- `mode1_overlay.py` 実装
- 動作確認：骨格線の重畳表示

### Step 08：GUI コントロールパネル ✅
- 前提：Step 07
- モード切替ボタン、キー入力、ガイド QLabel
- 動作確認：ボタン／キーでモード切替

### Step 09：モード2（マネキン） ✅
- 前提：Step 08
- 背景画像クロップ表示 + 3D マネキン正射影描画
- 動作確認：マネキンが体に追従、背景差し替えダイアログ

### Step 10：モード3（3D キャラクター） ✅
- 前提：Step 09
- 3D 空間（床・柱）+ マネキン透視投影 + 自動回転
- 動作確認：視点回転下でキャラクターが動きに連動

### Step 11：複数人対応の検証とフォールバック ✅
- 前提：Step 07〜10
- `num_poses` で複数人検出を検証、必要なら 1 人にフォールバック
- 結論：`num_poses=1` を既定とし、スライダーで動的変更可能とした

### Step 12：起動時カメラ確認・エラー処理 ✅
- 前提：Step 05/03
- `CameraNotFoundError` ハンドリング・`QMessageBox`・`closeEvent` のリソース解放
- 動作確認：カメラ未接続時のポップアップ + 安全終了

### Step 13：結合テスト・パフォーマンス確認 ✅
- 全モードで 24fps 以上を確認

---

## v0.3 で追加された Step（実装済み）

### Step 14：動画ファイル再生 ✅
- `camera.py` を「カメラ／動画ファイル統一インターフェース」に拡張（`is_video_file` / `source_fps` / `frame_pos` / `seek` / `loop` / `speed` / `switch_source`）
- 画面下部に `VideoControlPanel`（再生/シーク/ループ/速度）を追加
- `V` キー・「動画選択」ボタン、`C` キー・「カメラ」ボタンで切替
- 推定速度の都合で 30fps emit 制限を CaptureWorker に導入
- 動作確認：動画ファイルを開いて Mode2/3 で骨格推定動作、シークやループも正常

### Step 15：トレイル + 時系列グラフ ✅
- `trails.py`（TrailBuffer）と `MannequinRenderer._draw_trails`
- `app/ui/control_panels.py` に `TimeSeriesGraph`（pyqtgraph）+ 各種 ControlPanel を追加
- 認識失敗時に点・線とも一斉に消える設計（buffer リセット側に集約）
- グラフ X 軸は動画再生時 `frame_idx / source_fps`、カメラ時 `perf_counter` ベース
- 動作確認：手足の動きが軌跡と時系列で可視化される

### Step 16：動画 MP4 書出 + 音声 mux ✅
- `app/video_export.py`（`VideoExporter`）を実装
- オフスクリーン FBO で Mode2.draw → cv2.VideoWriter → ffmpeg で元音声を mux
- 「動画書出」「サンプル書出（先頭 300 フレーム）」の 2 ボタン
- ffmpeg は PATH または Shotcut/Krita/OBS 同梱版を自動検出
- グラフ widget は `QGraphicsView.render` で右上に焼き込み
- 内部的に `gl_widget.setUpdatesEnabled(False)` でライブ描画を抑制（context 干渉防止）
- 動作確認：音声付きの解析動画が生成される、ボーン/グラフが焼き込まれる

### Step 17：パラメータ永続化 ✅
- `app/user_settings.py`（JSON load/save）
- 起動時に各 ControlPanel / Renderer / Estimator に反映
- 終了時に全 UI 値を `user_settings.json` に保存
- 動作確認：スライダーを動かして終了 → 再起動で復元

### Step 18：AppSettings / PoseStream リファクタ ✅
- `app/app_settings.py`（パラメータ Model）
- `app/pose_stream.py`（フレームブローカ：seek_gen 弾き + t_video 計算）
- MainWindow を wiring 専任に分離、計算と状態保持を Model 層へ
- 内部 FBO 1280×720 二段化（GLWidget）を導入してウィンドウサイズ非依存の塗りコスト
- F キー廃止、デバッグラベル常時表示、左カラムのスタイル統一
- 動作確認：シーク前後で時系列・トレイル・推定状態が綺麗にリセットされる

---

## 今後検討中（未着手）

| 項目 | 内容 | 規模 |
|------|------|------|
| Mode3 視点パラメータ拡充 | 視点距離スライダー、上下角度、ターゲットオフセット | 小 |
| Renderer の AppSettings 直接購読 | MainWindow から `set_xxx` 呼び出しを排除 | 中 |
| 書き出し中のフリーズ対策強化 | 別スレッド／FBO 二重化／キャンセル応答性 | 中 |
| `app/ui/control_panels.py` の更なる分割 | パネル単位でファイル分離 | 中 |
| Mode3 mesh モードの精度向上 | skinning を実装に組み込む | 大 |
| MediaPipe lite モデルの選択 | 設定で heavy/full/lite を切替 | 小 |

---

## 実装順序の根拠（v0.3 追記版）

```
基盤（Step 01-04）
  → Config・Camera・PoseEstimator を先に固める

GUI 骨格（Step 05-06）
  → OpenGL ウィンドウと描画ループを確立

モード実装（Step 07-10）
  → モード1 → GUI → モード2 → モード3 の順に積み上げ

品質・安定化（Step 11-13）
  → 複数人対応・エラー処理・パフォーマンス

機能拡張フェーズ（Step 14-17）
  → 動画再生・解析可視化（トレイル/グラフ）・解析動画書出・ユーザー設定永続化
  → 展示利用に加えて「研究素材作成ツール」としての価値が出る

構造リファクタ（Step 18）
  → 機能追加が落ち着いたタイミングで Model 層（AppSettings/PoseStream）を導入
  → 内部 FBO 二段化を入れて、ウィンドウサイズと描画コストを切り離す
  → 以降の機能追加で MainWindow に if が積まれていく構造的負債を防ぐ
```
