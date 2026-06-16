# 03_interfaces_v0.4.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | AppState.background_image_pathを削除・draw()シグネチャ確定 | 承認済み |
| v0.3 | 各モード固有メソッド追加・定義漏れ補完・定数定義追加 | 承認済み |
| v0.4 | エラー処理経路・logging方針を明記 | 承認済み |
| v1.0 | 人間承認・メジャーバージョン確定 | 承認済み |

---

# インターフェース定義：AIスケルトン体験デモ

## 1. データ構造

### 1-1. ランドマークデータ

```python
from dataclasses import dataclass

@dataclass
class Landmark:
    x: float          # 画面幅に対する正規化座標 [0.0, 1.0]
    y: float          # 画面高さに対する正規化座標 [0.0, 1.0]
    z: float          # 奥行き（腰を基準とした相対値）
    visibility: float # 可視信頼度 [0.0, 1.0]

@dataclass
class PoseLandmarkResult:
    landmarks: list[Landmark]  # 33点のランドマーク（MediaPipe定義順）
    person_index: int          # 人物インデックス（複数人対応時）
```

### 1-2. アプリ状態（AppState）

```python
from enum import IntEnum
from dataclasses import dataclass

class DisplayMode(IntEnum):
    OVERLAY   = 1   # モード1：オーバーレイ
    MANNEQUIN = 2   # モード2：マネキン
    THREE_D   = 3   # モード3：3Dキャラクター

@dataclass
class AppState:
    current_mode: DisplayMode   # 現在の表示モード
    camera_device_index: int    # カメラデバイス番号
```

### 1-3. 骨格定数（pose_constants.py）

全モードが共通参照する定数ファイルを別途用意する。

```python
class PoseLandmark:
    NOSE           = 0
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28
    # （他33点分は実装時にMediaPipe定義に準拠して完成させる）

POSE_CONNECTIONS: list[tuple[int, int]] = [
    # 顔（目と口のみ接続）
    (1, 2), (2, 3),     # 左目：内→中→外
    (4, 5), (5, 6),     # 右目：内→中→外
    (9, 10),            # 口：左〜右
    # 上半身
    (11, 12),           # 左肩〜右肩
    (11, 13), (13, 15), # 左腕
    (15, 17), (15, 19), (15, 21),  # 左手
    (17, 19),
    (12, 14), (14, 16), # 右腕
    (16, 18), (16, 20), (16, 22),  # 右手
    (18, 20),
    # 体幹
    (11, 23), (12, 24), (23, 24),  # 肩〜腰
    # 下半身
    (23, 25), (25, 27), # 左脚
    (27, 29), (27, 31), (29, 31),  # 左足
    (24, 26), (26, 28), # 右脚
    (28, 30), (28, 32), (30, 32),  # 右足
]
```

### 1-4. 設定ファイル（config.yaml）

```yaml
camera:
  device_index: 0
  width: 1280
  height: 720
  fps: 30

display:
  width: 1280
  height: 720

pose:
  num_poses: 3
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  model_path: "assets/models/pose_landmarker.task"

mode2:
  background_image: "assets/backgrounds/default.jpg"

mode3:
  character_model: "assets/characters/default"
  rotation_speed: 0.3
```

---

## 2. エラー処理方針

### 方針

| エラー種別 | 表示方法 | 処理後の動作 |
|-----------|---------|------------|
| 致命的エラー（起動失敗） | QMessageBox ポップアップ | アプリ終了 |
| 軽微な警告（実行中） | logging.warning（ターミナル） | 処理継続 |

### logging 設定

```python
# main.py で一度だけ設定する
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
```

### エラー処理経路

```
# 致命的エラー（起動時）
main.py
  → ConfigLoader.__init__()  ─ ConfigLoadError送出
  → Camera.start()           ─ CameraNotFoundError送出
  └─ except で捕捉
       → MainWindow.show_error(message)  # QMessageBox表示
       → アプリ終了

# 軽微な警告（実行中）
GLWidget.update_frame()
  → Camera.read_frame() が None を返す
  └─ logging.warning("フレーム取得失敗") → 処理継続（前フレームを維持）

GLWidget.update_frame()
  → PoseEstimator.estimate() が空リスト を返す
  └─ logging.warning("骨格検出なし") → 処理継続（描画はスキップ）
```

---

## 3. モジュール間インターフェース

### 3-1. ConfigLoader

```python
class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        """config.yamlを読み込む。失敗時はConfigLoadErrorを送出する。"""

    def get(self, key: str, default=None) -> Any:
        """ドット区切りキーで設定値を取得。例: get("camera.device_index")"""

    def set(self, key: str, value: Any) -> None:
        """ランタイムで設定値を上書きする。ファイルへの書き戻しは行わない。"""
```

### 3-2. Camera

```python
class Camera:
    def __init__(self, device_index: int, width: int,
                 height: int, fps: int) -> None: ...

    def start(self) -> None:
        """カメラを起動する。
        カメラが見つからない場合は CameraNotFoundError を送出する。"""

    def read_frame(self) -> np.ndarray | None:
        """最新フレームをBGR numpy配列で返す。
        取得できない場合は None を返す（例外は送出しない）。"""

    def release(self) -> None:
        """カメラリソースを解放する。"""
```

### 3-3. PoseEstimator

```python
class PoseEstimator:
    def __init__(self, model_path: str,
                 num_poses: int,
                 min_detection_confidence: float,
                 min_tracking_confidence: float) -> None: ...

    def estimate(self, frame: np.ndarray) -> list[PoseLandmarkResult]:
        """BGRフレームを受け取り、全人物のランドマークリストを返す。
        検出できない場合は空リストを返す。"""

    def release(self) -> None:
        """MediaPipeのリソースを解放する。アプリ終了時に呼ぶ。"""
```

### 3-4. BaseModeとサブクラス

```python
from abc import ABC, abstractmethod

class BaseMode(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """OpenGLリソースの初期化。GLWidget.initializeGL()から呼ばれる。
        set_mode()でモード切り替えが発生した場合も再初期化される。"""

    @abstractmethod
    def draw(self, frame: np.ndarray | None,
             results: list[PoseLandmarkResult],
             width: int, height: int) -> None:
        """paintGL()から呼ばれる描画メソッド。
        frame   : OpenCVのBGRフレーム（Noneの場合あり）
        results : PoseEstimatorの推定結果（空リストの場合あり）
        width   : 描画領域の幅（px）
        height  : 描画領域の高さ（px）"""

    def on_resize(self, width: int, height: int) -> None:
        """ウィンドウリサイズ時にGLWidget.resizeGL()から呼ばれる（任意実装）。"""

    def on_mode_enter(self) -> None:
        """このモードがアクティブになった時に呼ばれる（任意実装）。
        OpenGLコンテキスト確立後のみ呼ばれる。
        前モードのOpenGL状態汚染をリセットするために使用する。"""

    def on_mode_exit(self) -> None:
        """このモードが非アクティブになった時に呼ばれる（任意実装）。"""


class Mode2Mannequin(BaseMode):
    def set_background(self, image_path: str) -> None:
        """背景画像を差し替える。ランタイムで即時反映する。"""


class Mode3D(BaseMode):
    def set_character(self, character_path: str) -> None:
        """キャラクターモデルを差し替える。"""
```

### 3-4b. MannequinRenderer（モード2・3共用）

```python
class MannequinRenderer:
    """骨格データを受け取り3Dマネキンを描画する共用レンダラー。
    正射影（モード2）と透視投影（モード3）を切り替え可能。
    """

    def setup_lighting(self) -> None:
        """ライティングを設定する。initializeGL後に呼ぶ。"""

    def update_rotation(self, delta: float) -> None:
        """視点回転角度を更新する（モード3用）。"""

    def draw_ortho(self, results: list[PoseLandmarkResult],
                   view_x: int, view_y: int,
                   view_w: int, view_h: int) -> None:
        """正射影でマネキンを描画する（モード2用）。
        MediaPipeの正規化座標（0〜1）をそのまま使用する。"""

    def draw_perspective(self, results: list[PoseLandmarkResult],
                         view_x: int, view_y: int,
                         view_w: int, view_h: int) -> None:
        """透視投影でマネキンを描画する（モード3用）。
        _rotation_y に従って視点を自動回転する。"""
```

### 3-5. CaptureWorker

```python
class CaptureWorker(QThread):
    """カメラ取得・骨格推定をバックグラウンドで実行し、結果をシグナルで通知する。"""

    # メインスレッドへの通知シグナル（フレーム・推定結果・FPS）
    frame_ready = pyqtSignal(np.ndarray, list, float)

    def __init__(self, camera: Camera, estimator: PoseEstimator,
                 parent=None) -> None: ...

    def run(self) -> None:
        """スレッドのメインループ。
        カメラ取得 → 骨格推定 → frame_ready シグナル送信を繰り返す。
        FPSを1秒ごとに計測してシグナルに含める。"""

    def stop(self) -> None:
        """ワーカーを停止し、スレッドの終了を待機する。"""
```

### 3-6. GLWidget

```python
class GLWidget(QOpenGLWidget):
    def __init__(self, config: ConfigLoader, parent=None) -> None: ...

    def set_mode(self, mode: BaseMode) -> None:
        """アクティブモードを切り替える。
        旧モードの on_mode_exit() を呼ぶ。
        OpenGLコンテキスト確立済みの場合は新モードの initialize() と on_mode_enter() を呼ぶ。
        未確立の場合は initializeGL() で initialize/on_mode_enter を呼ぶ。"""

    def update_frame(self, frame: np.ndarray | None,
                     results: list[PoseLandmarkResult]) -> None:
        """camera・pose_estimatorから最新データを受け取り update() をトリガーする。
        frame が None の場合は logging.warning を出力して前フレームを維持する。"""

    def initializeGL(self) -> None:
        """OpenGLの初期化。アクティブモードの initialize() を呼ぶ。"""

    def paintGL(self) -> None:
        """アクティブモードの draw(frame, results, width, height) を呼ぶ。"""

    def resizeGL(self, w: int, h: int) -> None:
        """アクティブモードの on_resize(w, h) を呼ぶ。"""
```

### 3-6. MainWindow

```python
class MainWindow(QMainWindow):
    def __init__(self, config: ConfigLoader, camera: Camera,
                 estimator: PoseEstimator) -> None: ...

    def switch_mode(self, mode_id: int) -> None:
        """モードを切り替え、モード名QLabel・ボタン状態を更新する。"""

    def open_background_dialog(self) -> None:
        """背景画像ファイル選択ダイアログを開き、
        ConfigLoader.set("mode2.background_image", path) でランタイム反映する。"""

    def show_error(self, message: str) -> None:
        """QMessageBox.critical() でエラーをポップアップ表示する。
        致命的エラー（CameraNotFoundError・ConfigLoadError）の表示に使用する。"""

    def closeEvent(self, event) -> None:
        """ウィンドウ終了時に CaptureWorker.stop()・Camera.release()・
        PoseEstimator.release() を呼ぶ。"""

    # --- 内部メソッド ---
    def _toggle_ui(self) -> None:
        """Hキー：モード名・ガイド・デバッグのみ表示切り替え（パネルは常時表示）。"""

    def _toggle_debug(self) -> None:
        """Fキー：デバッグQLabelの表示切り替え（FPS・人数・解像度）。"""

    def _update_overlay_positions(self) -> None:
        """showEvent/resizeEvent時に全オーバーレイウィジェットの位置を更新する。"""
```

---

## 4. エラー定義

```python
class CameraNotFoundError(Exception):
    """起動時にカメラデバイスが見つからない場合に送出する。"""

class ConfigLoadError(Exception):
    """config.yamlの読み込み・パースに失敗した場合に送出する。"""
```

---

## 5. キー入力マッピング

| キー | 動作 |
|------|------|
| `1` | モード1（オーバーレイ）に切り替え |
| `2` | モード2（マネキン）に切り替え |
| `3` | モード3（3Dキャラクター）に切り替え |
| `F` | FPS・デバッグ情報の表示／非表示切り替え（デフォルト：非表示） |
| `H` | モード名・ガイド・デバッグ情報の表示／非表示切り替え（ボタンパネルは常時表示） |
| `Q` / `Esc` | アプリ終了（closeEvent経由） |
