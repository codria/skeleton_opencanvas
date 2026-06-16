# 03_interfaces_v0.5.md

## バージョン履歴

| バージョン | 変更内容 | ステータス |
|-----------|---------|-----------|
| v0.1 | 初版作成 | 承認済み |
| v0.2 | AppState.background_image_pathを削除・draw()シグネチャ確定 | 承認済み |
| v0.3 | 各モード固有メソッド追加・定義漏れ補完・定数定義追加 | 承認済み |
| v0.4 | エラー処理経路・logging方針を明記 | 承認済み |
| v0.5 | AppSettings / PoseStream / VideoExporter / TrailBuffer / TimeSeriesGraph / 各 ControlPanel / MannequinRenderer / Camera 拡張 / PoseEstimator 拡張 / CaptureWorker (frame_idx + seek_gen) / GLWidget 二段化 を反映。F キーは廃止、B/G/L/M/T/V/C/+/- 追加 | 要承認 |

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
    z: float          # 奥行き（腰原点の相対値）
    visibility: float # 可視信頼度 [0.0, 1.0]

@dataclass
class PoseLandmarkResult:
    landmarks: list[Landmark]        # 画像座標のランドマーク（x,y∈[0,1]）
    world_landmarks: list[Landmark]  # 真の 3D 座標（腰原点・メートル単位、y は下が +）
    person_index: int                # 人物インデックス（複数人対応時）
```

### 1-2. PoseFrame（PoseStream の出力）

```python
@dataclass
class PoseFrame:
    frame: np.ndarray     # BGR ndarray
    results: list         # PoseLandmarkResult のリスト
    fps: float            # 推定スレッドの実行 fps（表示用）
    frame_idx: int        # 動画ファイルなら 0-based ID、カメラなら -1
    seek_gen: int         # シーク世代（古い frame を弾くため）
    t_video: float | None # 動画内時刻（秒）。カメラ入力なら None
    is_video: bool        # 元ソースが動画ファイルか
```

### 1-3. 骨格定数（`pose_constants.py`）

```python
class PoseLandmark:
    NOSE           = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE       = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER= 4
    RIGHT_EYE      = 5
    RIGHT_EYE_OUTER= 6
    LEFT_EAR       = 7
    RIGHT_EAR      = 8
    MOUTH_LEFT     = 9
    MOUTH_RIGHT    = 10
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_PINKY     = 17
    RIGHT_PINKY    = 18
    LEFT_INDEX     = 19
    RIGHT_INDEX    = 20
    LEFT_THUMB     = 21
    RIGHT_THUMB    = 22
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28
    LEFT_HEEL      = 29
    RIGHT_HEEL     = 30
    LEFT_FOOT_INDEX  = 31
    RIGHT_FOOT_INDEX = 32

POSE_CONNECTIONS: list[tuple[int, int]] = [
    # 顔・体幹・上下肢の MediaPipe 標準接続セット（実装参照）
]
```

### 1-4. 設定ファイル（`config.yaml`）

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
  num_poses: 1                # 初期値（user_settings で上書きされる）
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  smoothing_alpha: 0.25       # EMA 平滑化係数
  model_path: "assets/models/pose_landmarker.task"

mode2:
  background_image: "assets/backgrounds/default.jpg"

mode3:
  character_model: "assets/characters/BlockMan.gltf"
  rotation_speed: 30.0
```

### 1-5. ユーザー永続化（`user_settings.json`）

スライダーで調整される値を起動時 load／終了時 save。`app/user_settings.py` の `DEFAULTS` がデフォルト。

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

## 2. エラー処理方針

| エラー種別 | 表示方法 | 処理後の動作 |
|-----------|---------|------------|
| 致命的エラー（起動失敗） | `QMessageBox.critical` ポップアップ | アプリ終了 |
| 軽微な警告（実行中） | `logging.warning` | 処理継続 |
| 動画書出のキャンセル | プログレスダイアログ内 | 動画ファイル削除しない／無音のまま残す |
| ffmpeg 不在 | `logging.info` | 音声 mux スキップ（無音 mp4） |

### エラー定義

```python
class CameraNotFoundError(Exception):
    """カメラデバイスが見つからない／開けない場合に送出。"""

class SourceOpenError(Exception):
    """動画ファイル／カメラのソース切替に失敗した場合に送出。"""

class ConfigLoadError(Exception):
    """config.yaml の読み込み・パースに失敗した場合に送出。"""

class ExportCancelled(Exception):
    """VideoExporter 実行中にユーザーがキャンセルした場合に送出（内部例外）。"""
```

---

## 3. Model 層

### 3-1. `AppSettings`

```python
class AppSettings(QObject):
    """スライダー値・UI トグル等のパラメータを一元管理。
    値変更時に対応する pyqtSignal を emit する。"""

    # 値ごとの変更通知シグナル
    num_poses_changed         = pyqtSignal(int)
    smoothing_alpha_changed   = pyqtSignal(float)
    graph_scale_changed       = pyqtSignal(float)
    graph_visible_changed     = pyqtSignal(bool)
    show_bones_changed        = pyqtSignal(bool)
    trail_point_size_changed  = pyqtSignal(float)
    trail_line_width_changed  = pyqtSignal(float)
    trail_max_points_changed  = pyqtSignal(int)
    overlay_alpha_changed     = pyqtSignal(float)
    mode2_size_scale_changed  = pyqtSignal(float)
    mode3_speed_changed       = pyqtSignal(float)
    mode3_angle_changed       = pyqtSignal(float)

    def get(self, key: str) -> Any
    def set(self, key: str, value: Any) -> None  # 同値ならスキップ
    def to_dict(self) -> dict
    def save(self) -> None  # user_settings.save 経由でファイルへ

    # 主要キーの type-safe ショートカット（getter のみ。書き換えは必ず set 経由）
    @property num_poses, smoothing_alpha, graph_scale, graph_visible,
              show_bones, trail_point_size, trail_line_width, trail_max_points,
              overlay_alpha, mode2_size_scale, mode3_speed, mode3_angle
```

### 3-2. `PoseStream` / `PoseFrame`

```python
class PoseStream(QObject):
    """worker からのフレームを集約して subscriber に配るブローカ。
    seek_gen 弾きと t_video 計算をここに集中。"""

    frame_arrived = pyqtSignal(object)  # PoseFrame

    def set_seek_gen(self, gen: int) -> None
    def set_source(self, is_video: bool, source_fps: float) -> None
    def push(self, frame: np.ndarray, results: list,
             fps: float, frame_idx: int, seek_gen: int) -> None
```

`push` は worker から呼ばれる：`worker.frame_ready.connect(pose_stream.push)`。

### 3-3. `user_settings`（モジュール関数）

```python
SETTINGS_FILE = Path("user_settings.json")
DEFAULTS: dict = { ... }   # 既定値

def load() -> dict                # 失敗時はデフォルト
def save(values: dict) -> None    # 失敗してもログ警告のみで例外を上に伝えない
```

---

## 4. Controller 層

### 4-1. `Camera`

```python
class Camera:
    def __init__(self, device_index: int, width: int, height: int, fps: int)

    def start(self) -> None                          # 起動時の疎通確認、失敗で CameraNotFoundError
    def read_frame(self) -> np.ndarray | None        # 取得失敗時は None（例外は出さない）
    def release(self) -> None

    # ソース切替（カメラ ⇄ 動画ファイル）
    def switch_source(self, source: int | str) -> None  # 失敗で SourceOpenError

    # 動画ファイル時のみ意味を持つ
    @property is_video_file: bool
    @property source_fps: float                       # 元動画 fps（または 0 for camera fallback）
    @property frame_pos: int                          # CAP_PROP_POS_FRAMES（次に読むフレーム）
    @property frame_count: int                        # 全フレーム数
    @property paused: bool
    @property loop: bool
    @property speed: float
    @property device_index: int

    def seek(self, frame_index: int) -> None
    def set_paused(self, paused: bool) -> None
    def toggle_paused(self) -> bool
    def toggle_loop(self) -> bool
    def set_speed(self, speed: float) -> None
```

### 4-2. `PoseEstimator`

```python
class PoseEstimator:
    def __init__(self, model_path: str, num_poses: int,
                 min_detection_confidence: float,
                 min_tracking_confidence: float,
                 smoothing_alpha: float = 0.4)

    def estimate(self, frame: np.ndarray) -> list[PoseLandmarkResult]
    def set_num_poses(self, num_poses: int) -> None   # MediaPipe を再生成（数秒ブロック）
    def set_smoothing_alpha(self, alpha: float) -> None
    def reset_timestamp(self) -> None                 # シーク時に呼ぶ（10s ジャンプ + EMA クリア）
    def release(self) -> None

    @property num_poses: int
    @property smoothing_alpha: float
```

### 4-3. `CaptureWorker`

```python
class CaptureWorker(QThread):
    # (frame, results, fps, frame_idx, seek_gen)
    # frame_idx: 動画なら 0-based ID、カメラは -1
    # seek_gen: シーク／ソース切替のたびに +1。古い emit はメインで弾く
    frame_ready = pyqtSignal(np.ndarray, list, float, int, int)

    def __init__(self, camera: Camera, estimator: PoseEstimator, parent=None)
    def run(self) -> None
    def stop(self) -> None

    def switch_source(self, source: int | str) -> None  # _seek_gen += 1
    def seek(self, frame_index: int) -> None             # _seek_gen += 1

    @property seek_gen: int
```

内部にカメラ取得スレッド（latest_frame・latest_frame_idx を lock 付きで保持）と推定スレッドを持つ。emit は 30fps に上限制限。

### 4-4. `MainWindow`

```python
class MainWindow(QMainWindow):
    def __init__(self, config: ConfigLoader, camera: Camera, estimator: PoseEstimator)

    # モード操作
    def switch_mode(self, mode_id: int) -> None
    def open_background_dialog(self) -> None       # Mode2 背景差し替え
    def open_video_dialog(self) -> None            # 動画ファイル選択
    def restore_camera(self) -> None               # カメラ復帰

    # 動画コントロール
    def _toggle_pause(self) -> None
    def _on_seek(self, frame_index: int) -> None
    def _toggle_loop(self) -> None
    def _on_speed_changed(self, speed: float) -> None
    def _on_export_requested(self) -> None         # 全フレーム書出
    def _on_export_sample_requested(self) -> None  # 先頭 300 フレームだけ書出
    def _do_export(self, max_frames: int | None, title: str) -> None
    def _append_graphs_for_export(self, results, t_video: float) -> None  # VideoExporter 用 hook
    @staticmethod _extract_graph_values(results) -> tuple[dict, dict]     # ライブ／書出共通

    # キー
    def _toggle_ui(self) -> None        # H: モード名・ガイド・デバッグ・スライダーを toggle
    def _toggle_graph(self) -> None     # G: グラフ単独 toggle（H と独立）
    def _toggle_t_pose(self) -> None    # T: T ポーズ固定
    def _toggle_mannequin_style(self) -> None  # M: primitive / mesh 切替
    def _adjust_scale(self, delta: float) -> None  # +/- マネキンサイズ

    # フレーム到着（PoseStream から）
    def _on_frame_arrived(self, pf: PoseFrame) -> None

    # 終了
    def closeEvent(self, event) -> None  # AppSettings.save → worker.stop → release
```

---

## 5. View 層

### 5-1. `GLWidget`

```python
class GLWidget(QOpenGLWidget):
    INTERNAL_W = 1280
    INTERNAL_H = 720

    def __init__(self, config: ConfigLoader, parent=None)

    def set_mode(self, mode: BaseMode) -> None
    def update_frame(self, frame: np.ndarray | None, results: list) -> None

    def initializeGL(self) -> None                # 内部 FBO を作成 + アクティブモード initialize
    def paintGL(self) -> None                     # 二段描画（FBO → 画面 quad）
    def resizeGL(self, w: int, h: int) -> None

    def toggle_bones(self) -> None                # B キー：ボーン表示
    def draw_bone_overlay(self, results, frame,
                          win_w: int, win_h: int) -> None  # VideoExporter からも呼ぶ

    @property show_bones: bool
```

### 5-2. `BaseMode` とサブクラス

```python
class BaseMode(ABC):
    @abstractmethod
    def initialize(self) -> None
    @abstractmethod
    def draw(self, frame: np.ndarray | None, results: list,
             width: int, height: int) -> None

    def on_resize(self, width: int, height: int) -> None
    def on_mode_enter(self) -> None
    def on_mode_exit(self) -> None
    def on_wheel(self, delta_y: int) -> None      # マウスホイール（Mode3 の視点距離など）


class Mode1Overlay(BaseMode): pass

class Mode2Mannequin(BaseMode):
    @property renderer: MannequinRenderer
    def set_background(self, image_path: str) -> None
    def set_camera_overlay_alpha(self, alpha: float) -> None  # 0.0-1.0

class Mode3D(BaseMode):
    @property renderer: MannequinRenderer
    @property rotation_paused: bool
    @property rotation_speed: float
    def set_camera_overlay_alpha(self, alpha: float) -> None
    def set_rotation_speed(self, speed: float) -> None        # deg/sec
    def set_view_angle(self, angle: float) -> None            # deg [0,360)
    def toggle_rotation_paused(self) -> bool
```

### 5-3. `MannequinRenderer`

```python
class MannequinRenderer:
    """マネキン本体描画 + トレイル描画。可視性は完全独立。"""

    def initialize(self) -> None
    def setup_lighting(self) -> None

    def draw_ortho(self, results, view_x, view_y, view_w, view_h) -> None
    def draw_perspective(self, results, view_x, view_y, view_w, view_h) -> None

    # 設定群
    def set_style(self, style: str) -> None        # "primitive" | "mesh"
    def toggle_style(self) -> str
    def set_scale_factor(self, factor: float) -> None
    def adjust_line_scale(self, delta: float) -> float
    def set_raw_size_scale(self, scale: float) -> None  # Mode2 太さ
    @property raw_size_scale: float
    @property scale_factor: float

    # 可視性（独立）
    def set_mannequin_visible(self, v: bool) -> None
    def toggle_mannequin_visible(self) -> bool
    def set_trail_visible(self, v: bool) -> None
    def toggle_trail_visible(self) -> bool
    @property mannequin_visible: bool
    @property trail_visible: bool

    # トレイル
    def set_trail_point_size(self, size: float) -> None
    def set_trail_line_width(self, width: float) -> None
    def set_trail_max_points(self, n: int) -> None
    def reset_trail(self) -> None                  # シーク時に呼ぶ

    # 視点（Mode3）
    @property rotation_y: float
    def set_view_angle(self, angle: float) -> None
    def rotate(self, delta_y: float) -> None
```

### 5-4. `TrailBuffer`

```python
class TrailBuffer:
    """両手両足の軌跡を deque で保持。認識失敗時は buffer をクリアして
    点・線とも同時に消す一元設計。"""

    def update(self, landmarks, tp) -> None        # tp は座標変換関数
    def mark_all_invisible(self) -> None           # results 空時に呼ぶ（buffer クリア）
    def is_currently_visible(self, pid: int) -> bool
    def reset(self) -> None
    def set_max_points(self, n: int) -> None
    def items(self) -> Iterator[tuple[int, tuple, list]]   # (pid, color, points)
    def is_empty(self) -> bool
```

### 5-5. `TimeSeriesGraph`（`app/ui/control_panels.py`）

```python
class TimeSeriesGraph(QWidget):
    """pyqtgraph PlotWidget をラップした時系列グラフ。"""

    MAX_DURATION_SEC = 10.0
    BUFFER_SIZE = 600

    def __init__(self, title: str = "Head Y",
                 y_range: tuple[float, float] = (0.0, 1.0),
                 invert_y: bool = True,
                 curves: dict[str, tuple[float, float, float]] | None = None,
                 parent=None)

    def append(self, values, draw: bool = True,
               t_override: float | None = None) -> None
    def reset(self) -> None
    def set_font_scale(self, scale: float) -> None
```

`values` は `{label: value | None}` の dict。`t_override` で動画書出時の動画内時刻を指定可能。

### 5-6. ControlPanel 群

すべて `QWidget` のサブクラスで `app/ui/control_panels.py` に定義。signal は値変更時に emit。

```python
class Mode3ControlPanel(QWidget):
    pause_toggled = pyqtSignal()
    speed_changed = pyqtSignal(float)   # deg/sec
    angle_changed = pyqtSignal(float)   # deg
    def update_state(self, paused, speed, angle): ...

class PoseControlPanel(QWidget):
    num_poses_changed = pyqtSignal(int)
    def set_value(self, n: int): ...

class SmoothingControlPanel(QWidget):
    alpha_changed = pyqtSignal(float)
    def set_value(self, alpha: float): ...

class GraphSizeControlPanel(QWidget):
    scale_changed = pyqtSignal(float)
    def value(self) -> float
    def set_value(self, scale: float): ...

class Mode2ControlPanel(QWidget):
    size_changed = pyqtSignal(float)
    def set_value(self, scale: float): ...

class TrailControlPanel(QWidget):
    point_size_changed = pyqtSignal(float)
    line_width_changed = pyqtSignal(float)
    max_points_changed = pyqtSignal(int)
    def set_values(self, point_size, line_width, max_points): ...

class OverlayControlPanel(QWidget):
    alpha_changed = pyqtSignal(float)
    def value(self) -> float
    def set_value(self, alpha: float): ...

class VideoControlPanel(QWidget):
    pause_toggled = pyqtSignal()
    seek_requested = pyqtSignal(int)
    loop_toggled = pyqtSignal()
    speed_changed = pyqtSignal(float)
    export_requested = pyqtSignal()
    export_sample_requested = pyqtSignal()
    def update_state(self, paused, loop, speed, frame_pos, frame_count, fps): ...
```

### 5-7. `VideoExporter`

```python
class VideoExporter:
    def __init__(self, input_path: str, output_path: str,
                 estimator: PoseEstimator,
                 mode2: Mode2Mannequin,
                 gl_widget: GLWidget,
                 progress_cb=None,
                 cancel_check=None,
                 max_frames: int | None = None,    # None で全フレーム
                 graph_widgets=None,                # 右上に焼き込む TimeSeriesGraph
                 graph_update_cb=None)              # フレームごとの hook

    def run(self) -> tuple[int, int]   # (written, total)

    # 後処理
    @staticmethod _find_ffmpeg() -> str | None
    def _mux_audio_from_source(self, video_path: str, source_path: str) -> bool
```

### 5-8. `ConfigLoader`

```python
class ConfigLoader:
    def __init__(self, config_path: str) -> None  # 失敗で ConfigLoadError
    def get(self, key: str, default=None) -> Any  # ドット区切りキー
    def set(self, key: str, value: Any) -> None   # ランタイム上書き（ファイル書き戻しなし）
```

---

## 6. キー入力マッピング

| キー | 動作 |
|------|------|
| `1` / `2` / `3` | モード切替 |
| `B` | ボーン線オーバーレイ表示／非表示 |
| `M` | マネキンスタイル切替（primitive / mesh） |
| `T` | T ポーズ固定（造形確認用） |
| `V` | 動画ファイル選択ダイアログ |
| `C` | カメラ入力に復帰 |
| `Space` | 動画再生／一時停止 |
| `L` | 動画ループ on/off |
| `G` | 時系列グラフ表示／非表示 |
| `H` | モード名・ガイド・デバッグ・スライダー群を一括 toggle（グラフはこの対象から除外） |
| `+` / `-` | マネキンサイズの増減（Mode2 は太さ、Mode3 は scale_factor） |
| `Q` / `Esc` | アプリ終了 |

`F` キー（デバッグ表示 toggle）は廃止。デバッグラベルは常時表示で、`H` で全 UI と一緒に非表示にする。

---

## 7. logging 方針

```python
# main.py で一度だけ設定
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
```

各モジュールは `logger = logging.getLogger(__name__)` を使う。
