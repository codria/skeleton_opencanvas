"""
main_window.py
PyQt6 メインウィンドウ・モード切り替え制御・操作ガイドQLabel管理を担当する。
GLWidgetをウィンドウ全体に配置し、ボタンパネルをオーバーレイする。
"""

from __future__ import annotations
import logging
import os
import time
from collections import deque
import numpy as np
import pyqtgraph as pg

# pyqtgraph の OpenGL backend は PyQt6 で QOpenGLWidget と context 競合する可能性があり、
# 環境次第で逆に詰まることがある。一旦 OFF で software rendering に。
# （切り分け：固まらなくなれば OpenGL backend が原因、固まれば他が原因）
pg.setConfigOption('useOpenGL', False)
pg.setConfigOption('enableExperimental', False)
pg.setConfigOption('antialias', True)
from PyQt6.QtWidgets import (
    QMainWindow, QMessageBox, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSlider,
    QProgressDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from app.gl_widget import GLWidget
from app.camera import Camera, SourceOpenError
from app.pose_estimator import PoseEstimator
from app.capture_worker import CaptureWorker
from app.pose_constants import PoseLandmark
from app.t_pose import t_pose_result
from app.trails import TRAIL_COLORS
from app.video_export import VideoExporter, ExportCancelled
from app.modes.mode1_overlay import Mode1Overlay
from app.modes.mode2_mannequin import Mode2Mannequin
from app.modes.mode3_3d import Mode3D

logger = logging.getLogger(__name__)

MODES = {
    1: "モード1：オーバーレイ",
    2: "モード2：マネキン",
    3: "モード3：3Dキャラクター",
}

GUIDE_TEXT = (
    "[1/2/3]モード [B]ボーン [M]マネキン [T]Tポーズ [+/-]サイズ\n"
    "[V]動画 [C]カメラ [Space]停止 [L]ループ  "
    "[F]デバッグ [G]グラフ [H]UI [Q]終了"
)

LABEL_STYLE = (
    "color: white; background-color: rgba(0,0,0,140);"
    "padding: 6px 12px; border-radius: 6px; font-size: 12px;"
)
MODE_LABEL_STYLE = (
    "color: #00e5ff; background-color: rgba(0,0,0,140);"
    "padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: bold;"
)
BTN_STYLE = """
    QPushButton {
        background-color: #333; color: white;
        border: 1px solid #555; border-radius: 4px;
        padding: 4px 16px; font-size: 13px;
    }
    QPushButton:hover { background-color: #555; }
    QPushButton:checked { background-color: #0078d4; border-color: #0078d4; }
"""
PANEL_STYLE = "background-color: rgba(26,26,26,200); border-top: 1px solid #444;"
VCTL_STYLE = (
    "background-color: rgba(26,26,26,200); "
    "border-top: 1px solid #444; border-bottom: 1px solid #444;"
)
VCTL_LABEL_STYLE = "color: white; font-size: 12px; padding: 0 4px;"

SPEED_CYCLE = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


class TimeSeriesGraph(QWidget):
    """画面右上にオーバーレイ表示する時系列グラフ。
    現状：頭（鼻）のワールド Y 座標（メートル、腰原点）を最新 10 秒分プロット。
    検出失敗フレームは値を None で送れば NaN にして折線を切る。
    """

    MAX_DURATION_SEC = 10.0
    # バッファ最大要素数（10 秒 × 60fps の余裕を見て）
    BUFFER_SIZE = 600

    def __init__(self, title: str = "Head Y", y_range: tuple[float, float] = (0.0, 1.0),
                 invert_y: bool = True,
                 curves: dict[str, tuple[float, float, float]] | None = None,
                 parent=None) -> None:
        """
        curves: {label: (r, g, b)} の dict（RGB は 0〜1 範囲）。None なら 1 本線（シアン）。
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 設定（useOpenGL / antialias）は module-level で済ませてある
        self._plot = pg.PlotWidget(background=(20, 20, 28, 200))
        self._plot.setLabel('left', title, color='w')
        self._plot.setLabel('bottom', 'Time (s)', color='w')
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        y_margin = (y_range[1] - y_range[0]) * 0.05
        self._plot.setYRange(y_range[0] - y_margin, y_range[1] + y_margin, padding=0.0)
        self._plot.getViewBox().enableAutoRange(axis='y', enable=False)
        if invert_y:
            self._plot.getViewBox().invertY(True)
        plot_item = self._plot.getPlotItem()
        for side in ('right', 'top'):
            plot_item.showAxis(side, True)
            ax = plot_item.getAxis(side)
            ax.setStyle(showValues=False, tickLength=0)
            ax.setTextPen('w')

        left_axis = self._plot.getAxis('left')
        left_axis.setTextPen('w')
        y_lo, y_hi = y_range
        major_steps = 5
        major_ticks = [
            (y_lo + (y_hi - y_lo) * i / major_steps,
             f"{y_lo + (y_hi - y_lo) * i / major_steps:.2f}")
            for i in range(major_steps + 1)
        ]
        left_axis.setTicks([major_ticks, []])
        self._plot.getAxis('bottom').setTextPen('w')

        # 複数 curve 対応：curves 未指定なら従来通り 1 本線
        if curves is None:
            curves = {'main': (0.0, 0.9, 1.0)}
        self._curves: dict[str, object] = {}
        for label, color in curves.items():
            r, g, b = color
            pen = pg.mkPen(
                color=(int(r * 255), int(g * 255), int(b * 255)), width=2
            )
            self._curves[label] = self._plot.plot(pen=pen)
        layout.addWidget(self._plot)

        # deque(maxlen) で自動的に古い要素を捨てる（list.pop(0) の O(n) 回避）
        self._t_buf: deque = deque(maxlen=self.BUFFER_SIZE)
        self._v_bufs: dict[str, deque] = {
            label: deque(maxlen=self.BUFFER_SIZE) for label in self._curves
        }
        self._t0: float | None = None

    def reset(self) -> None:
        self._t_buf.clear()
        for buf in self._v_bufs.values():
            buf.clear()
        self._t0 = None
        for curve in self._curves.values():
            curve.setData([], [])

    def set_font_scale(self, scale: float) -> None:
        """軸目盛・ラベル文字をグラフサイズスケールに合わせて拡大／縮小する。
        グラフを大きくしても文字が相対的に小さくならないように。
        """
        base_pt = 9
        pt = max(6, int(round(base_pt * scale)))
        font = QFont()
        font.setPointSize(pt)
        for side in ('left', 'bottom'):
            self._plot.getAxis(side).setStyle(tickFont=font)

    def append(self, values, draw: bool = True,
               t_override: float | None = None) -> None:
        """新規データを追加。
        values は dict {label: value} か float 単一値か None（検出失敗）。
        None / 欠損ラベルは NaN として折線を切る。
        draw=False の時はバッファ更新だけ行い、setData/setXRange は呼ばない。
        t_override が指定されたらその値を時刻 t として使う（動画書出用、動画内時刻を渡す）。
        指定なしならライブ用に perf_counter() からの相対秒。
        deque(maxlen=BUFFER_SIZE) なので古い要素は自動で捨てられる（時間ベース cutoff 不要）。
        """
        if t_override is not None:
            if self._t0 is None:
                self._t0 = 0.0
            t = float(t_override)
        else:
            now = time.perf_counter()
            if self._t0 is None:
                self._t0 = now
            t = now - self._t0

        self._t_buf.append(t)
        if isinstance(values, dict):
            for label in self._curves:
                v = values.get(label)
                self._v_bufs[label].append(float('nan') if v is None else float(v))
        else:
            label = next(iter(self._curves))
            for k in self._v_bufs:
                if k == label:
                    self._v_bufs[k].append(
                        float('nan') if values is None else float(values)
                    )
                else:
                    self._v_bufs[k].append(float('nan'))

        if not draw:
            return

        # setData に渡す前に numpy 配列化（pyqtgraph 内部の dtype 変換コストを 1 回で済ませる）
        t_arr = np.fromiter(self._t_buf, dtype=np.float64, count=len(self._t_buf))
        self._plot.setUpdatesEnabled(False)
        try:
            for label, curve in self._curves.items():
                v_arr = np.fromiter(self._v_bufs[label], dtype=np.float64,
                                    count=len(self._v_bufs[label]))
                curve.setData(t_arr, v_arr)
            self._plot.setXRange(max(0.0, t - self.MAX_DURATION_SEC), t, padding=0)
        finally:
            self._plot.setUpdatesEnabled(True)


class Mode3ControlPanel(QWidget):
    """Mode3 アクティブ時のみ表示する回転コントロール。
    1段目: 停止/再生ボタン + 回転速度スライダー
    2段目: 視点角度スライダー（0〜360°、自動回転中も追従、操作で即時上書き）
    """

    pause_toggled = pyqtSignal()
    speed_changed = pyqtSignal(float)   # deg/sec
    angle_changed = pyqtSignal(float)   # deg (0〜360)

    SPEED_MAX = 90.0
    SLIDER_RES = 100

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(86)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(4)

        # --- 1段目: 停止/再生 + 速度 ---
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self._btn_pause = QPushButton("停止")
        self._btn_pause.setStyleSheet(BTN_STYLE)
        self._btn_pause.setFixedWidth(60)
        self._btn_pause.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_pause.clicked.connect(self.pause_toggled.emit)

        speed_prefix = QLabel("速度")
        speed_prefix.setStyleSheet(VCTL_LABEL_STYLE)
        speed_prefix.setFixedWidth(36)

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(0, self.SLIDER_RES)
        self._speed_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        self._speed_label = QLabel("30 °/s")
        self._speed_label.setStyleSheet(VCTL_LABEL_STYLE)
        self._speed_label.setFixedWidth(58)

        row1.addWidget(self._btn_pause)
        row1.addWidget(speed_prefix)
        row1.addWidget(self._speed_slider, stretch=1)
        row1.addWidget(self._speed_label)
        outer.addLayout(row1)

        # --- 2段目: 視点角度 ---
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        angle_prefix = QLabel("角度")
        angle_prefix.setStyleSheet(VCTL_LABEL_STYLE)
        # 1段目の「停止ボタン (60) + spacing (8) + 速度ラベル (36)」と
        # 揃えるため固定幅 60+8+36 = 104
        angle_prefix.setFixedWidth(104)

        self._angle_slider = QSlider(Qt.Orientation.Horizontal)
        self._angle_slider.setRange(0, 360)
        self._angle_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._angle_slider.valueChanged.connect(self._on_angle_changed)

        self._angle_label = QLabel("0°")
        self._angle_label.setStyleSheet(VCTL_LABEL_STYLE)
        self._angle_label.setFixedWidth(58)

        row2.addWidget(angle_prefix)
        row2.addWidget(self._angle_slider, stretch=1)
        row2.addWidget(self._angle_label)
        outer.addLayout(row2)

    def _on_speed_changed(self, v: int) -> None:
        speed = v / self.SLIDER_RES * self.SPEED_MAX
        self._speed_label.setText(f"{speed:.0f} °/s")
        self.speed_changed.emit(speed)

    def _on_angle_changed(self, v: int) -> None:
        self._angle_label.setText(f"{v}°")
        self.angle_changed.emit(float(v))

    def update_state(self, paused: bool, speed: float, angle: float) -> None:
        self._btn_pause.setText("再生" if paused else "停止")

        # 速度
        sv = int(max(0.0, min(speed, self.SPEED_MAX)) / self.SPEED_MAX * self.SLIDER_RES)
        if self._speed_slider.value() != sv:
            self._speed_slider.blockSignals(True)
            self._speed_slider.setValue(sv)
            self._speed_slider.blockSignals(False)
        self._speed_label.setText(f"{speed:.0f} °/s")

        # 角度
        av = int(round(angle)) % 360
        if self._angle_slider.value() != av:
            self._angle_slider.blockSignals(True)
            self._angle_slider.setValue(av)
            self._angle_slider.blockSignals(False)
        self._angle_label.setText(f"{av}°")


class PoseControlPanel(QWidget):
    """num_poses スライダー（常時表示）。MediaPipe PoseLandmarker を再作成するため
    切替時に 1〜2 秒ブロックする。"""

    num_poses_changed = pyqtSignal(int)

    NUM_MIN = 1
    NUM_MAX = 3

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        prefix = QLabel("検出人数")
        prefix.setStyleSheet(VCTL_LABEL_STYLE)
        prefix.setFixedWidth(70)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(self.NUM_MIN, self.NUM_MAX)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._label = QLabel("1")
        self._label.setStyleSheet(VCTL_LABEL_STYLE)
        self._label.setFixedWidth(30)

        layout.addWidget(prefix)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)

    def _on_slider_changed(self, v: int) -> None:
        self._label.setText(str(v))
        self.num_poses_changed.emit(int(v))

    def set_value(self, n: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(n))
        self._slider.blockSignals(False)
        self._label.setText(str(int(n)))


class SmoothingControlPanel(QWidget):
    """ランドマーク平滑化（指数移動平均）の係数 α を調整するスライダー。
    α=1.0 で平滑化なし（生値）、α が小さいほど滑らか（追従遅れ大）。
    """

    alpha_changed = pyqtSignal(float)

    SLIDER_RES = 100
    ALPHA_MIN = 0.05
    ALPHA_MAX = 1.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        prefix = QLabel("平滑化α")
        prefix.setStyleSheet(VCTL_LABEL_STYLE)
        prefix.setFixedWidth(70)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self.SLIDER_RES)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._label = QLabel("0.40")
        self._label.setStyleSheet(VCTL_LABEL_STYLE)
        self._label.setFixedWidth(50)

        layout.addWidget(prefix)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)

    def _on_slider_changed(self, v: int) -> None:
        alpha = self.ALPHA_MIN + (self.ALPHA_MAX - self.ALPHA_MIN) * v / self.SLIDER_RES
        self._label.setText(f"{alpha:.2f}")
        self.alpha_changed.emit(alpha)

    def set_value(self, alpha: float) -> None:
        v = int(round((alpha - self.ALPHA_MIN) / (self.ALPHA_MAX - self.ALPHA_MIN)
                       * self.SLIDER_RES))
        self._slider.blockSignals(True)
        self._slider.setValue(max(0, min(self.SLIDER_RES, v)))
        self._slider.blockSignals(False)
        self._label.setText(f"{alpha:.2f}")


class GraphSizeControlPanel(QWidget):
    """グラフ（X / Y 時系列）の表示サイズ係数スライダー。
    基準サイズ 360 x 180px に係数を掛けて固定サイズで描画する
    （ウィンドウサイズには追従しない）。
    """

    scale_changed = pyqtSignal(float)  # 0.3〜3.0

    SLIDER_RES = 100
    SCALE_MIN = 0.3
    SCALE_MAX = 3.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        prefix = QLabel("グラフ")
        prefix.setStyleSheet(VCTL_LABEL_STYLE)
        prefix.setFixedWidth(70)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self.SLIDER_RES)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._label = QLabel("1.00")
        self._label.setStyleSheet(VCTL_LABEL_STYLE)
        self._label.setFixedWidth(50)

        layout.addWidget(prefix)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)

    def _on_slider_changed(self, v: int) -> None:
        scale = self.SCALE_MIN + (self.SCALE_MAX - self.SCALE_MIN) * v / self.SLIDER_RES
        self._label.setText(f"{scale:.2f}")
        self.scale_changed.emit(scale)

    def value(self) -> float:
        v = self._slider.value()
        return self.SCALE_MIN + (self.SCALE_MAX - self.SCALE_MIN) * v / self.SLIDER_RES

    def set_value(self, scale: float) -> None:
        v = int(round((scale - self.SCALE_MIN) / (self.SCALE_MAX - self.SCALE_MIN)
                       * self.SLIDER_RES))
        self._slider.blockSignals(True)
        self._slider.setValue(max(0, min(self.SLIDER_RES, v)))
        self._slider.blockSignals(False)
        self._label.setText(f"{scale:.2f}")


class Mode2ControlPanel(QWidget):
    """Mode2 アクティブ時のみ表示するマネキン太さ調整スライダー。
    被写体とカメラの距離に応じてリアルタイムに調整する。
    """

    size_changed = pyqtSignal(float)  # 0.05〜1.0

    SLIDER_RES = 100
    SIZE_MIN = 0.05
    SIZE_MAX = 1.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        prefix = QLabel("太さ")
        prefix.setStyleSheet(VCTL_LABEL_STYLE)
        prefix.setFixedWidth(70)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self.SLIDER_RES)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._label = QLabel("0.28")
        self._label.setStyleSheet(VCTL_LABEL_STYLE)
        self._label.setFixedWidth(50)

        layout.addWidget(prefix)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)

    def _on_slider_changed(self, v: int) -> None:
        scale = self.SIZE_MIN + (self.SIZE_MAX - self.SIZE_MIN) * v / self.SLIDER_RES
        self._label.setText(f"{scale:.2f}")
        self.size_changed.emit(scale)

    def set_value(self, scale: float) -> None:
        """初期値を設定する（シグナル抑制）。"""
        v = int(round((scale - self.SIZE_MIN) / (self.SIZE_MAX - self.SIZE_MIN)
                       * self.SLIDER_RES))
        self._slider.blockSignals(True)
        self._slider.setValue(max(0, min(self.SLIDER_RES, v)))
        self._slider.blockSignals(False)
        self._label.setText(f"{scale:.2f}")


class TrailControlPanel(QWidget):
    """両手両足の軌跡（トレイル）描画設定。Mode2/3 アクティブ時に表示する。
    - 点の大きさ（GL_POINTS、px）
    - 線の太さ（GL_LINE_STRIP、px）
    - 長さ（保持点数）
    """

    point_size_changed = pyqtSignal(float)
    line_width_changed = pyqtSignal(float)
    max_points_changed = pyqtSignal(int)

    SLIDER_RES = 100
    POINT_MIN, POINT_MAX = 0.0, 50.0
    LINE_MIN, LINE_MAX = 0.0, 25.0
    LEN_MIN, LEN_MAX = 8, 256

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(116)  # 3 段

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(4)

        # --- 1段: 点 ---
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        p_prefix = QLabel("軌跡 点")
        p_prefix.setStyleSheet(VCTL_LABEL_STYLE)
        p_prefix.setFixedWidth(70)
        self._slider_pt = QSlider(Qt.Orientation.Horizontal)
        self._slider_pt.setRange(0, self.SLIDER_RES)
        self._slider_pt.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider_pt.valueChanged.connect(self._on_pt_changed)
        self._label_pt = QLabel("6.0px")
        self._label_pt.setStyleSheet(VCTL_LABEL_STYLE)
        self._label_pt.setFixedWidth(56)
        row1.addWidget(p_prefix)
        row1.addWidget(self._slider_pt, stretch=1)
        row1.addWidget(self._label_pt)
        outer.addLayout(row1)

        # --- 2段: 線 ---
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        l_prefix = QLabel("軌跡 線")
        l_prefix.setStyleSheet(VCTL_LABEL_STYLE)
        l_prefix.setFixedWidth(70)
        self._slider_ln = QSlider(Qt.Orientation.Horizontal)
        self._slider_ln.setRange(0, self.SLIDER_RES)
        self._slider_ln.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider_ln.valueChanged.connect(self._on_ln_changed)
        self._label_ln = QLabel("3.0px")
        self._label_ln.setStyleSheet(VCTL_LABEL_STYLE)
        self._label_ln.setFixedWidth(56)
        row2.addWidget(l_prefix)
        row2.addWidget(self._slider_ln, stretch=1)
        row2.addWidget(self._label_ln)
        outer.addLayout(row2)

        # --- 3段: 長さ ---
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        n_prefix = QLabel("軌跡 長")
        n_prefix.setStyleSheet(VCTL_LABEL_STYLE)
        n_prefix.setFixedWidth(70)
        self._slider_n = QSlider(Qt.Orientation.Horizontal)
        self._slider_n.setRange(self.LEN_MIN, self.LEN_MAX)
        self._slider_n.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider_n.valueChanged.connect(self._on_n_changed)
        self._label_n = QLabel("32")
        self._label_n.setStyleSheet(VCTL_LABEL_STYLE)
        self._label_n.setFixedWidth(56)
        row3.addWidget(n_prefix)
        row3.addWidget(self._slider_n, stretch=1)
        row3.addWidget(self._label_n)
        outer.addLayout(row3)

    @staticmethod
    def _slider_to_value(v: int, lo: float, hi: float, res: int) -> float:
        return lo + (hi - lo) * v / res

    @staticmethod
    def _value_to_slider(val: float, lo: float, hi: float, res: int) -> int:
        return int(round((val - lo) / (hi - lo) * res))

    def _on_pt_changed(self, v: int) -> None:
        val = self._slider_to_value(v, self.POINT_MIN, self.POINT_MAX, self.SLIDER_RES)
        self._label_pt.setText(f"{val:.1f}px")
        self.point_size_changed.emit(val)

    def _on_ln_changed(self, v: int) -> None:
        val = self._slider_to_value(v, self.LINE_MIN, self.LINE_MAX, self.SLIDER_RES)
        self._label_ln.setText(f"{val:.1f}px")
        self.line_width_changed.emit(val)

    def _on_n_changed(self, v: int) -> None:
        self._label_n.setText(str(int(v)))
        self.max_points_changed.emit(int(v))

    def set_values(self, point_size: float, line_width: float, max_points: int) -> None:
        sv_pt = self._value_to_slider(point_size, self.POINT_MIN, self.POINT_MAX, self.SLIDER_RES)
        sv_ln = self._value_to_slider(line_width, self.LINE_MIN, self.LINE_MAX, self.SLIDER_RES)
        for s in (self._slider_pt, self._slider_ln, self._slider_n):
            s.blockSignals(True)
        self._slider_pt.setValue(max(0, min(self.SLIDER_RES, sv_pt)))
        self._slider_ln.setValue(max(0, min(self.SLIDER_RES, sv_ln)))
        self._slider_n.setValue(max(self.LEN_MIN, min(self.LEN_MAX, int(max_points))))
        for s in (self._slider_pt, self._slider_ln, self._slider_n):
            s.blockSignals(False)
        self._label_pt.setText(f"{point_size:.1f}px")
        self._label_ln.setText(f"{line_width:.1f}px")
        self._label_n.setText(str(int(max_points)))


class OverlayControlPanel(QWidget):
    """Mode2/3 アクティブ時に表示する「実写半透明オーバーレイ」の透過度スライダー。"""

    alpha_changed = pyqtSignal(float)  # 0.0〜1.0

    SLIDER_RES = 100

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        prefix = QLabel("実写透過")
        prefix.setStyleSheet(VCTL_LABEL_STYLE)
        prefix.setFixedWidth(70)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self.SLIDER_RES)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._label = QLabel("0%")
        self._label.setStyleSheet(VCTL_LABEL_STYLE)
        self._label.setFixedWidth(40)

        layout.addWidget(prefix)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)

    def _on_slider_changed(self, v: int) -> None:
        alpha = v / self.SLIDER_RES
        self._label.setText(f"{int(round(alpha * 100))}%")
        self.alpha_changed.emit(alpha)

    def value(self) -> float:
        return self._slider.value() / self.SLIDER_RES


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


class VideoControlPanel(QWidget):
    """動画再生時のみ表示されるコントロールバー（▶⏸ / シーク / 時刻 / ループ / 速度 / 書出）。"""

    pause_toggled = pyqtSignal()
    seek_requested = pyqtSignal(int)        # frame_index
    loop_toggled = pyqtSignal()
    speed_changed = pyqtSignal(float)
    export_requested = pyqtSignal()
    export_sample_requested = pyqtSignal()

    SLIDER_MAX = 1000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(48)
        self._dragging = False
        self._speed_idx = SPEED_CYCLE.index(1.0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        self._btn_pause = QPushButton("停止")
        self._btn_pause.setStyleSheet(BTN_STYLE)
        self._btn_pause.setFixedWidth(60)
        self._btn_pause.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_pause.clicked.connect(self.pause_toggled.emit)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self.SLIDER_MAX)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setStyleSheet(VCTL_LABEL_STYLE)
        self._time_label.setFixedWidth(110)

        self._btn_loop = QPushButton("ループ ON")
        self._btn_loop.setStyleSheet(BTN_STYLE)
        self._btn_loop.setFixedWidth(100)
        self._btn_loop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_loop.clicked.connect(self.loop_toggled.emit)

        self._btn_speed = QPushButton("1.0x")
        self._btn_speed.setStyleSheet(BTN_STYLE)
        self._btn_speed.setFixedWidth(60)
        self._btn_speed.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_speed.clicked.connect(self._on_speed_click)

        self._btn_export = QPushButton("動画書出")
        self._btn_export.setStyleSheet(BTN_STYLE)
        self._btn_export.setFixedWidth(90)
        self._btn_export.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_export.clicked.connect(self.export_requested.emit)

        self._btn_export_sample = QPushButton("サンプル書出")
        self._btn_export_sample.setStyleSheet(BTN_STYLE)
        self._btn_export_sample.setFixedWidth(110)
        self._btn_export_sample.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_export_sample.clicked.connect(self.export_sample_requested.emit)

        layout.addWidget(self._btn_pause)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._time_label)
        layout.addWidget(self._btn_loop)
        layout.addWidget(self._btn_speed)
        layout.addWidget(self._btn_export)
        layout.addWidget(self._btn_export_sample)

        self._total_frames = 0
        self._fps = 30.0

    def _on_slider_pressed(self) -> None:
        self._dragging = True

    def _on_slider_released(self) -> None:
        self._dragging = False
        if self._total_frames > 0:
            ratio = self._slider.value() / self.SLIDER_MAX
            frame_index = int(ratio * (self._total_frames - 1))
            self.seek_requested.emit(frame_index)

    def _on_speed_click(self) -> None:
        self._speed_idx = (self._speed_idx + 1) % len(SPEED_CYCLE)
        new_speed = SPEED_CYCLE[self._speed_idx]
        self._btn_speed.setText(f"{new_speed:.1f}x")
        self.speed_changed.emit(new_speed)

    def update_state(self, paused: bool, loop: bool, speed: float,
                     frame_pos: int, frame_count: int, fps: float) -> None:
        """フレーム到着時に呼ぶ。スライダー位置・時刻・ボタン状態を更新する。"""
        self._total_frames = frame_count
        self._fps = max(fps, 1.0)

        self._btn_pause.setText("再生" if paused else "停止")
        self._btn_loop.setText(f"ループ {'ON' if loop else 'OFF'}")

        # 速度ボタンのインデックスを外部状態と同期
        if speed in SPEED_CYCLE:
            self._speed_idx = SPEED_CYCLE.index(speed)
            self._btn_speed.setText(f"{speed:.1f}x")

        if frame_count > 0 and not self._dragging:
            ratio = frame_pos / max(frame_count - 1, 1)
            self._slider.blockSignals(True)
            self._slider.setValue(int(ratio * self.SLIDER_MAX))
            self._slider.blockSignals(False)

        cur_sec = frame_pos / self._fps
        tot_sec = frame_count / self._fps
        self._time_label.setText(f"{_fmt_time(cur_sec)} / {_fmt_time(tot_sec)}")


class MainWindow(QMainWindow):
    def __init__(self, config, camera: Camera, estimator: PoseEstimator) -> None:
        super().__init__()
        self._config = config
        self._camera = camera
        self._estimator = estimator
        self._current_mode_id = 1
        self._ui_visible = True
        self._graph_visible = True
        self._debug_visible = False
        self._show_t_pose: bool = False
        # グラフ描画の間引き用フレームカウンタ（1 = 毎フレーム）
        self._graph_frame_counter: int = 0
        self._graph_draw_every: int = 1
        # モードインスタンスは初回切替時に生成して使い回す
        self._mode1: Mode1Overlay | None = None
        self._mode2: Mode2Mannequin | None = None
        self._mode3: Mode3D | None = None

        self.setWindowTitle("AIスケルトン体験デモ")
        w = config.get("display.width", 1280)
        h = config.get("display.height", 720)
        self.resize(w, h)

        # GLWidget をセントラルウィジェット（ウィンドウ全体）に配置
        self._gl_widget = GLWidget(config, parent=self)
        self.setCentralWidget(self._gl_widget)

        font = QFont("Meiryo", 10)
        font_bold = QFont("Meiryo", 11, QFont.Weight.Bold)

        # --- ボタンパネル（GLWidgetの子・下部オーバーレイ・常時表示）---
        self._panel = QWidget(self._gl_widget)
        self._panel.setFixedHeight(50)
        self._panel.setStyleSheet(PANEL_STYLE)
        panel_layout = QHBoxLayout(self._panel)
        panel_layout.setContentsMargins(10, 5, 10, 5)
        panel_layout.setSpacing(8)

        self._btn1 = QPushButton("1: オーバーレイ")
        self._btn2 = QPushButton("2: マネキン")
        self._btn3 = QPushButton("3: 3Dキャラ")
        self._btn_video = QPushButton("動画選択")
        self._btn_camera = QPushButton("カメラ")
        self._btn_bg = QPushButton("背景選択")

        for btn in [self._btn1, self._btn2, self._btn3]:
            btn.setCheckable(True)
            btn.setStyleSheet(BTN_STYLE)
        for btn in [self._btn_video, self._btn_camera, self._btn_bg]:
            btn.setStyleSheet(BTN_STYLE)
        # ボタンにフォーカスを残さない（Space/L キーをグローバル扱いするため）
        for btn in [self._btn1, self._btn2, self._btn3,
                    self._btn_video, self._btn_camera, self._btn_bg]:
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._btn1.clicked.connect(lambda: self._on_btn_click(1))
        self._btn2.clicked.connect(lambda: self._on_btn_click(2))
        self._btn3.clicked.connect(lambda: self._on_btn_click(3))
        self._btn_video.clicked.connect(self.open_video_dialog)
        self._btn_camera.clicked.connect(self.restore_camera)
        self._btn_bg.clicked.connect(self.open_background_dialog)

        panel_layout.addWidget(self._btn1)
        panel_layout.addWidget(self._btn2)
        panel_layout.addWidget(self._btn3)
        panel_layout.addStretch()
        panel_layout.addWidget(self._btn_video)
        panel_layout.addWidget(self._btn_camera)
        panel_layout.addWidget(self._btn_bg)

        # --- モード名（左上）---
        self._mode_label = QLabel(MODES[1], self._gl_widget)
        self._mode_label.setStyleSheet(MODE_LABEL_STYLE)
        self._mode_label.setFont(font_bold)
        self._mode_label.adjustSize()
        self._mode_label.move(16, 10)
        self._mode_label.raise_()

        # --- ガイド（右下）---
        self._guide_label = QLabel(GUIDE_TEXT, self._gl_widget)
        self._guide_label.setStyleSheet(LABEL_STYLE)
        self._guide_label.setFont(font)
        self._guide_label.adjustSize()
        self._guide_label.raise_()

        # --- デバッグ（モード名の下）---
        self._debug_label = QLabel("", self._gl_widget)
        self._debug_label.setStyleSheet(LABEL_STYLE)
        self._debug_label.setFont(font)
        self._debug_label.hide()
        self._debug_label.raise_()

        # --- 動画コントロールパネル（動画再生時のみ表示）---
        self._video_control = VideoControlPanel(self._gl_widget)
        self._video_control.hide()
        self._video_control.pause_toggled.connect(self._toggle_pause)
        self._video_control.seek_requested.connect(self._on_seek)
        self._video_control.loop_toggled.connect(self._toggle_loop)
        self._video_control.speed_changed.connect(self._on_speed_changed)
        self._video_control.export_requested.connect(self._on_export_requested)
        self._video_control.export_sample_requested.connect(self._on_export_sample_requested)
        self._video_control.raise_()

        # --- 時系列グラフ（画面右上、X→Y の 2 段）---
        # 両手両足の画像座標 X / Y を時系列でプロット
        # 色はトレイルと同じ TRAIL_COLORS を流用して見た目を統一
        graph_curves = {
            'LW': TRAIL_COLORS[PoseLandmark.LEFT_WRIST],
            'RW': TRAIL_COLORS[PoseLandmark.RIGHT_WRIST],
            'LA': TRAIL_COLORS[PoseLandmark.LEFT_ANKLE],
            'RA': TRAIL_COLORS[PoseLandmark.RIGHT_ANKLE],
        }
        # X：左→右 = 0→1（反転不要）
        self._graph_x = TimeSeriesGraph(
            title="両手両足 X",
            y_range=(0.0, 1.0),
            invert_y=False,
            curves=graph_curves,
            parent=self._gl_widget,
        )
        self._graph_x.raise_()
        # Y：上→下 = 0→1（反転して画面上端が上に見えるように）
        self._graph_y = TimeSeriesGraph(
            title="両手両足 Y",
            y_range=(0.0, 1.0),
            invert_y=True,
            curves=graph_curves,
            parent=self._gl_widget,
        )
        self._graph_y.raise_()

        # --- Mode3 回転コントロール（Mode3 アクティブ時のみ表示）---
        self._mode3_ctrl = Mode3ControlPanel(self._gl_widget)
        self._mode3_ctrl.hide()
        self._mode3_ctrl.pause_toggled.connect(self._toggle_mode3_rotation)
        self._mode3_ctrl.speed_changed.connect(self._on_mode3_speed_changed)
        self._mode3_ctrl.angle_changed.connect(self._on_mode3_angle_changed)
        self._mode3_ctrl.raise_()

        # --- Mode2 マネキンサイズコントロール（Mode2 アクティブ時のみ表示）---
        self._mode2_ctrl = Mode2ControlPanel(self._gl_widget)
        self._mode2_ctrl.hide()
        self._mode2_ctrl.set_value(0.28)  # RAW_SIZE_SCALE のデフォルト
        self._mode2_ctrl.size_changed.connect(self._on_mode2_size_changed)
        self._mode2_ctrl.raise_()

        # --- 検出人数コントロール（全モード共通、常時表示）---
        self._pose_ctrl = PoseControlPanel(self._gl_widget)
        self._pose_ctrl.set_value(self._estimator.num_poses)
        self._pose_ctrl.num_poses_changed.connect(self._on_num_poses_changed)
        self._pose_ctrl.raise_()

        # --- 平滑化αコントロール（全モード共通、常時表示）---
        self._smoothing_ctrl = SmoothingControlPanel(self._gl_widget)
        self._smoothing_ctrl.set_value(self._estimator.smoothing_alpha)
        self._smoothing_ctrl.alpha_changed.connect(self._on_smoothing_alpha_changed)
        self._smoothing_ctrl.raise_()

        # --- グラフサイズ係数コントロール（全モード共通、常時表示）---
        self._graph_size_ctrl = GraphSizeControlPanel(self._gl_widget)
        self._graph_size_ctrl.set_value(1.0)
        self._graph_scale: float = 1.0
        self._graph_size_ctrl.scale_changed.connect(self._on_graph_scale_changed)
        self._graph_size_ctrl.raise_()
        # 初期フォントスケールを反映
        self._graph_x.set_font_scale(1.0)
        self._graph_y.set_font_scale(1.0)

        # --- 軌跡コントロール（Mode2/3 アクティブ時のみ表示）---
        self._trail_ctrl = TrailControlPanel(self._gl_widget)
        self._trail_ctrl.hide()
        self._trail_ctrl.set_values(point_size=6.0, line_width=3.0, max_points=32)
        self._trail_ctrl.point_size_changed.connect(self._on_trail_point_size_changed)
        self._trail_ctrl.line_width_changed.connect(self._on_trail_line_width_changed)
        self._trail_ctrl.max_points_changed.connect(self._on_trail_max_points_changed)
        self._trail_ctrl.raise_()

        # --- 実写オーバーレイ透過コントロール（Mode2/3 アクティブ時のみ表示）---
        self._overlay_ctrl = OverlayControlPanel(self._gl_widget)
        self._overlay_ctrl.hide()
        self._overlay_ctrl.alpha_changed.connect(self._on_overlay_alpha_changed)
        self._overlay_ctrl.raise_()

        # --- ワーカー ---
        self._worker = CaptureWorker(camera, estimator, parent=self)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.start()

        # デフォルトモード
        self._mode1 = Mode1Overlay()
        self._gl_widget.set_mode(self._mode1)
        self._update_buttons()

        logger.info(f"MainWindow 初期化完了 size={w}x{h}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_overlay_positions()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_overlay_positions()

    def _update_overlay_positions(self) -> None:
        """全オーバーレイウィジェットの位置・サイズを更新する。"""
        gw = self._gl_widget.width()
        gh = self._gl_widget.height()
        panel_h = self._panel.height()
        vctl_h = self._video_control.height() if self._video_control.isVisible() else 0

        # ボタンパネル：下部全幅
        self._panel.setGeometry(0, gh - panel_h, gw, panel_h)

        # 動画コントロール：ボタンパネルの直上・全幅（動画時のみ可視）
        self._video_control.setGeometry(
            0, gh - panel_h - self._video_control.height(),
            gw, self._video_control.height()
        )

        # モード名：左上
        self._mode_label.adjustSize()
        self._mode_label.move(16, 10)

        # デバッグ：モード名の下
        self._debug_label.adjustSize()
        self._debug_label.move(16, 10 + self._mode_label.height() + 6)

        # 時系列グラフ：右上に X グラフ、その下に Y グラフ
        # 基準サイズ 360x180 に係数を掛けた固定サイズ（ウィンドウサイズ非追従）
        graph_w = max(80, int(360 * self._graph_scale))
        graph_h = max(40, int(180 * self._graph_scale))
        self._graph_x.setGeometry(gw - graph_w - 10, 10, graph_w, graph_h)
        self._graph_y.setGeometry(
            gw - graph_w - 10, 10 + graph_h + 8, graph_w, graph_h
        )
        # 後続レイアウトで「グラフの下」基準値として 2 段の合計を保持
        graph_h = graph_h * 2 + 8

        # ---- スライダー群：左側（モード名の下に縦並び）----
        left_x = 16
        left_panel_w = min(320, gw // 3)
        slider_y = 10 + self._mode_label.height() + 8
        if self._debug_visible:
            slider_y += self._debug_label.height() + 6

        # 検出人数（常時、_ui_visible 時のみ）
        self._pose_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._pose_ctrl.height()
        )
        if self._pose_ctrl.isVisible():
            slider_y += self._pose_ctrl.height() + 6

        # 平滑化α（常時、_ui_visible 時のみ）
        self._smoothing_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._smoothing_ctrl.height()
        )
        if self._smoothing_ctrl.isVisible():
            slider_y += self._smoothing_ctrl.height() + 6

        # グラフサイズ係数（常時、_ui_visible 時のみ）
        self._graph_size_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._graph_size_ctrl.height()
        )
        if self._graph_size_ctrl.isVisible():
            slider_y += self._graph_size_ctrl.height() + 6

        # 軌跡コントロール（Mode2/3 時のみ）
        self._trail_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._trail_ctrl.height()
        )
        if self._trail_ctrl.isVisible():
            slider_y += self._trail_ctrl.height() + 6

        # 実写オーバーレイ（Mode2/3 時のみ）
        self._overlay_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._overlay_ctrl.height()
        )
        if self._overlay_ctrl.isVisible():
            slider_y += self._overlay_ctrl.height() + 6

        # Mode2 / Mode3 コントロール（排他、アクティブ時のみ可視）
        self._mode2_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._mode2_ctrl.height()
        )
        self._mode3_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._mode3_ctrl.height()
        )

        # ガイド：ボタンパネル＋動画コントロールの上・右下
        self._guide_label.adjustSize()
        x = gw - self._guide_label.width() - 10
        y = gh - panel_h - vctl_h - self._guide_label.height() - 10
        self._guide_label.move(x, y)

    def _update_buttons(self) -> None:
        self._btn1.setChecked(self._current_mode_id == 1)
        self._btn2.setChecked(self._current_mode_id == 2)
        self._btn3.setChecked(self._current_mode_id == 3)

    def _on_btn_click(self, mode_id: int) -> None:
        self.switch_mode(mode_id)
        self._update_buttons()

    def switch_mode(self, mode_id: int) -> None:
        if mode_id == self._current_mode_id:
            return

        if mode_id == 1:
            if self._mode1 is None:
                self._mode1 = Mode1Overlay()
            self._gl_widget.set_mode(self._mode1)
        elif mode_id == 2:
            if self._mode2 is None:
                self._mode2 = Mode2Mannequin(self._config)
                self._mode2.set_camera_overlay_alpha(self._overlay_ctrl.value())
            self._gl_widget.set_mode(self._mode2)
        elif mode_id == 3:
            if self._mode3 is None:
                self._mode3 = Mode3D(self._config)
                self._mode3.set_camera_overlay_alpha(self._overlay_ctrl.value())
            self._gl_widget.set_mode(self._mode3)
        else:
            return

        self._current_mode_id = mode_id
        self._mode_label.setText(MODES[mode_id])
        self._update_overlay_positions()
        self._update_buttons()
        logger.info(f"モード切り替え: {mode_id}")

    def _toggle_ui(self) -> None:
        """Hキーでモード名・ガイド・デバッグ・左側スライダー群・グラフを切り替え（ボタンパネルは常時表示）。"""
        self._ui_visible = not self._ui_visible
        self._mode_label.setVisible(self._ui_visible)
        self._guide_label.setVisible(self._ui_visible)
        self._debug_label.setVisible(self._ui_visible and self._debug_visible)
        # 時系列グラフ
        self._graph_x.setVisible(self._ui_visible and self._graph_visible)
        self._graph_y.setVisible(self._ui_visible and self._graph_visible)
        # 左側スライダー群（モード依存と AND）
        self._pose_ctrl.setVisible(self._ui_visible)
        self._smoothing_ctrl.setVisible(self._ui_visible)
        self._graph_size_ctrl.setVisible(self._ui_visible)
        self._trail_ctrl.setVisible(self._ui_visible and self._current_mode_id in (2, 3))
        self._overlay_ctrl.setVisible(self._ui_visible and self._current_mode_id in (2, 3))
        self._mode2_ctrl.setVisible(
            self._ui_visible and self._current_mode_id == 2 and self._mode2 is not None
        )
        self._mode3_ctrl.setVisible(
            self._ui_visible and self._current_mode_id == 3 and self._mode3 is not None
        )
        self._update_overlay_positions()
        logger.info(f"UI表示: {'ON' if self._ui_visible else 'OFF'}")

    def _toggle_debug(self) -> None:
        """FキーでデバッグQLabelの表示切り替え。"""
        self._debug_visible = not self._debug_visible
        self._debug_label.setVisible(self._ui_visible and self._debug_visible)
        logger.info(f"デバッグ表示: {'ON' if self._debug_visible else 'OFF'}")

    def _toggle_graph(self) -> None:
        """Gキーで時系列グラフ単独の表示切替（Hキーの全体非表示とは独立）。"""
        self._graph_visible = not self._graph_visible
        self._graph_x.setVisible(self._ui_visible and self._graph_visible)
        self._graph_y.setVisible(self._ui_visible and self._graph_visible)
        logger.info(f"グラフ表示: {'ON' if self._graph_visible else 'OFF'}")

    def open_background_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "背景画像を選択", "",
            "画像ファイル (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self._config.set("mode2.background_image", path)
            if self._mode2 is not None and self._current_mode_id == 2:
                self._mode2.set_background(path)
            logger.info(f"背景画像選択: {path}")

    def open_video_dialog(self) -> None:
        """動画ファイルを選択して映像ソースを切替する。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "動画ファイルを選択", "",
            "動画ファイル (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if not path:
            return
        try:
            self._worker.switch_source(path)
            logger.info(f"動画ファイル再生: {path}")
        except SourceOpenError as e:
            logger.warning(f"動画ファイル再生失敗: {e}")
            QMessageBox.warning(
                self, "動画ファイル再生失敗",
                f"動画ファイルを開けませんでした。\n\n{e}"
            )

    def restore_camera(self) -> None:
        """映像ソースを初期カメラデバイスへ戻す。"""
        if not self._camera.is_video_file:
            logger.info("既にカメラ入力です。")
            return
        try:
            self._worker.switch_source(self._camera.device_index)
            logger.info(f"カメラ復帰: device_index={self._camera.device_index}")
        except SourceOpenError as e:
            logger.warning(f"カメラ復帰失敗: {e}")
            QMessageBox.warning(
                self, "カメラ復帰失敗",
                f"カメラを開けませんでした。\n\n{e}"
            )

    # --- 動画コントロールパネルからのシグナル受信 -----------------------------

    def _toggle_pause(self) -> None:
        if not self._camera.is_video_file:
            return
        paused = self._camera.toggle_paused()
        logger.info(f"動画 {'一時停止' if paused else '再生'}")

    def _on_seek(self, frame_index: int) -> None:
        self._worker.seek(frame_index)
        logger.info(f"シーク: frame={frame_index}")

    def _toggle_loop(self) -> None:
        if not self._camera.is_video_file:
            return
        loop = self._camera.toggle_loop()
        logger.info(f"ループ {'ON' if loop else 'OFF'}")

    def _on_speed_changed(self, speed: float) -> None:
        self._camera.set_speed(speed)
        logger.info(f"再生速度: {speed:.1f}x")

    # --- 動画書出（Mode2 をオフスクリーン描画して MP4 出力）------------------

    def _on_export_requested(self) -> None:
        """全フレームを Mode2 で MP4 に書き出す。"""
        self._do_export(max_frames=None, title="動画書出")

    def _on_export_sample_requested(self) -> None:
        """最初の 300 フレームだけ Mode2 で MP4 に書き出す（確認用、短時間）。"""
        self._do_export(max_frames=300, title="動画サンプル書出")

    def _do_export(self, max_frames: int | None, title: str) -> None:
        """書出処理本体。max_frames=None で全フレーム、整数で先頭 N フレームだけ。"""
        if not self._camera.is_video_file:
            QMessageBox.information(self, title, "動画ファイル再生中のみ利用できます。")
            return
        # Mode2 が必要（初回切替時のみ生成されるため、ここで保証）
        if self._mode2 is None:
            self._mode2 = Mode2Mannequin(self._config)
            self._mode2.set_camera_overlay_alpha(self._overlay_ctrl.value())
            # GL コンテキスト未確立だと initialize できないので gl_widget.set_mode を経由
            self._gl_widget.makeCurrent()
            self._gl_widget._ensure_initialized(self._mode2)
            self._mode2.on_mode_enter()

        # 入力動画パスを取得（Camera 内に保持されてる）
        input_path = getattr(self._camera, "_source", None)
        if not isinstance(input_path, str):
            QMessageBox.warning(self, title, "動画ソースを取得できません。")
            return

        # 出力ファイル選択
        default_name = ""
        if max_frames is not None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            default_name = f"{base}_sample.mp4"
        out_path, _ = QFileDialog.getSaveFileName(
            self, f"出力 MP4 を指定（{title}）", default_name,
            "MP4 動画 (*.mp4)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        # 再生を一時停止しておく（推定スレッドが入力動画を読まないように）
        was_paused = self._camera.paused
        self._camera.set_paused(True)

        # プログレスダイアログ
        dlg = QProgressDialog("書き出し中...", "キャンセル", 0, 100, self)
        dlg.setWindowTitle(title)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()

        def on_progress(cur: int, total: int) -> None:
            if total > 0:
                pct = int(cur * 100 / total)
                dlg.setValue(pct)
                dlg.setLabelText(f"書き出し中... {cur}/{total} フレーム")
            else:
                dlg.setLabelText(f"書き出し中... {cur} フレーム")

        def cancel_check() -> bool:
            return dlg.wasCanceled()

        # 実行
        try:
            # 書出開始時にグラフバッファをリセット（書出中の動画内時刻 0 から始める）
            self._graph_x.reset()
            self._graph_y.reset()
            exporter = VideoExporter(
                input_path=input_path,
                output_path=out_path,
                estimator=self._estimator,
                mode2=self._mode2,
                gl_widget=self._gl_widget,
                progress_cb=on_progress,
                cancel_check=cancel_check,
                max_frames=max_frames,
                graph_widgets=[self._graph_x, self._graph_y],
                graph_update_cb=self._append_graphs_for_export,
            )
            written, total = exporter.run()
            dlg.close()
            QMessageBox.information(
                self, f"{title} 完了",
                f"{written}/{total} フレームを書き出しました。\n\n{out_path}"
            )
        except ExportCancelled:
            dlg.close()
            QMessageBox.information(self, title, "キャンセルされました。")
        except Exception as e:
            dlg.close()
            logger.exception(f"{title} エラー")
            QMessageBox.critical(self, f"{title} エラー", f"{e}")
        finally:
            # 再生状態を戻す
            self._camera.set_paused(was_paused)

    # --- Mode3 回転コントロール ----------------------------------------------

    def _toggle_mode3_rotation(self) -> None:
        if self._mode3 is None:
            return
        paused = self._mode3.toggle_rotation_paused()
        self._gl_widget.update()
        logger.info(f"Mode3 回転 {'停止' if paused else '再開'}")

    def _on_mode3_speed_changed(self, speed: float) -> None:
        if self._mode3 is None:
            return
        self._mode3.set_rotation_speed(speed)
        self._gl_widget.update()

    def _on_mode3_angle_changed(self, angle: float) -> None:
        if self._mode3 is None:
            return
        self._mode3.set_view_angle(angle)
        self._gl_widget.update()

    # --- Mode2 マネキンサイズ ------------------------------------------------

    def _on_mode2_size_changed(self, scale: float) -> None:
        if self._mode2 is None:
            return
        self._mode2.renderer.set_raw_size_scale(scale)
        self._gl_widget.update()

    # --- 検出人数 -----------------------------------------------------------

    def _on_num_poses_changed(self, n: int) -> None:
        """PoseLandmarker を再作成する。1〜2 秒固まる（推定スレッドがロック待ちになる）。"""
        logger.info(f"検出人数を変更: num_poses={n}（PoseLandmarker 再作成中）")
        self._estimator.set_num_poses(n)

    def _on_smoothing_alpha_changed(self, alpha: float) -> None:
        """指数移動平均の追従係数を変更する（リアルタイム反映）。"""
        self._estimator.set_smoothing_alpha(alpha)

    def _on_graph_scale_changed(self, scale: float) -> None:
        """グラフ表示サイズの係数を変更する。文字サイズも連動。"""
        self._graph_scale = scale
        self._graph_x.set_font_scale(scale)
        self._graph_y.set_font_scale(scale)
        self._update_overlay_positions()

    def _append_graphs_for_export(self, results, t_video: float) -> None:
        """動画書出ループから呼ばれる。フレーム結果でグラフ X/Y を更新する。
        ライブと違って worker が emit しないため、書出は自前でグラフを進める。
        t_video は動画内時刻（frame_idx / fps）。
        """
        x_values: dict[str, float | None] = {
            'LW': None, 'RW': None, 'LA': None, 'RA': None
        }
        y_values: dict[str, float | None] = {
            'LW': None, 'RW': None, 'LA': None, 'RA': None
        }
        if results:
            lms = results[0].landmarks
            for label, pid in (
                ('LW', PoseLandmark.LEFT_WRIST),
                ('RW', PoseLandmark.RIGHT_WRIST),
                ('LA', PoseLandmark.LEFT_ANKLE),
                ('RA', PoseLandmark.RIGHT_ANKLE),
            ):
                if pid < len(lms):
                    lm = lms[pid]
                    if lm.visibility >= 0.3:
                        x_values[label] = lm.x
                        y_values[label] = lm.y
        self._graph_x.append(x_values, draw=True, t_override=t_video)
        self._graph_y.append(y_values, draw=True, t_override=t_video)

    # --- 軌跡（トレイル）設定 ----------------------------------------------

    def _on_trail_point_size_changed(self, size: float) -> None:
        if self._mode2 is not None:
            self._mode2.renderer.set_trail_point_size(size)
        if self._mode3 is not None:
            self._mode3.renderer.set_trail_point_size(size)
        self._gl_widget.update()

    def _on_trail_line_width_changed(self, width: float) -> None:
        if self._mode2 is not None:
            self._mode2.renderer.set_trail_line_width(width)
        if self._mode3 is not None:
            self._mode3.renderer.set_trail_line_width(width)
        self._gl_widget.update()

    def _on_trail_max_points_changed(self, n: int) -> None:
        if self._mode2 is not None:
            self._mode2.renderer.set_trail_max_points(n)
        if self._mode3 is not None:
            self._mode3.renderer.set_trail_max_points(n)
        self._gl_widget.update()

    # --- 実写オーバーレイ透過 -------------------------------------------------

    def _on_overlay_alpha_changed(self, alpha: float) -> None:
        """Mode2/3 のオーバーレイ透過度を同期して更新する。"""
        if self._mode2 is not None:
            self._mode2.set_camera_overlay_alpha(alpha)
        if self._mode3 is not None:
            self._mode3.set_camera_overlay_alpha(alpha)
        self._gl_widget.update()

    def _toggle_t_pose(self) -> None:
        """T ポーズ表示の ON/OFF（造形確認用、検出結果を上書き）。"""
        self._show_t_pose = not self._show_t_pose
        # 停止中は frame_ready が来ないので、ここで T ポーズを直接 GLWidget に流し込む
        if self._show_t_pose:
            self._gl_widget.update_frame(self._gl_widget._frame, [t_pose_result()])
        self._gl_widget.update()
        logger.info(f"T ポーズ表示: {'ON' if self._show_t_pose else 'OFF'}")

    def _toggle_mannequin_style(self) -> None:
        """Mode2/Mode3 のマネキン描画スタイルを切替（primitive ⇄ mesh）。
        両モードのスタイルを同期させる。
        """
        new_style: str | None = None
        if self._mode2 is not None:
            new_style = self._mode2.renderer.toggle_style()
        if self._mode3 is not None:
            if new_style is None:
                new_style = self._mode3.renderer.toggle_style()
            else:
                self._mode3.renderer.set_style(new_style)
        self._gl_widget.update()
        if new_style is None:
            logger.info("マネキンスタイル切替: 対象モード未初期化")
        else:
            logger.info(f"マネキンスタイル: {new_style}")

    def _adjust_mannequin_scale(self, delta: float) -> None:
        """Mode2/Mode3 のマネキン表示サイズを増減する（同期）。"""
        new_scale: float | None = None
        if self._mode2 is not None:
            new_scale = self._mode2.renderer.adjust_scale(delta)
        if self._mode3 is not None:
            if new_scale is None:
                new_scale = self._mode3.renderer.adjust_scale(delta)
            else:
                self._mode3.renderer.set_scale_factor(new_scale)
        if new_scale is not None:
            logger.info(f"マネキンサイズ: {new_scale:.2f}x")

    def _adjust_scale(self, delta: float) -> None:
        """+/-: 現在のモードに応じて作用先を切り替える。
        Mode1 → 骨格線・関節点の太さ
        Mode2 → サイズスライダー値（raw_size_scale）を直接動かして UI も同期
        Mode3 → マネキン表示サイズ（scale_factor）
        """
        if self._current_mode_id == 1:
            if self._mode1 is not None:
                new_scale = self._mode1.adjust_line_scale(delta)
                logger.info(f"骨格線サイズ: {new_scale:.2f}x")
        elif self._current_mode_id == 2:
            if self._mode2 is not None:
                r = self._mode2.renderer
                # スライダーと同じ範囲 0.05〜1.0 でクランプ
                new_val = max(0.05, min(1.0, r.raw_size_scale + delta))
                r.set_raw_size_scale(new_val)
                self._mode2_ctrl.set_value(new_val)  # スライダー UI も同期
                logger.info(f"Mode2 マネキンサイズ: {new_val:.2f}")
        else:
            # Mode3 は scale_factor を操作
            self._adjust_mannequin_scale(delta)
        self._gl_widget.update()

    def _on_frame_ready(self, frame: np.ndarray, results: list, fps: float) -> None:
        # T ポーズ ON 時は検出結果を T ポーズに差し替え（造形確認用）
        if self._show_t_pose:
            results = [t_pose_result()]
        self._gl_widget.update_frame(frame, results)
        if self._debug_visible:
            persons = len(results)
            self._debug_label.setText(
                f"FPS: {fps:.1f}  人数: {persons}  "
                f"{self._gl_widget.width()}x{self._gl_widget.height()}"
            )
            self._debug_label.adjustSize()

        # 時系列グラフに両手両足の X / Y 座標（画像座標 [0,1]）
        x_values: dict[str, float | None] = {
            'LW': None, 'RW': None, 'LA': None, 'RA': None
        }
        y_values: dict[str, float | None] = {
            'LW': None, 'RW': None, 'LA': None, 'RA': None
        }
        if results:
            lms = results[0].landmarks
            for label, pid in (
                ('LW', PoseLandmark.LEFT_WRIST),
                ('RW', PoseLandmark.RIGHT_WRIST),
                ('LA', PoseLandmark.LEFT_ANKLE),
                ('RA', PoseLandmark.RIGHT_ANKLE),
            ):
                if pid < len(lms):
                    lm = lms[pid]
                    if lm.visibility >= 0.3:
                        x_values[label] = lm.x
                        y_values[label] = lm.y
        # 3 フレームに 1 回だけグラフを再描画（バッファ更新は毎フレーム）
        self._graph_frame_counter = (self._graph_frame_counter + 1) % self._graph_draw_every
        draw_now = (self._graph_frame_counter == 0)
        self._graph_x.append(x_values, draw=draw_now)
        self._graph_y.append(y_values, draw=draw_now)

        # 動画コントロールパネルの可視性・状態更新
        is_video = self._camera.is_video_file
        if is_video != self._video_control.isVisible():
            self._video_control.setVisible(is_video)
            self._update_overlay_positions()
        if is_video:
            self._video_control.update_state(
                paused=self._camera.paused,
                loop=self._camera.loop,
                speed=self._camera.speed,
                frame_pos=self._camera.frame_pos,
                frame_count=self._camera.frame_count,
                fps=self._camera.source_fps,
            )

        # Mode3 回転コントロールの可視性・状態更新（_ui_visible と AND）
        want_mode3 = (self._ui_visible
                      and self._current_mode_id == 3 and self._mode3 is not None)
        if want_mode3 != self._mode3_ctrl.isVisible():
            self._mode3_ctrl.setVisible(want_mode3)
            self._update_overlay_positions()
        if want_mode3:
            self._mode3_ctrl.update_state(
                paused=self._mode3.rotation_paused,
                speed=self._mode3.rotation_speed,
                angle=self._mode3.renderer.rotation_y,
            )

        # 実写オーバーレイ透過コントロールの可視性（Mode2/3 アクティブ時、_ui_visible と AND）
        want_overlay = (self._ui_visible and self._current_mode_id in (2, 3))
        if want_overlay != self._overlay_ctrl.isVisible():
            self._overlay_ctrl.setVisible(want_overlay)
            self._update_overlay_positions()

        # 軌跡コントロールの可視性（Mode2/3 アクティブ時、_ui_visible と AND）
        want_trail = (self._ui_visible and self._current_mode_id in (2, 3))
        if want_trail != self._trail_ctrl.isVisible():
            self._trail_ctrl.setVisible(want_trail)
            self._update_overlay_positions()

        # Mode2 サイズコントロールの可視性（Mode2 のみ、_ui_visible と AND）
        want_mode2 = (self._ui_visible
                      and self._current_mode_id == 2 and self._mode2 is not None)
        if want_mode2 != self._mode2_ctrl.isVisible():
            self._mode2_ctrl.setVisible(want_mode2)
            self._update_overlay_positions()

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "エラー", message)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif key == Qt.Key.Key_1:
            self.switch_mode(1)
        elif key == Qt.Key.Key_2:
            self.switch_mode(2)
        elif key == Qt.Key.Key_3:
            self.switch_mode(3)
        elif key == Qt.Key.Key_F:
            self._toggle_debug()
        elif key == Qt.Key.Key_H:
            self._toggle_ui()
        elif key == Qt.Key.Key_G:
            self._toggle_graph()
        elif key == Qt.Key.Key_B:
            self._gl_widget.toggle_bones()
        elif key == Qt.Key.Key_V:
            self.open_video_dialog()
        elif key == Qt.Key.Key_C:
            self.restore_camera()
        elif key == Qt.Key.Key_Space:
            self._toggle_pause()
        elif key == Qt.Key.Key_L:
            self._toggle_loop()
        elif key == Qt.Key.Key_M:
            self._toggle_mannequin_style()
        elif key == Qt.Key.Key_T:
            self._toggle_t_pose()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._adjust_scale(+0.1)
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._adjust_scale(-0.1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        logger.info("アプリ終了処理を開始します。")
        self._worker.stop()
        self._camera.release()
        self._estimator.release()
        event.accept()
