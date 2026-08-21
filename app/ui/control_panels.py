"""
control_panels.py
時系列グラフ・各種設定スライダー群の QWidget をまとめる。
MainWindow から作成・配置されるだけの依存薄い UI 部品。
"""

from __future__ import annotations
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
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


# === ControlPanel 共通スタイル（MainWindow からも import される）===
BTN_STYLE = """
    QPushButton {
        background-color: #333; color: white;
        border: 1px solid #555; border-radius: 4px;
        padding: 4px 16px; font-size: 13px;
    }
    QPushButton:hover { background-color: #555; }
    QPushButton:checked { background-color: #0078d4; border-color: #0078d4; }
"""
VCTL_STYLE = (
    "background-color: rgba(0,0,0,140); "
    "border-radius: 6px;"
)
# 画面下部の動画コントロール用（全幅 bar、両端ボーダー）
VCTL_BAR_STYLE = (
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

        # set_font_scale で軸ラベルを再設定するためにタイトルを保持
        self._left_title = title
        self._bottom_title = 'Time (s)'

        # 設定（useOpenGL / antialias）は module-level で済ませてある
        self._plot = pg.PlotWidget(background=(20, 20, 28, 200))
        self._plot.setLabel('left', self._left_title, color='w')
        self._plot.setLabel('bottom', self._bottom_title, color='w')
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
        """軸目盛・軸ラベル文字をグラフサイズスケールに合わせて拡大／縮小する。
        グラフを大きくしても文字が相対的に小さくならないように。
        """
        base_pt = 9
        pt = max(6, int(round(base_pt * scale)))
        font = QFont()
        font.setPointSize(pt)
        # 目盛文字
        for side in ('left', 'bottom'):
            self._plot.getAxis(side).setStyle(tickFont=font)
        # 軸ラベル：font-size と color は HTML 風スタイルとしてまとめて渡す
        # （color='w' を keyword で渡すと HTML スタイル展開時に上書きされて黒くなる）
        self._plot.setLabel('left', self._left_title,
                            **{'color': '#fff', 'font-size': f'{pt}pt'})
        self._plot.setLabel('bottom', self._bottom_title,
                            **{'color': '#fff', 'font-size': f'{pt}pt'})

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


class DetectAreaControlPanel(QWidget):
    """検出エリア矩形（画像正規化 x, y, w, h）を 4 本のスライダーで調整する。
    この矩形の外に鼻がある人物は判定対象から外れる（端の写り込み除外用）。
    どれか 1 本でも動かすと area_changed(x, y, w, h) を emit する。
    """

    area_changed = pyqtSignal(float, float, float, float)  # x, y, w, h（各 0〜1）

    RES = 1000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(VCTL_STYLE)
        self.setFixedHeight(128)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 6)
        root.setSpacing(3)

        header = QLabel("検出エリア（端の写り込み除外）")
        header.setStyleSheet(VCTL_LABEL_STYLE)
        root.addWidget(header)

        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        for key, name in (("x", "左端"), ("y", "上端"), ("w", "幅"), ("h", "高さ")):
            row = QHBoxLayout()
            row.setSpacing(8)
            prefix = QLabel(name)
            prefix.setStyleSheet(VCTL_LABEL_STYLE)
            prefix.setFixedWidth(40)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, self.RES)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.valueChanged.connect(self._on_any_changed)
            vlabel = QLabel("0.00")
            vlabel.setStyleSheet(VCTL_LABEL_STYLE)
            vlabel.setFixedWidth(42)
            row.addWidget(prefix)
            row.addWidget(slider, stretch=1)
            row.addWidget(vlabel)
            root.addLayout(row)
            self._sliders[key] = slider
            self._value_labels[key] = vlabel

    def _values(self) -> tuple[float, float, float, float]:
        return tuple(self._sliders[k].value() / self.RES for k in ("x", "y", "w", "h"))

    def _on_any_changed(self, _v: int) -> None:
        x, y, w, h = self._values()
        for k, val in (("x", x), ("y", y), ("w", w), ("h", h)):
            self._value_labels[k].setText(f"{val:.2f}")
        self.area_changed.emit(x, y, w, h)

    def set_values(self, x: float, y: float, w: float, h: float) -> None:
        for k, val in (("x", x), ("y", y), ("w", w), ("h", h)):
            s = self._sliders[k]
            s.blockSignals(True)
            s.setValue(max(0, min(self.RES, int(round(val * self.RES)))))
            s.blockSignals(False)
            self._value_labels[k].setText(f"{val:.2f}")


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

    def set_value(self, alpha: float) -> None:
        v = int(round(max(0.0, min(1.0, alpha)) * self.SLIDER_RES))
        self._slider.blockSignals(True)
        self._slider.setValue(v)
        self._slider.blockSignals(False)
        self._label.setText(f"{int(round(alpha * 100))}%")


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
        self.setStyleSheet(VCTL_BAR_STYLE)
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
