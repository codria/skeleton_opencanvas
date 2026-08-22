"""
main_window.py
PyQt6 メインウィンドウ・モード切り替え制御・操作ガイドQLabel管理を担当する。
GLWidgetをウィンドウ全体に配置し、ボタンパネルをオーバーレイする。
"""

from __future__ import annotations
import logging
import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QMessageBox, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressDialog,
)
from PyQt6.QtCore import Qt
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
from app.modes.mode4_gesture import Mode4Gesture
from app.app_settings import AppSettings
from app.pose_stream import PoseStream, PoseFrame
from app.ui.control_panels import (
    TimeSeriesGraph,
    Mode3ControlPanel,
    PoseControlPanel,
    SmoothingControlPanel,
    GraphSizeControlPanel,
    DetectAreaControlPanel,
    Mode2ControlPanel,
    TrailControlPanel,
    OverlayControlPanel,
    VideoControlPanel,
    BTN_STYLE,
)

logger = logging.getLogger(__name__)

MODES = {
    1: "モード1：オーバーレイ",
    2: "モード2：マネキン",
    3: "モード3：3Dキャラクター",
    4: "モード4：体験",
}

GUIDE_TEXT = (
    "[1/2/3/4]モード [B]ボーン [M]マネキン切替 [S]体験切替 [T]Tポーズ [F]鏡表示 [+/-]サイズ\n"
    "[V]動画 [C]カメラ [Space]停止 [L]ループ  "
    "[G]グラフ [H]UI [Q]終了"
)

LABEL_STYLE = (
    "color: white; background-color: rgba(0,0,0,140);"
    "padding: 6px 12px; border-radius: 6px; font-size: 12px;"
)
MODE_LABEL_STYLE = (
    "color: #00e5ff; background-color: rgba(0,0,0,140);"
    "padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: bold;"
)
PANEL_STYLE = "background-color: rgba(26,26,26,200); border-top: 1px solid #444;"
# 検出エリア矩形の題名ラベル（矩形左上に重ねる）。枠色に合わせた文字色。
DETECT_MASK_LABEL_STYLE = (
    "color: #ff7043; background-color: rgba(0,0,0,150);"
    "padding: 2px 8px; border-radius: 4px; font-size: 12px;"
)
DETECT_FILTER_LABEL_STYLE = (
    "color: #34e07a; background-color: rgba(0,0,0,150);"
    "padding: 2px 8px; border-radius: 4px; font-size: 12px;"
)


class MainWindow(QMainWindow):
    def __init__(self, config, camera: Camera, estimator: PoseEstimator) -> None:
        super().__init__()
        self._config = config
        self._camera = camera
        self._estimator = estimator
        # AppSettings: スライダー値・UI トグル等のパラメータを一元管理する Model
        # （内部で user_settings.load() を呼んで前回値を引き継ぐ）
        self._app_settings = AppSettings(self)
        self._current_mode_id = 1
        self._ui_visible = True
        self._graph_visible = self._app_settings.graph_visible
        self._show_t_pose: bool = False
        # PoseStream: worker からのフレームを集約して subscriber に配るブローカ
        # （seek_gen 弾きと t_video 計算をここで集中）
        self._pose_stream = PoseStream(self)
        self._pose_stream.set_source(
            is_video=self._camera.is_video_file,
            source_fps=self._camera.source_fps,
        )
        self._pose_stream.frame_arrived.connect(self._on_frame_arrived)
        # グラフ描画の間引き用フレームカウンタ（1 = 毎フレーム）
        self._graph_frame_counter: int = 0
        self._graph_draw_every: int = 1
        # モードインスタンスは初回切替時に生成して使い回す
        self._mode1: Mode1Overlay | None = None
        self._mode2: Mode2Mannequin | None = None
        self._mode3: Mode3D | None = None
        self._mode4: Mode4Gesture | None = None

        self.setWindowTitle("AIスケルトン体験デモ  -  Created by Maeda")
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
        self._btn4 = QPushButton("4: 体験")
        self._btn_video = QPushButton("動画選択")
        self._btn_camera = QPushButton("カメラ")
        self._btn_bg = QPushButton("背景選択")

        for btn in [self._btn1, self._btn2, self._btn3, self._btn4]:
            btn.setCheckable(True)
            btn.setStyleSheet(BTN_STYLE)
        for btn in [self._btn_video, self._btn_camera, self._btn_bg]:
            btn.setStyleSheet(BTN_STYLE)
        # ボタンにフォーカスを残さない（Space/L キーをグローバル扱いするため）
        for btn in [self._btn1, self._btn2, self._btn3, self._btn4,
                    self._btn_video, self._btn_camera, self._btn_bg]:
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._btn1.clicked.connect(lambda: self._on_btn_click(1))
        self._btn2.clicked.connect(lambda: self._on_btn_click(2))
        self._btn3.clicked.connect(lambda: self._on_btn_click(3))
        self._btn4.clicked.connect(lambda: self._on_btn_click(4))
        self._btn_video.clicked.connect(self.open_video_dialog)
        self._btn_camera.clicked.connect(self.restore_camera)
        self._btn_bg.clicked.connect(self.open_background_dialog)

        panel_layout.addWidget(self._btn1)
        panel_layout.addWidget(self._btn2)
        panel_layout.addWidget(self._btn3)
        panel_layout.addWidget(self._btn4)
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

        # --- デバッグ（モード名の下、常時表示。H キーでだけ非表示）---
        self._debug_label = QLabel("", self._gl_widget)
        self._debug_label.setStyleSheet(LABEL_STYLE)
        self._debug_label.setFont(font)
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
        # config.yaml の display.graph_enabled が false なら widget を作らない。
        # 重い環境（ノート PC 等）でフリーズを避けたい時に使う。
        # 以後 self._graph_x / _graph_y は None or TimeSeriesGraph のどちらか。
        self._graph_enabled: bool = bool(config.get("display.graph_enabled", True))
        if self._graph_enabled:
            # 両手両足の画像座標 X / Y を時系列でプロット
            # 色はトレイルと同じ TRAIL_COLORS を流用して見た目を統一
            graph_curves = {
                'LW': TRAIL_COLORS[PoseLandmark.LEFT_WRIST],
                'RW': TRAIL_COLORS[PoseLandmark.RIGHT_WRIST],
                'LA': TRAIL_COLORS[PoseLandmark.LEFT_ANKLE],
                'RA': TRAIL_COLORS[PoseLandmark.RIGHT_ANKLE],
            }
            # X：左→右 = 0→1（反転不要）
            self._graph_x: TimeSeriesGraph | None = TimeSeriesGraph(
                title="両手両足 X",
                y_range=(0.0, 1.0),
                invert_y=False,
                curves=graph_curves,
                parent=self._gl_widget,
            )
            self._graph_x.raise_()
            # Y：上→下 = 0→1（反転して画面上端が上に見えるように）
            self._graph_y: TimeSeriesGraph | None = TimeSeriesGraph(
                title="両手両足 Y",
                y_range=(0.0, 1.0),
                invert_y=True,
                curves=graph_curves,
                parent=self._gl_widget,
            )
            self._graph_y.raise_()
        else:
            self._graph_x = None
            self._graph_y = None
            logger.info("時系列グラフは config.display.graph_enabled=false により無効化")

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
        self._mode2_ctrl.set_value(self._app_settings.mode2_size_scale)
        self._mode2_ctrl.size_changed.connect(self._on_mode2_size_changed)
        self._mode2_ctrl.raise_()

        # --- 検出人数コントロール（全モード共通、常時表示）---
        # user_settings の num_poses で estimator を更新
        # （同値なら set_num_poses 内で noop）
        self._estimator.set_num_poses(self._app_settings.num_poses)
        self._pose_ctrl = PoseControlPanel(self._gl_widget)
        self._pose_ctrl.set_value(self._estimator.num_poses)
        self._pose_ctrl.num_poses_changed.connect(self._on_num_poses_changed)
        self._pose_ctrl.raise_()

        # --- 平滑化αコントロール（全モード共通、常時表示）---
        self._estimator.set_smoothing_alpha(self._app_settings.smoothing_alpha)
        self._smoothing_ctrl = SmoothingControlPanel(self._gl_widget)
        self._smoothing_ctrl.set_value(self._estimator.smoothing_alpha)
        self._smoothing_ctrl.alpha_changed.connect(self._on_smoothing_alpha_changed)
        self._smoothing_ctrl.raise_()

        # --- グラフサイズ係数コントロール（全モード共通、常時表示）---
        self._graph_size_ctrl = GraphSizeControlPanel(self._gl_widget)
        self._graph_scale: float = self._app_settings.graph_scale
        self._graph_size_ctrl.set_value(self._graph_scale)
        self._graph_size_ctrl.scale_changed.connect(self._on_graph_scale_changed)
        self._graph_size_ctrl.raise_()
        # 初期フォントスケールを反映（グラフ無効時は no-op）
        if self._graph_x is not None:
            self._graph_x.set_font_scale(self._graph_scale)
            self._graph_y.set_font_scale(self._graph_scale)

        # --- 検出エリアコントロール（2 段：① 入力マスク / ② 検出後フィルタ）---
        # 永続値を estimator（除外）と GLWidget（矩形表示）に反映
        self._estimator.set_mask_area(*self._app_settings.mask_area)
        self._estimator.set_filter_area(*self._app_settings.filter_area)
        self._gl_widget.set_mask_area(*self._app_settings.mask_area)
        self._gl_widget.set_filter_area(*self._app_settings.filter_area)
        self._gl_widget.set_show_detect_area(self._ui_visible)

        # 映像入力範囲：画面端からの余白。採用範囲：入力範囲の内側からの余白
        # （必ず 入力範囲 ⊇ 採用範囲）。
        self._mask_area_ctrl = DetectAreaControlPanel(
            "映像入力範囲 余白（画面端から）", self._gl_widget)
        mx, my, mw, mh = self._app_settings.mask_area
        self._mask_area_ctrl.set_margins(mx, 1.0 - mx - mw, my, 1.0 - my - mh)
        self._mask_area_ctrl.margins_changed.connect(self._on_mask_margins_changed)
        self._mask_area_ctrl.raise_()

        self._filter_area_ctrl = DetectAreaControlPanel(
            "採用範囲 余白（入力範囲の内側から）", self._gl_widget)
        self._filter_area_ctrl.set_margins(*self._app_settings.filter_inset)
        self._filter_area_ctrl.margins_changed.connect(self._on_filter_margins_changed)
        self._filter_area_ctrl.raise_()

        # 題名ラベル：入力範囲=矩形左上、採用範囲=矩形右上
        self._mask_area_label = QLabel("映像入力範囲", self._gl_widget)
        self._mask_area_label.setStyleSheet(DETECT_MASK_LABEL_STYLE)
        self._mask_area_label.setFont(font)
        self._mask_area_label.adjustSize()
        self._mask_area_label.raise_()
        self._filter_area_label = QLabel("採用範囲（鼻基準）", self._gl_widget)
        self._filter_area_label.setStyleSheet(DETECT_FILTER_LABEL_STYLE)
        self._filter_area_label.setFont(font)
        self._filter_area_label.adjustSize()
        self._filter_area_label.raise_()

        # --- 軌跡コントロール（Mode2/3 アクティブ時のみ表示）---
        self._trail_ctrl = TrailControlPanel(self._gl_widget)
        self._trail_ctrl.hide()
        self._trail_ctrl.set_values(
            point_size=self._app_settings.trail_point_size,
            line_width=self._app_settings.trail_line_width,
            max_points=self._app_settings.trail_max_points,
        )
        self._trail_ctrl.point_size_changed.connect(self._on_trail_point_size_changed)
        self._trail_ctrl.line_width_changed.connect(self._on_trail_line_width_changed)
        self._trail_ctrl.max_points_changed.connect(self._on_trail_max_points_changed)
        self._trail_ctrl.raise_()

        # --- 実写オーバーレイ透過コントロール（Mode2/3 アクティブ時のみ表示）---
        self._overlay_ctrl = OverlayControlPanel(self._gl_widget)
        self._overlay_ctrl.hide()
        self._overlay_ctrl.set_value(self._app_settings.overlay_alpha)
        self._overlay_ctrl.alpha_changed.connect(self._on_overlay_alpha_changed)
        self._overlay_ctrl.raise_()

        # 起動時に show_bones を復元
        if self._app_settings.show_bones:
            self._gl_widget.toggle_bones()

        # --- ワーカー ---
        self._worker = CaptureWorker(camera, estimator, parent=self)
        # worker → PoseStream → MainWindow の流れ。
        # PoseStream が seek_gen 弾きと t_video 計算をやってくれる。
        self._worker.frame_ready.connect(self._pose_stream.push)
        # 起動時に mirror_display を復元し、以後は AppSettings 経由で反映
        self._worker.set_mirror(self._app_settings.mirror_display)
        self._app_settings.mirror_display_changed.connect(self._worker.set_mirror)
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

        # ---- 左列の幅は最初に決める（モード名・デバッグ・スライダー群で統一） ----
        left_x = 16
        left_panel_w = min(320, gw // 3)

        # モード名：左上（自然な幅で）
        self._mode_label.adjustSize()
        self._mode_label.move(left_x, 10)

        # デバッグ：モード名の下。左 panel と同じ幅に固定して縦並びを揃える。
        self._debug_label.setFixedWidth(left_panel_w)
        self._debug_label.adjustSize()  # setFixedWidth 後は高さだけ調整される
        self._debug_label.move(left_x, 10 + self._mode_label.height() + 6)

        # 時系列グラフ：右上に X グラフ、その下に Y グラフ
        # 基準サイズ 360x180 に係数を掛けた固定サイズ（ウィンドウサイズ非追従）
        # graph_enabled=false なら widget が None なので setGeometry は skip
        # Mode4 は Y グラフだけ（X グラフは非表示、Y を上に単独配置）
        graph_w = max(80, int(360 * self._graph_scale))
        graph_h = max(40, int(180 * self._graph_scale))
        if self._graph_x is not None:
            x_visible = self._graph_visible and self._current_mode_id != 4
            y_visible = self._graph_visible
            self._graph_x.setVisible(x_visible)
            self._graph_y.setVisible(y_visible)
            self._graph_x.setGeometry(gw - graph_w - 10, 10, graph_w, graph_h)
            # X が非表示なら Y を上（10 px）に、表示なら X の下に配置
            y_top = 10 if not x_visible else 10 + graph_h + 8
            self._graph_y.setGeometry(gw - graph_w - 10, y_top, graph_w, graph_h)
        # 後続レイアウトで「グラフの下」基準値として 2 段の合計を保持
        graph_h = graph_h * 2 + 8

        # ---- スライダー群：左側（モード名 → デバッグ → の下に縦並び）----
        # デバッグラベルは常時表示なので、その高さは常に加算する
        slider_y = 10 + self._mode_label.height() + 8
        if self._debug_label.isVisible():
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

        # 検出エリア ① マスク（常時、_ui_visible 時のみ）
        self._mask_area_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._mask_area_ctrl.height()
        )
        if self._mask_area_ctrl.isVisible():
            slider_y += self._mask_area_ctrl.height() + 6

        # 検出エリア ② 判定（常時、_ui_visible 時のみ）
        self._filter_area_ctrl.setGeometry(
            left_x, slider_y, left_panel_w, self._filter_area_ctrl.height()
        )
        if self._filter_area_ctrl.isVisible():
            slider_y += self._filter_area_ctrl.height() + 6

        # 題名ラベル：マスク=矩形左上、判定=矩形右上（映像上の位置。UI 表示時のみ）
        if self._mask_area_label.isVisible():
            mx, my, _mw, _mh = self._gl_widget.area_rect_widget_px(
                self._app_settings.mask_area)
            self._mask_area_label.adjustSize()
            self._mask_area_label.move(mx + 4, my + 4)
        if self._filter_area_label.isVisible():
            fx, fy, fw, _fh = self._gl_widget.area_rect_widget_px(
                self._app_settings.filter_area)
            self._filter_area_label.adjustSize()
            self._filter_area_label.move(
                fx + fw - self._filter_area_label.width() - 4, fy + 4)

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
        self._btn4.setChecked(self._current_mode_id == 4)

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
                self._apply_user_settings_to_renderer(self._mode2.renderer,
                                                      include_mode2=True)
            self._gl_widget.set_mode(self._mode2)
        elif mode_id == 3:
            if self._mode3 is None:
                self._mode3 = Mode3D(self._config)
                self._mode3.set_camera_overlay_alpha(self._overlay_ctrl.value())
                self._apply_user_settings_to_renderer(self._mode3.renderer,
                                                      include_mode2=False)
            self._gl_widget.set_mode(self._mode3)
        elif mode_id == 4:
            if self._mode4 is None:
                self._mode4 = Mode4Gesture(self._config)
                # 前回終了時のサブモードを AppSettings から復元（永続化）
                try:
                    self._mode4.set_sub_mode(
                        self._app_settings.get("mode4_sub_mode")
                    )
                except Exception:
                    pass
            self._gl_widget.set_mode(self._mode4)
        else:
            return

        self._current_mode_id = mode_id
        self._update_mode_label()
        self._update_overlay_positions()
        self._update_buttons()
        logger.info(f"モード切り替え: {mode_id}")

    def _update_mode_label(self) -> None:
        """モード名ラベルを現在モード + サブモード情報付きで更新する。"""
        base = MODES.get(self._current_mode_id, "")
        if self._current_mode_id == 4 and self._mode4 is not None:
            base = f"{base}（{self._mode4.sub_mode_label}）"
        self._mode_label.setText(base)

    def _toggle_ui(self) -> None:
        """Hキーでモード名・ガイド・デバッグ・左側スライダー群を切り替える。
        グラフは Hキー対象外（G キーで独立に管理）。
        """
        self._ui_visible = not self._ui_visible
        self._mode_label.setVisible(self._ui_visible)
        self._guide_label.setVisible(self._ui_visible)
        self._debug_label.setVisible(self._ui_visible)
        # 左側スライダー群（モード依存と AND）
        self._pose_ctrl.setVisible(self._ui_visible)
        self._smoothing_ctrl.setVisible(self._ui_visible)
        self._graph_size_ctrl.setVisible(self._ui_visible)
        self._mask_area_ctrl.setVisible(self._ui_visible)
        self._filter_area_ctrl.setVisible(self._ui_visible)
        self._mask_area_label.setVisible(self._ui_visible)
        self._filter_area_label.setVisible(self._ui_visible)
        self._gl_widget.set_show_detect_area(self._ui_visible)
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

    def _toggle_graph(self) -> None:
        """Gキーで時系列グラフ単独の表示切替（H キーの全体非表示とは独立）。
        graph_enabled=false ならグラフ widget が無いので no-op。
        表示/位置決定は _update_overlay_positions に集約（Mode4 は Y だけ等の
        モード別分岐を含むため）。
        """
        if self._graph_x is None:
            return
        self._graph_visible = not self._graph_visible
        self._update_overlay_positions()
        logger.info(f"グラフ表示: {'ON' if self._graph_visible else 'OFF'}")

    def _toggle_mode4_sub(self) -> None:
        """Sキーで Mode4 のサブモード循環切替（Mode4 アクティブ時のみ有効）。"""
        if self._mode4 is None:
            return
        new_sub = self._mode4.toggle_sub_mode()
        self._app_settings.set("mode4_sub_mode", new_sub)
        if self._current_mode_id == 4:
            self._update_mode_label()
        self._gl_widget.update()

    def _reset_graphs(self) -> None:
        """両グラフのバッファをリセット（シーク・ソース切替・書出開始/終了で呼ぶ）。
        graph_enabled=false なら no-op。"""
        if self._graph_x is not None:
            self._graph_x.reset()
            self._graph_y.reset()

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
            # PoseStream に新しいソース情報と最新 seek_gen を伝える
            self._pose_stream.set_source(
                is_video=self._camera.is_video_file,
                source_fps=self._camera.source_fps,
            )
            self._pose_stream.set_seek_gen(self._worker.seek_gen)
            self._reset_graphs()
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
            self._pose_stream.set_source(
                is_video=self._camera.is_video_file,
                source_fps=self._camera.source_fps,
            )
            self._pose_stream.set_seek_gen(self._worker.seek_gen)
            self._reset_graphs()
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
        # worker.seek 内で _seek_gen += 1 されているので、PoseStream に伝える。
        # 以後、これ未満の seek_gen の emit は PoseStream.push で弾かれる。
        self._pose_stream.set_seek_gen(self._worker.seek_gen)
        # シーク = 時間ジャンプなので、連続性に依存する状態を全部リセットする：
        # - グラフ: 時間軸が連続のままだと過去データがそのまま伸びて見える
        # - トレイル: 軌跡が前位置と新位置を直線で結んでしまう
        # - PoseEstimator: VIDEO モードのタイムスタンプ単調増加 & EMA 前回値
        self._reset_graphs()
        if self._mode2 is not None:
            self._mode2.renderer.reset_trail()
        if self._mode3 is not None:
            self._mode3.renderer.reset_trail()
        self._estimator.reset_timestamp()
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
            self._reset_graphs()
            exporter = VideoExporter(
                input_path=input_path,
                output_path=out_path,
                estimator=self._estimator,
                mode2=self._mode2,
                gl_widget=self._gl_widget,
                progress_cb=on_progress,
                cancel_check=cancel_check,
                max_frames=max_frames,
                graph_widgets=(
                    [self._graph_x, self._graph_y]
                    if self._graph_x is not None else None
                ),
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
            # 書出中は t_override（動画内秒数）で append したので _t0=0.0 が残る。
            # ライブ側は perf_counter ベースで動くため、リセットしないと
            # 次の append で巨大な t（perf_counter() からの絶対秒）に飛び、
            # X 範囲が遠くに飛んでグラフが固まったように見える。
            self._reset_graphs()

    # --- Mode3 回転コントロール ----------------------------------------------

    def _toggle_mode3_rotation(self) -> None:
        if self._mode3 is None:
            return
        paused = self._mode3.toggle_rotation_paused()
        self._gl_widget.update()
        logger.info(f"Mode3 回転 {'停止' if paused else '再開'}")

    def _on_mode3_speed_changed(self, speed: float) -> None:
        self._app_settings.set("mode3_speed", float(speed))
        if self._mode3 is None:
            return
        self._mode3.set_rotation_speed(speed)
        self._gl_widget.update()

    def _on_mode3_angle_changed(self, angle: float) -> None:
        self._app_settings.set("mode3_angle", float(angle))
        if self._mode3 is None:
            return
        self._mode3.set_view_angle(angle)
        self._gl_widget.update()

    # --- Mode2 マネキンサイズ ------------------------------------------------

    def _on_mode2_size_changed(self, scale: float) -> None:
        self._app_settings.set("mode2_size_scale", float(scale))
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

    @staticmethod
    def _clamp_area(x: float, y: float, w: float, h: float
                    ) -> tuple[float, float, float, float]:
        """x+w<=1, y+h<=1 に収める。フィルタ・矩形表示・永続化を一致させる。"""
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.01), 1.0 - x)
        h = min(max(h, 0.01), 1.0 - y)
        return x, y, w, h

    def _apply_filter_area(self) -> None:
        """マスク＋内側余白から算出した判定領域を estimator と GLWidget に反映する。"""
        fx, fy, fw, fh = self._app_settings.filter_area
        self._estimator.set_filter_area(fx, fy, fw, fh)
        self._gl_widget.set_filter_area(fx, fy, fw, fh)

    def _on_mask_margins_changed(self, l: float, r: float,
                                 t: float, b: float) -> None:
        """① 入力マスク領域（画面端からの余白）を変更する。外側は入力前に黒塗り。
        判定領域はマスク内側余白で定義されるので、マスク変更に追従して再反映する。"""
        x, y, w, h = self._clamp_area(l, t, 1.0 - l - r, 1.0 - t - b)
        self._app_settings.set_mask_area(x, y, w, h)
        self._estimator.set_mask_area(x, y, w, h)
        self._gl_widget.set_mask_area(x, y, w, h)
        self._apply_filter_area()   # マスク移動に判定を追従
        self._update_overlay_positions()

    def _on_filter_margins_changed(self, l: float, r: float,
                                   t: float, b: float) -> None:
        """② 判定領域（マスク内側からの余白）を変更する。鼻がこの外の人物を捨てる。"""
        self._app_settings.set_filter_inset(l, r, t, b)
        self._apply_filter_area()
        self._update_overlay_positions()

    def _on_graph_scale_changed(self, scale: float) -> None:
        """グラフ表示サイズの係数を変更する。文字サイズも連動。
        graph_enabled=false でも graph_scale 値だけは保持する（書出時の右上スペース算出に使う）。"""
        self._graph_scale = scale
        if self._graph_x is not None:
            self._graph_x.set_font_scale(scale)
            self._graph_y.set_font_scale(scale)
        self._update_overlay_positions()

    @staticmethod
    def _extract_graph_values(results) -> tuple[dict, dict]:
        """results から両手両足の (x, y) を取り出して二つの dict にする。
        ライブと書き出しで同じ抽出ロジックを使うために 1 箇所に集約。
        visibility が閾値未満や検出失敗は None（NaN）にして折線を切る。
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
        return x_values, y_values

    def _append_graphs_for_export(self, results, t_video: float) -> None:
        """動画書出ループから呼ばれる。フレーム結果でグラフ X/Y を更新する。
        ライブと違って worker emit が止まるため、書出は自前でグラフを進める。
        t_video は動画内時刻（frame_idx / fps）。
        graph_enabled=false なら no-op。
        """
        if self._graph_x is None:
            return
        x_values, y_values = self._extract_graph_values(results)
        self._graph_x.append(x_values, draw=True, t_override=t_video)
        self._graph_y.append(y_values, draw=True, t_override=t_video)

    # --- 軌跡（トレイル）設定 ----------------------------------------------

    def _on_trail_point_size_changed(self, size: float) -> None:
        self._app_settings.set("trail_point_size", float(size))
        if self._mode2 is not None:
            self._mode2.renderer.set_trail_point_size(size)
        if self._mode3 is not None:
            self._mode3.renderer.set_trail_point_size(size)
        self._gl_widget.update()

    def _on_trail_line_width_changed(self, width: float) -> None:
        self._app_settings.set("trail_line_width", float(width))
        if self._mode2 is not None:
            self._mode2.renderer.set_trail_line_width(width)
        if self._mode3 is not None:
            self._mode3.renderer.set_trail_line_width(width)
        self._gl_widget.update()

    def _on_trail_max_points_changed(self, n: int) -> None:
        self._app_settings.set("trail_max_points", int(n))
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

    def _toggle_mirror(self) -> None:
        """画面全体の鏡表示 ON/OFF。AppSettings 経由で永続化＋ワーカーに反映。"""
        new_val = not self._app_settings.mirror_display
        self._app_settings.set("mirror_display", new_val)
        logger.info(f"鏡表示: {'ON' if new_val else 'OFF'}")

    def _toggle_mannequin_style(self) -> None:
        """Mode2/Mode3 のマネキン描画スタイルを循環切替。
        primitive → mesh → hidden → primitive の順で回る。
        両モードのスタイルを同期させ、AppSettings にも書き戻して永続化する。
        Mode2/3 未初期化ならスタイル文字列だけ AppSettings で進める。
        """
        # 現在の値を AppSettings から取得して次に進める
        cycle = ("primitive", "mesh", "hidden")
        cur = self._app_settings.mannequin_style
        try:
            idx = cycle.index(cur)
        except ValueError:
            idx = -1
        new_style = cycle[(idx + 1) % len(cycle)]
        # renderer に反映
        if self._mode2 is not None:
            self._mode2.renderer.set_style(new_style)
        if self._mode3 is not None:
            self._mode3.renderer.set_style(new_style)
        # 永続化 & UI 通知
        self._app_settings.set("mannequin_style", new_style)
        self._gl_widget.update()
        logger.info(f"マネキンスタイル切替: {new_style}")

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

    def _on_frame_arrived(self, pf: PoseFrame) -> None:
        """PoseStream から PoseFrame を受け取って各表示要素を更新する。
        seek_gen 弾きと t_video 計算は PoseStream 側で済んでいるので、
        ここは「PoseFrame → 表示更新」だけに専念する。
        """
        results = pf.results
        # T ポーズ ON 時は検出結果を T ポーズに差し替え（造形確認用）
        if self._show_t_pose:
            results = [t_pose_result()]
        self._gl_widget.update_frame(pf.frame, results)
        if self._debug_label.isVisible():
            persons = len(results)
            self._debug_label.setText(
                f"FPS: {pf.fps:.1f}  人数: {persons}  "
                f"{self._gl_widget.width()}x{self._gl_widget.height()}"
            )
            self._debug_label.adjustSize()

        # 時系列グラフに両手両足の X / Y 座標（画像座標 [0,1]）
        # graph_enabled=false のときは抽出も含めて丸ごとスキップ（CPU 節約）
        if self._graph_x is not None:
            x_values, y_values = self._extract_graph_values(results)
            # 3 フレームに 1 回だけグラフを再描画（バッファ更新は毎フレーム）
            self._graph_frame_counter = (self._graph_frame_counter + 1) % self._graph_draw_every
            draw_now = (self._graph_frame_counter == 0)
            # 動画ファイル時は PoseStream が計算済みの t_video（動画内秒数）、
            # カメラ時は None → TimeSeriesGraph 側で perf_counter ベースになる。
            self._graph_x.append(x_values, draw=draw_now, t_override=pf.t_video)
            self._graph_y.append(y_values, draw=draw_now, t_override=pf.t_video)

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
        elif key == Qt.Key.Key_4:
            self.switch_mode(4)
        elif key == Qt.Key.Key_S:
            self._toggle_mode4_sub()
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
        elif key == Qt.Key.Key_F:
            self._toggle_mirror()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._adjust_scale(+0.1)
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._adjust_scale(-0.1)
        else:
            super().keyPressEvent(event)

    def _apply_user_settings_to_renderer(self, renderer, include_mode2: bool) -> None:
        """Mode2/3 が新規作成された時、AppSettings の現在値を renderer に反映する。
        スライダー操作のたびに handler 経由で _app_settings.set() しているので、
        ここからは常に最新値が読める。
        """
        renderer.set_trail_point_size(self._app_settings.trail_point_size)
        renderer.set_trail_line_width(self._app_settings.trail_line_width)
        renderer.set_trail_max_points(self._app_settings.trail_max_points)
        renderer.set_trail_visible(self._app_settings.trail_visible)
        # マネキン描画スタイル（primitive / mesh / hidden）も復元
        try:
            renderer.set_style(self._app_settings.mannequin_style)
        except ValueError:
            # 過去互換：未知の値なら primitive にフォールバック
            renderer.set_style("primitive")
        if include_mode2:
            renderer.set_raw_size_scale(self._app_settings.mode2_size_scale)

    def _sync_settings_from_runtime(self) -> None:
        """AppSettings に持っていない / handler 経由で更新されない実行時状態を、
        保存前にここで AppSettings 側へ写し戻す（save 前に呼ぶ）。
        """
        self._app_settings.set("num_poses", int(self._estimator.num_poses))
        self._app_settings.set("smoothing_alpha", float(self._estimator.smoothing_alpha))
        self._app_settings.set("graph_scale", float(self._graph_scale))
        self._app_settings.set("graph_visible", bool(self._graph_visible))
        self._app_settings.set("show_bones",
                                bool(getattr(self._gl_widget, "show_bones", False)))
        self._app_settings.set("overlay_alpha", float(self._overlay_ctrl.value()))
        # trail_visible / mannequin_style は renderer が持つので、生きている方から吸い上げる
        # （両方生きていれば mode2 優先、無ければ既存値のまま）
        active_renderer = None
        if self._mode2 is not None:
            active_renderer = self._mode2.renderer
        elif self._mode3 is not None:
            active_renderer = self._mode3.renderer
        if active_renderer is not None:
            self._app_settings.set("trail_visible",
                                    bool(active_renderer.trail_visible))
            self._app_settings.set("mannequin_style",
                                    str(active_renderer.style))
        if self._mode4 is not None:
            self._app_settings.set("mode4_sub_mode", self._mode4.sub_mode)

    def closeEvent(self, event) -> None:
        logger.info("アプリ終了処理を開始します。")
        try:
            self._sync_settings_from_runtime()
            self._app_settings.save()
        except Exception as e:
            logger.warning(f"設定保存中に例外: {e}")
        self._worker.stop()
        self._camera.release()
        self._estimator.release()
        event.accept()
