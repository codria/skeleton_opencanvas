"""
gl_widget.py
QOpenGLWidget継承クラス・全モード共通の描画ループを担当する。
デバッグ情報はMainWindow側のQLabelで管理する。
"""

from __future__ import annotations
import logging
import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLFramebufferObject
from PyQt6.QtCore import Qt
from OpenGL.GL import (
    glClearColor, glClear,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glEnable, glDisable, glColor3f, glColor4f, glLineWidth, glPointSize,
    glBegin, glEnd, glVertex2f, glTexCoord2f, glBindTexture,
    glBlendFunc, glHint,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION, GL_MODELVIEW,
    GL_LINES, GL_LINE_LOOP, GL_POINTS, GL_QUADS, GL_TEXTURE_2D,
    GL_DEPTH_TEST, GL_LIGHTING, GL_BLEND,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT,
    GL_POINT_SMOOTH, GL_POINT_SMOOTH_HINT,
    GL_NICEST,
)

logger = logging.getLogger(__name__)


class GLWidget(QOpenGLWidget):
    # 内部レンダリング解像度（ウィンドウサイズに関わらず固定）。
    # paintGL は常にこの解像度の FBO に描画し、ウィンドウサイズの矩形に拡大して表示する。
    # ウィンドウを大きくしても OpenGL 塗りつぶしコストが増えず、メインスレッドが詰まらない。
    INTERNAL_W = 1280
    INTERNAL_H = 720

    def __init__(self, config, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._frame: np.ndarray | None = None
        self._results: list = []
        self._active_mode = None
        self._gl_initialized = False
        self._show_bones = False  # Bキーで切り替え
        self._internal_fbo: QOpenGLFramebufferObject | None = None
        # 検出エリア矩形（画像正規化 x,y,w,h）と、その可視化 ON/OFF。
        # mask=入力マスク領域 / filter=検出後フィルタ領域 の 2 系統。
        # 可視化は UI 表示（H）に連動。除外自体は PoseEstimator 側で常時有効。
        self._mask_area: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self._filter_area: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self._show_detect_area = False

        logger.info("GLWidget 初期化")

    def set_mode(self, mode) -> None:
        """アクティブモードを切り替える。
        初回のみ initialize() を呼び、以降は on_mode_enter() のみ呼ぶ。
        """
        if self._active_mode is not None:
            self._active_mode.on_mode_exit()
        self._active_mode = mode
        if self._gl_initialized:
            self.makeCurrent()
            self._ensure_initialized(mode)
            mode.on_mode_enter()
            # 動画一時停止中も切替を即座に画面反映するため明示的に再描画要求
            self.update()
        # 未確立の場合は initializeGL() で初期化される
        logger.info(f"モード切り替え: {type(mode).__name__}")

    def initializeGL(self) -> None:
        """OpenGL の初期化。アクティブモードの initialize() を呼ぶ。"""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self._gl_initialized = True
        self._ensure_internal_fbo()
        if self._active_mode is not None:
            self._ensure_initialized(self._active_mode)
            self._active_mode.on_mode_enter()
            logger.info(f"initializeGL: {type(self._active_mode).__name__} 初期化")
        logger.info("OpenGL 初期化完了")

    def _ensure_internal_fbo(self) -> None:
        """内部レンダリング用 FBO を 1 度だけ作る。"""
        if self._internal_fbo is not None:
            return
        self._internal_fbo = QOpenGLFramebufferObject(
            self.INTERNAL_W, self.INTERNAL_H,
            QOpenGLFramebufferObject.Attachment.CombinedDepthStencil,
        )
        logger.info(f"内部 FBO 作成: {self.INTERNAL_W}x{self.INTERNAL_H}")

    @staticmethod
    def _ensure_initialized(mode) -> None:
        """モードの initialize() を一度だけ呼ぶ。GLTF 等の重いリソースの再読込を防ぐ。"""
        if not getattr(mode, "_initialized", False):
            mode.initialize()
            mode._initialized = True

    def toggle_bones(self) -> None:
        """Bキー：ボーン表示の切り替え（全モード共通）。停止中も即時反映。"""
        self._show_bones = not self._show_bones
        self.update()
        logger.info(f"ボーン表示: {'ON' if self._show_bones else 'OFF'}")

    @property
    def show_bones(self) -> bool:
        return self._show_bones

    def set_mask_area(self, x: float, y: float, w: float, h: float) -> None:
        """① 入力マスク領域（画像正規化）を更新して再描画する。"""
        self._mask_area = (x, y, w, h)
        self.update()

    def set_filter_area(self, x: float, y: float, w: float, h: float) -> None:
        """② 検出後フィルタ領域（画像正規化）を更新して再描画する。"""
        self._filter_area = (x, y, w, h)
        self.update()

    def set_show_detect_area(self, on: bool) -> None:
        """検出エリア矩形の可視化 ON/OFF（UI 表示に連動）。"""
        self._show_detect_area = bool(on)
        self.update()

    @staticmethod
    def _aspect_viewport(frame, win_w: int, win_h: int):
        """カメラアスペクトに合わせたレターボックス viewport (x,y,w,h) を返す。"""
        if frame is not None:
            cam_aspect = frame.shape[1] / frame.shape[0]
        else:
            cam_aspect = 16 / 9
        if win_w / win_h > cam_aspect:
            view_h = win_h
            view_w = int(win_h * cam_aspect)
            return (win_w - view_w) // 2, 0, view_w, view_h
        view_w = win_w
        view_h = int(win_w / cam_aspect)
        return 0, (win_h - view_h) // 2, view_w, view_h

    def area_rect_widget_px(self, area) -> tuple[int, int, int, int]:
        """検出エリア(画像正規化)を、このウィジェットのピクセル座標での
        矩形 (x, y, w, h) に変換する。オーバーレイ QLabel の配置に使う。
        内部 FBO(1280x720) はウィジェット全体に引き伸ばされて表示される。"""
        ax, ay, aw, ah = area
        iw, ih = self.INTERNAL_W, self.INTERNAL_H
        vx, vy, vw, vh = self._aspect_viewport(self._frame, iw, ih)
        fx = vx + ax * vw
        fy = vy + ay * vh
        sx = self.width() / iw
        sy = self.height() / ih
        return int(fx * sx), int(fy * sy), int(aw * vw * sx), int(ah * vh * sy)

    def draw_detect_area_overlay(self, area, frame, win_w: int, win_h: int,
                                 outline=(0.2, 1.0, 0.4),
                                 dim_outside: bool = True) -> None:
        """検出エリア矩形を描く。dim_outside=True なら矩形外を薄暗く、
        outline 色で枠を描く。座標はカメラ映像と同じアスペクト補正 viewport 内。"""
        ax, ay, aw, ah = area
        view_x, view_y, view_w, view_h = self._aspect_viewport(frame, win_w, win_h)

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, view_w, view_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(view_x, view_y, view_w, view_h)

        rx = ax * view_w
        ry = ay * view_h
        rw = aw * view_w
        rh = ah * view_h

        def _quad(x0, y0, x1, y1):
            glBegin(GL_QUADS)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()

        # 矩形外（除外領域）を薄暗く：左/右/上/下の 4 帯
        if dim_outside:
            glColor4f(0.0, 0.0, 0.0, 0.45)
            if rx > 0:
                _quad(0, 0, rx, view_h)                   # 左
            if rx + rw < view_w:
                _quad(rx + rw, 0, view_w, view_h)         # 右
            if ry > 0:
                _quad(rx, 0, rx + rw, ry)                 # 上
            if ry + rh < view_h:
                _quad(rx, ry + rh, rx + rw, view_h)       # 下

        # 検出エリアの枠
        glColor4f(outline[0], outline[1], outline[2], 0.9)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(rx, ry)
        glVertex2f(rx + rw, ry)
        glVertex2f(rx + rw, ry + rh)
        glVertex2f(rx, ry + rh)
        glEnd()

        glDisable(GL_BLEND)

    def paintGL(self) -> None:
        """二段階レンダリング:
          Stage 1: 内部 FBO (1280x720 固定) にモード描画
          Stage 2: FBO のテクスチャを画面サイズの矩形に拡大描画
        ウィンドウサイズが大きくなっても OpenGL 塗りつぶしコストは Stage 1 の固定値で頭打ち。
        """
        if self._active_mode is None:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            return
        if self._internal_fbo is None:
            self._ensure_internal_fbo()

        # Stage 1: 内部 FBO に固定解像度で描画
        iw, ih = self.INTERNAL_W, self.INTERNAL_H
        self._internal_fbo.bind()
        try:
            glViewport(0, 0, iw, ih)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._active_mode.draw(self._frame, self._results, iw, ih)
            if self._show_bones and self._results:
                self.draw_bone_overlay(self._results, self._frame, iw, ih)
            if self._show_detect_area:
                # ① マスク領域：外側を暗く＋赤枠。② 判定領域：緑枠のみ。
                self.draw_detect_area_overlay(
                    self._mask_area, self._frame, iw, ih,
                    outline=(1.0, 0.4, 0.2), dim_outside=True)
                self.draw_detect_area_overlay(
                    self._filter_area, self._frame, iw, ih,
                    outline=(0.2, 1.0, 0.4), dim_outside=False)
        finally:
            self._internal_fbo.release()

        # Stage 2: 内部 FBO テクスチャを画面サイズに拡大描画
        # High DPI（Windows の 125%/150% 等）対策：self.width()/height() は論理 px、
        # glViewport は物理 px を要求するので devicePixelRatioF() を掛ける。
        # これをしないと「左下に小さく」表示されてしまう。
        dpr = self.devicePixelRatioF()
        fb_w = max(1, int(self.width() * dpr))
        fb_h = max(1, int(self.height() * dpr))
        glViewport(0, 0, fb_w, fb_h)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._blit_internal_fbo_to_screen()

    def _blit_internal_fbo_to_screen(self) -> None:
        """内部 FBO のカラーテクスチャを画面全体に拡大表示する。"""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_BLEND)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # 左上 (0,0)・右下 (1,1) の正規化座標で quad を張る
        glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._internal_fbo.texture())
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        # OpenGL テクスチャ Y は下が原点、画面 Y は上が原点なので v を反転
        glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 1.0)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def draw_bone_overlay(self, results, frame, win_w: int, win_h: int) -> None:
        """全モード共通のボーン表示オーバーレイ。
        ランドマーク座標を画面座標に変換して描画する。
        VideoExporter からも呼ばれるため、必要な状態は引数で受け取る。

        Args:
            results : PoseLandmarkResult のリスト
            frame   : 入力フレーム（アスペクト比計算用、None なら 16:9 想定）
            win_w   : 描画先 viewport の幅
            win_h   : 描画先 viewport の高さ
        """
        if not results:
            return
        from app.pose_constants import POSE_CONNECTIONS

        # カメラアスペクト比に合わせたビューポートを計算
        if frame is not None:
            cam_aspect = frame.shape[1] / frame.shape[0]
        else:
            cam_aspect = 16 / 9
        win_aspect = win_w / win_h
        if win_aspect > cam_aspect:
            view_h = win_h
            view_w = int(win_h * cam_aspect)
            view_x = (win_w - view_w) // 2
            view_y = 0
        else:
            view_w = win_w
            view_h = int(win_w / cam_aspect)
            view_x = 0
            view_y = (win_h - view_h) // 2

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        # アンチエイリアス：線・点とも GL_BLEND と組合せて滑らかに描く
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, view_w, view_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(view_x, view_y, view_w, view_h)

        MIN_VIS = 0.4

        try:
            for result in results:
                lms = result.landmarks

                # ボーン線（黄色）
                glColor3f(1.0, 1.0, 0.0)
                glLineWidth(2.0)
                glBegin(GL_LINES)
                for (a, b) in POSE_CONNECTIONS:
                    if (lms[a].visibility >= MIN_VIS and
                            lms[b].visibility >= MIN_VIS):
                        glVertex2f(lms[a].x * view_w, lms[a].y * view_h)
                        glVertex2f(lms[b].x * view_w, lms[b].y * view_h)
                glEnd()

                # 関節点（シアン）
                glColor3f(0.0, 1.0, 1.0)
                glPointSize(8.0)
                glBegin(GL_POINTS)
                for lm in lms:
                    if lm.visibility >= MIN_VIS:
                        glVertex2f(lm.x * view_w, lm.y * view_h)
                glEnd()
        finally:
            glDisable(GL_LINE_SMOOTH)
            glDisable(GL_POINT_SMOOTH)
            glDisable(GL_BLEND)

    def resizeGL(self, w: int, h: int) -> None:
        """ウィンドウサイズが変わっても内部 FBO は固定。
        モードには内部解像度を通知する（実際の描画解像度なので）。"""
        if self._active_mode is not None:
            self._active_mode.on_resize(self.INTERNAL_W, self.INTERNAL_H)
        logger.debug(f"GLWidget リサイズ: {w}x{h} (内部 {self.INTERNAL_W}x{self.INTERNAL_H})")

    def update_frame(self, frame: np.ndarray | None,
                     results: list) -> None:
        """最新データを受け取り再描画をトリガーする。"""
        if frame is None:
            logger.warning("フレーム取得失敗。前フレームを維持します。")
            return
        self._frame = frame
        self._results = results
        self.update()

    def wheelEvent(self, event) -> None:
        """マウスホイールを active mode に転送する。停止中も即座に画面に反映する。"""
        if self._active_mode is not None:
            self._active_mode.on_wheel(event.angleDelta().y())
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)
