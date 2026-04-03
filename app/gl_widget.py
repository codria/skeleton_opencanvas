"""
gl_widget.py
QOpenGLWidget継承クラス・全モード共通の描画ループを担当する。
"""

from __future__ import annotations
import logging
import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QFont, QColor
from OpenGL.GL import (
    glClearColor, glClear, glEnable, glDisable,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glBegin, glEnd, glTexCoord2f, glVertex2f,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_PROJECTION, GL_MODELVIEW, GL_QUADS,
)

logger = logging.getLogger(__name__)


class GLWidget(QOpenGLWidget):
    def __init__(self, config, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # カメラフレームと推定結果
        self._frame: np.ndarray | None = None
        self._results: list = []
        self._texture_id: int | None = None

        # デバッグ表示
        self._show_debug = False
        self._current_fps = 0.0

        logger.info("GLWidget 初期化")

    def toggle_debug(self) -> None:
        """FPS・デバッグ情報の表示／非表示を切り替える。"""
        self._show_debug = not self._show_debug
        state = "ON" if self._show_debug else "OFF"
        logger.info(f"デバッグ表示: {state}")

    def initializeGL(self) -> None:
        """OpenGL の初期化。"""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self._texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        logger.info("OpenGL 初期化完了")

    def paintGL(self) -> None:
        """フレームをテクスチャとして描画する。"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self._frame is None:
            return

        w, h = self.width(), self.height()

        # 2D 投影設定
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(0, 0, w, h)

        # カメラ映像をテクスチャに転送（BGR→RGB変換）
        rgb_frame = self._frame[:, :, ::-1]
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            rgb_frame.shape[1], rgb_frame.shape[0],
            0, GL_RGB, GL_UNSIGNED_BYTE,
            rgb_frame.tobytes()
        )

        # フルスクリーンに描画
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(w, 0)
        glTexCoord2f(1, 1); glVertex2f(w, h)
        glTexCoord2f(0, 1); glVertex2f(0, h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # デバッグ情報をQPainterで描画（OpenGLコンテキスト外のオーバーレイ）
        if self._show_debug:
            painter = QPainter(self)
            painter.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
            painter.setPen(QColor(0, 255, 0))
            persons = len(self._results)
            debug_text = (
                f"FPS: {self._current_fps:.1f}\n"
                f"人数: {persons}\n"
                f"解像度: {w}x{h}"
            )
            x, y = 10, 20
            for line in debug_text.split("\n"):
                painter.drawText(x, y, line)
                y += 22
            painter.end()

    def resizeGL(self, w: int, h: int) -> None:
        """ウィンドウリサイズ時の処理。"""
        glViewport(0, 0, w, h)
        logger.debug(f"GLWidget リサイズ: {w}x{h}")

    def update_frame(
        self,
        frame: np.ndarray | None,
        results: list,
        fps: float = 0.0,
    ) -> None:
        """camera・pose_estimator から最新データを受け取り再描画をトリガーする。"""
        if frame is None:
            logger.warning("フレーム取得失敗。前フレームを維持します。")
            return
        self._frame = frame
        self._results = results
        self._current_fps = fps
        self.update()


