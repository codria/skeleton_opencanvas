"""
mode1_overlay.py
モード1：カメラ映像をOpenGLテクスチャとして表示し、骨格線をプリミティブで重畳描画する。
"""

from __future__ import annotations
import logging
import numpy as np
from app.modes.base_mode import BaseMode
from app.pose_constants import POSE_CONNECTIONS
from OpenGL.GL import (
    glEnable, glDisable, glGenTextures, glBindTexture,
    glTexImage2D, glTexParameteri,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glBegin, glEnd, glTexCoord2f, glVertex2f,
    glColor3f, glLineWidth, glPointSize, glVertex2f,
    glClear, glClearColor,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_PROJECTION, GL_MODELVIEW, GL_QUADS, GL_LINES, GL_POINTS,
)

logger = logging.getLogger(__name__)

# 骨格線の色（R, G, B）
COLOR_LINE  = (0.0, 1.0, 0.0)   # 緑
COLOR_JOINT = (1.0, 1.0, 0.0)   # 黄
LINE_WIDTH  = 2.0
POINT_SIZE  = 10.0
MIN_VISIBILITY = 0.5             # この値未満のランドマークは描画しない


class Mode1Overlay(BaseMode):
    """モード1：カメラ映像 + 骨格線オーバーレイ表示。"""

    def __init__(self) -> None:
        self._texture_id: int | None = None
        # 骨格線・関節点の太さ倍率（+/- キーで調整）
        self._line_scale: float = 1.0

    @property
    def line_scale(self) -> float:
        return self._line_scale

    def set_line_scale(self, scale: float) -> None:
        self._line_scale = max(0.3, min(5.0, float(scale)))

    def adjust_line_scale(self, delta: float) -> float:
        self.set_line_scale(self._line_scale + delta)
        return self._line_scale

    def initialize(self) -> None:
        """OpenGLテクスチャを初期化する。"""
        self._texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        logger.info("Mode1Overlay 初期化完了")

    def on_mode_enter(self) -> None:
        from OpenGL.GL import glDisable, GL_LIGHTING, GL_DEPTH_TEST
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        logger.info("モード1（オーバーレイ）開始")

    def on_mode_exit(self) -> None:
        logger.info("モード1（オーバーレイ）終了")

    def draw(self, frame: np.ndarray | None,
             results: list, width: int, height: int) -> None:
        """カメラ映像を描画し、骨格線を重畳する。アスペクト比を維持する。"""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if frame is None:
            return

        # アスペクト比を維持したビューポートを計算（レターボックス）
        frame_h, frame_w = frame.shape[:2]
        frame_aspect = frame_w / frame_h
        window_aspect = width / height

        if window_aspect > frame_aspect:
            # ウィンドウが横長 → 上下にパディング
            view_h = height
            view_w = int(height * frame_aspect)
            view_x = (width - view_w) // 2
            view_y = 0
        else:
            # ウィンドウが縦長 → 左右にパディング
            view_w = width
            view_h = int(width / frame_aspect)
            view_x = 0
            view_y = (height - view_h) // 2

        # 2D 投影設定
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, view_w, view_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(view_x, view_y, view_w, view_h)

        # カメラ映像をテクスチャとして描画（BGR→RGB変換）
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            rgb_frame.shape[1], rgb_frame.shape[0],
            0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame.tobytes()
        )
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(view_w, 0)
        glTexCoord2f(1, 1); glVertex2f(view_w, view_h)
        glTexCoord2f(0, 1); glVertex2f(0, view_h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # 骨格線を描画
        if not results:
            return

        line_w = LINE_WIDTH * self._line_scale
        point_sz = POINT_SIZE * self._line_scale

        for result in results:
            lms = result.landmarks
            # 骨格線
            glColor3f(*COLOR_LINE)
            glLineWidth(line_w)
            glBegin(GL_LINES)
            for (a, b) in POSE_CONNECTIONS:
                if (lms[a].visibility >= MIN_VISIBILITY and
                        lms[b].visibility >= MIN_VISIBILITY):
                    glVertex2f(lms[a].x * view_w, lms[a].y * view_h)
                    glVertex2f(lms[b].x * view_w, lms[b].y * view_h)
            glEnd()

            # 関節点
            glColor3f(*COLOR_JOINT)
            glPointSize(point_sz)
            glBegin(GL_POINTS)
            for lm in lms:
                if lm.visibility >= MIN_VISIBILITY:
                    glVertex2f(lm.x * view_w, lm.y * view_h)
            glEnd()
