"""
camera_overlay.py
Mode2/Mode3 の背景の上に実写フレーム（カメラ or 動画）を半透明で重ね描きする
ためのユーティリティ。テクスチャを再利用して毎フレーム作り直さない。
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import (
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glPixelStorei,
    glEnable, glDisable, glBlendFunc, glColor4f,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glBegin, glEnd, glTexCoord2f, glVertex2f,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_PROJECTION, GL_MODELVIEW, GL_QUADS,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_DEPTH_TEST, GL_LIGHTING,
    GL_UNPACK_ALIGNMENT,
)


class CameraOverlay:
    """カメラ/動画フレームを指定アルファで半透明オーバーレイ描画する。
    OpenGL コンテキスト内で initialize() してから draw() で使う。
    """

    def __init__(self) -> None:
        self._texture_id: int | None = None
        self._tex_w: int = 0
        self._tex_h: int = 0

    def initialize(self) -> None:
        """OpenGL テクスチャを 1 つ確保する。"""
        self._texture_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def draw(self, frame: np.ndarray | None, alpha: float,
             view_x: int, view_y: int, view_w: int, view_h: int) -> None:
        """frame を view 内に半透明で描画する。
        alpha <= 0 や frame None の時は何もしない。
        viewport に対して画像のアスペクト比を維持してフィットさせる（レターボックス）。
        """
        if frame is None or alpha <= 0.0 or self._texture_id is None:
            return

        # BGR → RGB
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        img_h, img_w = rgb.shape[:2]

        # アスペクト比維持の uv 計算（テクスチャは画像全体、quad は viewport 全体）
        view_aspect = view_w / max(view_h, 1)
        img_aspect = img_w / max(img_h, 1)
        if view_aspect > img_aspect:
            # ビューが横長 → 上下を画像端まで使い、左右に余白
            ratio = img_aspect / view_aspect
            margin = (1.0 - ratio) / 2
            x0, x1 = margin, 1.0 - margin
            y0, y1 = 0.0, 1.0
        else:
            ratio = view_aspect / img_aspect
            margin = (1.0 - ratio) / 2
            x0, x1 = 0.0, 1.0
            y0, y1 = margin, 1.0 - margin

        # テクスチャアップロード（サイズが変わったときは glTexImage2D、同じなら subimage でも可だが簡便のため毎回 TexImage2D）
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            img_w, img_h, 0, GL_RGB, GL_UNSIGNED_BYTE,
            rgb.tobytes()
        )
        self._tex_w, self._tex_h = img_w, img_h

        # 描画設定
        glViewport(view_x, view_y, view_w, view_h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, float(alpha))
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(x0, y0)
        glTexCoord2f(1.0, 0.0); glVertex2f(x1, y0)
        glTexCoord2f(1.0, 1.0); glVertex2f(x1, y1)
        glTexCoord2f(0.0, 1.0); glVertex2f(x0, y1)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
