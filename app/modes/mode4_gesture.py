"""
mode4_gesture.py
モード4：ジェスチャー体験モード。
シンプルな骨格線オーバーレイ（「AI が体を認識できてる」ことを直感的に見せる）＋
サブモード切替（楽器演奏 / 魔法エフェクト etc.）で「遊び」を提供する。

Phase 1（現状）: 骨組みのみ。骨格線描画とサブモード切替インフラだけ実装。
Phase 2: 楽器サブモード（QSoundEffect で wav 再生）
Phase 3: 魔法サブモード（OpenGL パーティクル）
"""

from __future__ import annotations
import logging
from OpenGL.GL import (
    glClearColor, glClear,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glEnable, glDisable, glColor3f, glLineWidth, glPointSize,
    glBegin, glEnd, glVertex2f, glBlendFunc, glHint,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION, GL_MODELVIEW,
    GL_LINES, GL_POINTS,
    GL_DEPTH_TEST, GL_LIGHTING, GL_BLEND,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT,
    GL_POINT_SMOOTH, GL_POINT_SMOOTH_HINT,
    GL_NICEST,
)

from app.modes.base_mode import BaseMode
from app.pose_constants import POSE_CONNECTIONS

logger = logging.getLogger(__name__)


class Mode4Gesture(BaseMode):
    """ジェスチャー体験モード。骨格線オーバーレイ + サブモードで遊びを追加。"""

    # サブモード（S キーで循環）
    SUB_MODES: tuple[str, ...] = ("instrument", "magic")
    SUB_MODE_LABELS: dict[str, str] = {
        "instrument": "楽器",
        "magic": "魔法",
    }
    MIN_VIS = 0.4

    def __init__(self, config) -> None:
        self._config = config
        self._sub_mode: str = "instrument"

    # --- BaseMode 実装 -------------------------------------------------------

    def initialize(self) -> None:
        # OpenGL リソースの初期化。Phase 1 では特になし。
        pass

    def draw(self, frame, results, width: int, height: int) -> None:
        # 背景は暗めで統一（暗いほうがジェスチャーの動きが見やすい）
        glClearColor(0.05, 0.05, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not results:
            return

        # --- カメラアスペクトに合わせたビューポート ---
        if frame is not None:
            cam_aspect = frame.shape[1] / frame.shape[0]
        else:
            cam_aspect = 16 / 9
        win_aspect = width / max(height, 1)
        if win_aspect > cam_aspect:
            view_h = height
            view_w = int(height * cam_aspect)
            view_x = (width - view_w) // 2
            view_y = 0
        else:
            view_w = width
            view_h = int(width / cam_aspect)
            view_x = 0
            view_y = (height - view_h) // 2

        self._draw_skeleton(results, view_x, view_y, view_w, view_h)

        # サブモード固有の描画（Phase 2/3 で拡張）。今は no-op。
        if self._sub_mode == "instrument":
            self._draw_instrument_overlay(results, view_x, view_y, view_w, view_h)
        elif self._sub_mode == "magic":
            self._draw_magic_overlay(results, view_x, view_y, view_w, view_h)

    # --- 骨格線オーバーレイ --------------------------------------------------

    def _draw_skeleton(self, results, vx: int, vy: int, vw: int, vh: int) -> None:
        """gl_widget.draw_bone_overlay と同等の骨格線描画。
        黄色線＋シアン点、アンチエイリアス付き。
        """
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, vw, vh, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(vx, vy, vw, vh)

        try:
            for result in results:
                lms = result.landmarks

                # ボーン線（黄色）
                glColor3f(1.0, 1.0, 0.0)
                glLineWidth(3.0)
                glBegin(GL_LINES)
                for (a, b) in POSE_CONNECTIONS:
                    if (lms[a].visibility >= self.MIN_VIS and
                            lms[b].visibility >= self.MIN_VIS):
                        glVertex2f(lms[a].x * vw, lms[a].y * vh)
                        glVertex2f(lms[b].x * vw, lms[b].y * vh)
                glEnd()

                # 関節点（シアン）
                glColor3f(0.0, 1.0, 1.0)
                glPointSize(10.0)
                glBegin(GL_POINTS)
                for lm in lms:
                    if lm.visibility >= self.MIN_VIS:
                        glVertex2f(lm.x * vw, lm.y * vh)
                glEnd()
        finally:
            glDisable(GL_LINE_SMOOTH)
            glDisable(GL_POINT_SMOOTH)
            glDisable(GL_BLEND)

    # --- サブモード固有描画（Phase 2/3 で実装）-------------------------------

    def _draw_instrument_overlay(self, results, vx, vy, vw, vh) -> None:
        """楽器モード：ヒットゾーンや状態表示など。Phase 2 で実装。"""
        pass

    def _draw_magic_overlay(self, results, vx, vy, vw, vh) -> None:
        """魔法モード：パーティクルエフェクトなど。Phase 3 で実装。"""
        pass

    # --- サブモード切替 ------------------------------------------------------

    @property
    def sub_mode(self) -> str:
        return self._sub_mode

    @property
    def sub_mode_label(self) -> str:
        return self.SUB_MODE_LABELS.get(self._sub_mode, self._sub_mode)

    def set_sub_mode(self, sub: str) -> None:
        if sub in self.SUB_MODES:
            self._sub_mode = sub
            logger.info(f"Mode4 サブモード: {sub}")

    def toggle_sub_mode(self) -> str:
        """S キーで循環切替。次のサブモード名を返す。"""
        try:
            idx = self.SUB_MODES.index(self._sub_mode)
        except ValueError:
            idx = -1
        self._sub_mode = self.SUB_MODES[(idx + 1) % len(self.SUB_MODES)]
        logger.info(f"Mode4 サブモード切替: {self._sub_mode}")
        return self._sub_mode
