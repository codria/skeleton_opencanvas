"""
mode2_mannequin.py
モード2：固定背景画像の上に3Dマネキンを正射影で描画する。
カメラ映像は表示しない。背景はクロップ表示。
"""

from __future__ import annotations
import logging
import os
import numpy as np
from PIL import Image, ImageOps
from app.modes.base_mode import BaseMode
from app.mannequin.mannequin_renderer import MannequinRenderer
from app.camera_overlay import CameraOverlay
from OpenGL.GL import (
    glEnable, glDisable, glGenTextures, glBindTexture,
    glTexImage2D, glTexParameteri, glPixelStorei,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glBegin, glEnd, glTexCoord2f, glVertex2f,
    glColor3f, glClearColor, glClear,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_PROJECTION, GL_MODELVIEW, GL_QUADS,
    GL_UNPACK_ALIGNMENT,
)

logger = logging.getLogger(__name__)

DEFAULT_BG_COLOR = (0.1, 0.1, 0.2)


class Mode2Mannequin(BaseMode):
    """モード2：背景画像クロップ + 3Dマネキン正射影表示。"""

    def __init__(self, config) -> None:
        self._config = config
        self._bg_texture_id: int | None = None
        self._bg_width = 1
        self._bg_height = 1
        self._bg_path: str | None = None
        self._renderer = MannequinRenderer()
        self._camera_overlay = CameraOverlay()
        self._overlay_alpha: float = 0.0

    @property
    def renderer(self) -> MannequinRenderer:
        return self._renderer

    def set_camera_overlay_alpha(self, alpha: float) -> None:
        self._overlay_alpha = max(0.0, min(1.0, float(alpha)))

    def initialize(self) -> None:
        self._bg_texture_id = glGenTextures(1)
        self._renderer.setup_lighting()
        self._renderer.load_model(self._resolve_model_path())
        self._camera_overlay.initialize()
        self._load_background(self._config.get(
            "mode2.background_image", "assets/backgrounds/default.jpg"
        ))
        logger.info("Mode2Mannequin 初期化完了")

    def _resolve_model_path(self) -> str:
        # 共通キー（assets.character_model）を優先、無ければ旧 mode3 キーを参照
        candidate = (
            self._config.get("assets.character_model")
            or self._config.get("mode3.character_model")
            or "assets/characters/BlockMan.gltf"
        )
        if os.path.isdir(candidate):
            return os.path.join(candidate, "BlockMan.gltf")
        return candidate

    def set_background(self, image_path: str) -> None:
        """背景画像を差し替える。"""
        self._load_background(image_path)

    def _load_background(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"背景画像が見つかりません: {path}")
            self._bg_path = None
            return
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            self._bg_width, self._bg_height = img.size
            img_data = np.ascontiguousarray(np.array(img, dtype=np.uint8))
            glBindTexture(GL_TEXTURE_2D, self._bg_texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB,
                self._bg_width, self._bg_height,
                0, GL_RGB, GL_UNSIGNED_BYTE, img_data.tobytes()
            )
            self._bg_path = path
            logger.info(f"背景画像読み込み完了: {path}")
        except Exception as e:
            logger.error(f"背景画像の読み込みに失敗: {e}")
            self._bg_path = None

    def on_mode_enter(self) -> None:
        logger.info("モード2（マネキン）開始")

    def on_mode_exit(self) -> None:
        logger.info("モード2（マネキン）終了")

    def draw(self, frame: np.ndarray | None,
             results: list, width: int, height: int) -> None:
        glClearColor(*DEFAULT_BG_COLOR, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        win_aspect = width / height

        # --- 背景をクロップ表示（画面全体・2D） ---
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(0, 0, width, height)

        if self._bg_path:
            bg_aspect = self._bg_width / self._bg_height
            if win_aspect > bg_aspect:
                scale = win_aspect / bg_aspect
                u0, u1 = 0.0, 1.0
                m = (1.0 - 1.0 / scale) / 2
                v0, v1 = m, 1.0 - m
            else:
                scale = bg_aspect / win_aspect
                v0, v1 = 0.0, 1.0
                m = (1.0 - 1.0 / scale) / 2
                u0, u1 = m, 1.0 - m

            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self._bg_texture_id)
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_QUADS)
            glTexCoord2f(u0, v0); glVertex2f(0, 0)
            glTexCoord2f(u1, v0); glVertex2f(width, 0)
            glTexCoord2f(u1, v1); glVertex2f(width, height)
            glTexCoord2f(u0, v1); glVertex2f(0, height)
            glEnd()
            glDisable(GL_TEXTURE_2D)

        # --- マネキンをカメラアスペクト比基準のビューポートで描画 ---
        if frame is not None:
            cam_aspect = frame.shape[1] / frame.shape[0]
        else:
            cam_aspect = 16 / 9

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

        # 実写半透明オーバーレイ（背景画像の上、マネキンの下）
        if self._overlay_alpha > 0.0 and frame is not None:
            self._camera_overlay.draw(
                frame, self._overlay_alpha, view_x, view_y, view_w, view_h
            )

        if results:
            self._renderer.draw_ortho(results, view_x, view_y, view_w, view_h)
