"""
mode3_3d.py
モード3：3D空間（床グリッド）+ 自動回転視点で BlockMan マネキンを透視投影描画する。
"""

from __future__ import annotations
import math
import os
import time
import logging
import numpy as np
from app.modes.base_mode import BaseMode
from app.mannequin.mannequin_renderer import MannequinRenderer, VIEW_DISTANCE
from app.camera_overlay import CameraOverlay
from OpenGL.GL import (
    glClearColor, glClear,
    glEnable, glDisable, glColor3f, glLineWidth,
    glMatrixMode, glLoadIdentity, glViewport,
    glBegin, glEnd, glVertex3f,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION, GL_MODELVIEW,
    GL_DEPTH_TEST, GL_LIGHTING,
    GL_LINES,
)
from OpenGL.GLU import gluPerspective, gluLookAt

logger = logging.getLogger(__name__)

BG_COLOR = (0.08, 0.10, 0.14)
GRID_COLOR = (0.30, 0.34, 0.42)
GRID_HALF_SIZE = 3.0
GRID_STEP = 0.5
# VIEW_DISTANCE は mannequin_renderer から import（メートル単位）


class Mode3D(BaseMode):
    """モード3：3D空間 + 自動回転視点 + マネキン描画。"""

    def __init__(self, config) -> None:
        self._config = config
        self._renderer = MannequinRenderer()
        self._last_time: float | None = None
        # rotation_speed は deg/sec
        self._rotation_speed: float = float(
            config.get("mode3.rotation_speed", 30.0)
        )
        self._rotation_paused: bool = False
        self._camera_overlay = CameraOverlay()
        self._overlay_alpha: float = 0.0

    @property
    def renderer(self) -> MannequinRenderer:
        return self._renderer

    def set_camera_overlay_alpha(self, alpha: float) -> None:
        self._overlay_alpha = max(0.0, min(1.0, float(alpha)))

    # --- 回転制御 ----------------------------------------------------------

    @property
    def rotation_speed(self) -> float:
        return self._rotation_speed

    def set_rotation_speed(self, speed: float) -> None:
        """回転速度を設定する（deg/sec）。"""
        self._rotation_speed = max(-360.0, min(360.0, float(speed)))

    @property
    def rotation_paused(self) -> bool:
        return self._rotation_paused

    def set_rotation_paused(self, paused: bool) -> None:
        self._rotation_paused = bool(paused)

    def toggle_rotation_paused(self) -> bool:
        self._rotation_paused = not self._rotation_paused
        return self._rotation_paused

    def set_view_angle(self, angle_deg: float) -> None:
        """視点角度を直接指定する（角度スライダー操作用、自動回転を上書き）。"""
        self._renderer.set_rotation_y(angle_deg)

    def initialize(self) -> None:
        self._renderer.setup_lighting()
        model_path = self._resolve_model_path()
        self._renderer.load_model(model_path)
        self._camera_overlay.initialize()
        logger.info("Mode3D 初期化完了")

    def _resolve_model_path(self) -> str:
        # 共通キー（assets.character_model）を優先、無ければ mode3 セクションをフォールバック
        candidate = (
            self._config.get("assets.character_model")
            or self._config.get("mode3.character_model")
            or "assets/characters/BlockMan.gltf"
        )
        if os.path.isdir(candidate):
            return os.path.join(candidate, "BlockMan.gltf")
        return candidate

    def set_character(self, character_path: str) -> None:
        """キャラクターモデルを差し替える。"""
        if os.path.isdir(character_path):
            character_path = os.path.join(character_path, "BlockMan.gltf")
        self._renderer.load_model(character_path)
        logger.info(f"キャラクター差し替え: {character_path}")

    def on_mode_enter(self) -> None:
        self._last_time = None
        logger.info("モード3（3Dキャラクター）開始")

    def on_mode_exit(self) -> None:
        logger.info("モード3（3Dキャラクター）終了")

    def on_wheel(self, delta_y: int) -> None:
        """ホイールで視点距離を調整する。上スクロールで近づく、下スクロールで遠ざかる。"""
        # 1 ノッチ (120) で 0.3m 動く
        step = -delta_y / 120.0 * 0.3
        new_d = self._renderer.adjust_view_distance(step)
        logger.info(f"カメラ距離: {new_d:.2f}m")

    def draw(self, frame: np.ndarray | None,
             results: list, width: int, height: int) -> None:
        # delta_time ベースで回転角度を更新（停止中は進めない）
        now = time.perf_counter()
        dt = 0.0 if self._last_time is None else (now - self._last_time)
        self._last_time = now
        if not self._rotation_paused:
            self._renderer.update_rotation(self._rotation_speed * dt)

        glClearColor(*BG_COLOR, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._setup_view(width, height)
        self._draw_grid()

        # 実写半透明オーバーレイ（グリッドの上、マネキンの下）
        if self._overlay_alpha > 0.0 and frame is not None:
            self._camera_overlay.draw(
                frame, self._overlay_alpha, 0, 0, width, height
            )

        if results:
            # マネキン描画（MannequinRenderer 側で再度 setup_view される）
            self._renderer.draw_perspective(results, 0, 0, width, height)

    def _setup_view(self, width: int, height: int) -> None:
        """グリッド描画用の透視投影＋自動回転視点を設定する。
        視点距離は renderer.view_distance（ホイールで変動）に追従。
        """
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / max(height, 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        angle = self._renderer.rotation_y
        d = self._renderer.view_distance
        gluLookAt(
            math.sin(math.radians(angle)) * d,
            0.0,
            math.cos(math.radians(angle)) * d,
            0, 0, 0,
            0, -1, 0,
        )

    def _draw_grid(self) -> None:
        """マネキンの足元に格子線を描画する。床位置は renderer のスケールに連動。"""
        floor_y = self._renderer.primitive_foot_y
        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glColor3f(*GRID_COLOR)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        n_lines = int(GRID_HALF_SIZE / GRID_STEP)
        for i in range(-n_lines, n_lines + 1):
            x = i * GRID_STEP
            glVertex3f(x, floor_y, -GRID_HALF_SIZE)
            glVertex3f(x, floor_y,  GRID_HALF_SIZE)
            glVertex3f(-GRID_HALF_SIZE, floor_y, x)
            glVertex3f( GRID_HALF_SIZE, floor_y, x)
        glEnd()
