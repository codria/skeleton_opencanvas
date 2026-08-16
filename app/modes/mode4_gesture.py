"""
mode4_gesture.py
モード4：ジェスチャー体験モード。
背景にカメラ映像（Mode1 同等）、その上に骨格線オーバーレイ、
サブモード（楽器 / 魔法）でジェスチャー連動の効果音・エフェクトを出す。

サブモード:
    instrument : 右腕/左腕/ジャンプ/しゃがみで別々の音を鳴らす
    magic      : 両腕を肩以上に上げると火玉 charging、
                 両腕を下ろすと火玉が発射→着弾で爆発
"""

from __future__ import annotations
import logging
from OpenGL.GL import (
    glClearColor, glClear,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glEnable, glDisable, glColor3f, glColor4f, glLineWidth, glPointSize,
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
from app.camera_overlay import CameraOverlay
from app.sound_bank import SoundBank
from app.gesture_detector import GestureDetector

logger = logging.getLogger(__name__)


class Mode4Gesture(BaseMode):
    """ジェスチャー体験モード。"""

    SUB_MODES: tuple[str, ...] = ("instrument", "magic")
    SUB_MODE_LABELS: dict[str, str] = {
        "instrument": "楽器",
        "magic": "魔法",
    }
    MIN_VIS = 0.4

    # 魔法モードの内部状態
    _MAGIC_IDLE = "idle"
    _MAGIC_CHARGING = "charging"
    _MAGIC_FLYING = "flying"
    _MAGIC_EXPLODE = "explode"

    def __init__(self, config) -> None:
        self._config = config
        self._sub_mode: str = "instrument"

        # 描画・音・ジェスチャー検出器
        self._camera_overlay = CameraOverlay()
        self._sound_bank = SoundBank("assets/sounds")
        self._detector = GestureDetector()

        # 魔法モード状態
        self._magic_state = self._MAGIC_IDLE
        self._magic_x: float = 0.5    # 画像座標 0-1
        self._magic_y: float = 0.5
        self._magic_vx: float = 0.0   # 単位: 画像座標/フレーム
        self._magic_vy: float = 0.0
        self._magic_size: float = 0.02   # 半径（画像座標比）
        self._magic_frames: int = 0
        # 発射時の飛翔方向計算のため、charging 中の手位置履歴
        self._charge_hand_x: float | None = None
        self._charge_hand_y: float | None = None

        # 楽器モード：腕下ろしで音を止めるための前フレーム状態
        self._prev_right_arm_up: bool = False
        self._prev_left_arm_up: bool = False

    # --- BaseMode 実装 -------------------------------------------------------

    def initialize(self) -> None:
        self._camera_overlay.initialize()
        logger.info("Mode4Gesture 初期化完了")

    def on_mode_enter(self) -> None:
        # 状態リセット（前回の魔法状態などを持ち越さない）
        self._detector.reset()
        self._magic_state = self._MAGIC_IDLE
        self._prev_right_arm_up = False
        self._prev_left_arm_up = False
        logger.info(f"モード4（体験）開始 サブモード={self._sub_mode}")

    def on_mode_exit(self) -> None:
        # 継続再生中の腕上げ系音を停止しておく
        self._sound_bank.stop("right_arm_up")
        self._sound_bank.stop("left_arm_up")
        logger.info("モード4（体験）終了")

    def draw(self, frame, results, width: int, height: int) -> None:
        # 背景は黒でクリア
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # カメラアスペクトに合わせたビューポート
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

        # 背景：カメラ映像を不透明で描く（Mode1 と同じ見た目）
        if frame is not None:
            self._camera_overlay.draw(frame, 1.0, view_x, view_y, view_w, view_h)

        # 骨格線 + サブモード処理
        self._draw_skeleton(results, view_x, view_y, view_w, view_h)

        # ジェスチャー検出（両サブモード共通で状態更新）
        events = self._detector.detect(results)

        if self._sub_mode == "instrument":
            self._on_instrument_events(events)
        elif self._sub_mode == "magic":
            self._update_and_draw_magic(results, view_x, view_y, view_w, view_h)

    # --- 骨格線オーバーレイ --------------------------------------------------

    def _draw_skeleton(self, results, vx: int, vy: int, vw: int, vh: int) -> None:
        if not results:
            return
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

    # --- 楽器サブモード ------------------------------------------------------

    def _on_instrument_events(self, events: list[str]) -> None:
        """検出された event を音再生キーとして扱う。
        腕上げ系（right_arm_up / left_arm_up）は上げてる間だけ鳴らしたいので、
        「上げた瞬間 → play」「下ろした瞬間 → stop」の両エッジで制御する。
        jump / crouch は 1 発鳴らして自然に減衰させる（短音源想定）。
        """
        # 開始側：edge-triggered な event をそのまま再生
        for e in events:
            if self._sound_bank.play(e):
                logger.info(f"[Mode4/楽器] {e} 発火")

        # 終了側：右腕/左腕は下ろした瞬間に停止（ドラムロール等の長尺音源対策）
        cur_r = self._detector.right_arm_up
        cur_l = self._detector.left_arm_up
        if self._prev_right_arm_up and not cur_r:
            if self._sound_bank.stop("right_arm_up"):
                logger.info("[Mode4/楽器] right_arm_up 停止")
        if self._prev_left_arm_up and not cur_l:
            if self._sound_bank.stop("left_arm_up"):
                logger.info("[Mode4/楽器] left_arm_up 停止")
        self._prev_right_arm_up = cur_r
        self._prev_left_arm_up = cur_l

    # --- 魔法サブモード ------------------------------------------------------

    def _update_and_draw_magic(self, results, vx, vy, vw, vh) -> None:
        """魔法の状態機械更新 + 火玉描画。"""
        # 状態機械の更新
        both_up = self._detector.both_arms_up

        # 両手中央位置を取得（画像座標）
        hand_x = hand_y = None
        if results:
            lms = results[0].landmarks
            _LW, _RW = 15, 16
            if lms[_LW].visibility >= 0.5 and lms[_RW].visibility >= 0.5:
                hand_x = (lms[_LW].x + lms[_RW].x) / 2
                hand_y = (lms[_LW].y + lms[_RW].y) / 2

        if self._magic_state == self._MAGIC_IDLE:
            # 両腕上げが確立したら charging へ
            if both_up and hand_x is not None:
                self._magic_state = self._MAGIC_CHARGING
                self._magic_x = hand_x
                self._magic_y = hand_y
                self._magic_size = 0.02
                self._charge_hand_x = hand_x
                self._charge_hand_y = hand_y
                self._sound_bank.play("magic_charge")
                logger.info("[Mode4/魔法] charging 開始")

        elif self._magic_state == self._MAGIC_CHARGING:
            if hand_x is not None:
                self._magic_x = hand_x
                self._magic_y = hand_y
                self._charge_hand_x = hand_x
                self._charge_hand_y = hand_y
            # 火玉は徐々に大きく（最大 0.06）
            self._magic_size = min(0.06, self._magic_size + 0.0015)
            if not both_up:
                # 両腕降ろした → 発射
                self._magic_state = self._MAGIC_FLYING
                self._magic_frames = 0
                # 発射方向：シンプルに上方向、少し前フレーム手位置とのズレを加味
                self._magic_vx = 0.0
                self._magic_vy = -0.025    # 上に飛ぶ
                logger.info("[Mode4/魔法] 発射")

        elif self._magic_state == self._MAGIC_FLYING:
            self._magic_x += self._magic_vx
            self._magic_y += self._magic_vy
            self._magic_frames += 1
            # 画面外に出るか、一定フレーム経過で爆発
            if (self._magic_frames > 25 or
                    self._magic_y < 0.02 or self._magic_y > 0.98 or
                    self._magic_x < 0.02 or self._magic_x > 0.98):
                self._magic_state = self._MAGIC_EXPLODE
                self._magic_frames = 0
                self._sound_bank.play("magic_hit")
                logger.info("[Mode4/魔法] 爆発")

        elif self._magic_state == self._MAGIC_EXPLODE:
            # 徐々に拡大しながらフェード
            self._magic_size = min(0.22, self._magic_size + 0.012)
            self._magic_frames += 1
            if self._magic_frames > 22:
                self._magic_state = self._MAGIC_IDLE

        # 描画
        if self._magic_state != self._MAGIC_IDLE:
            self._draw_fireball(vx, vy, vw, vh)

    def _draw_fireball(self, vx, vy, vw, vh) -> None:
        """火玉/爆発を描く。オレンジ色の点で表現（サイズ = 半径）。"""
        # 状態別のアルファ
        if self._magic_state == self._MAGIC_EXPLODE:
            # フェード（1 → 0）
            alpha = max(0.0, 1.0 - self._magic_frames / 22.0)
            r, g, b = 1.0, 0.65, 0.15
        elif self._magic_state == self._MAGIC_CHARGING:
            alpha = 0.8
            r, g, b = 1.0, 0.5, 0.1
        else:  # flying
            alpha = 0.95
            r, g, b = 1.0, 0.55, 0.1

        # 描画設定
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # viewport を確認
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, vw, vh, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(vx, vy, vw, vh)

        try:
            # 点サイズは半径 * 短辺 の 2 倍程度
            short = min(vw, vh)
            pt_size = max(8.0, self._magic_size * short * 2.0)
            glPointSize(pt_size)
            glColor4f(r, g, b, alpha)
            glBegin(GL_POINTS)
            glVertex2f(self._magic_x * vw, self._magic_y * vh)
            glEnd()
            # 内側にもう一段小さく明るい点（ハイライト）
            glPointSize(max(4.0, pt_size * 0.5))
            glColor4f(1.0, 0.95, 0.6, alpha)
            glBegin(GL_POINTS)
            glVertex2f(self._magic_x * vw, self._magic_y * vh)
            glEnd()
        finally:
            glDisable(GL_POINT_SMOOTH)
            glDisable(GL_BLEND)

    # --- サブモード切替 ------------------------------------------------------

    @property
    def sub_mode(self) -> str:
        return self._sub_mode

    @property
    def sub_mode_label(self) -> str:
        return self.SUB_MODE_LABELS.get(self._sub_mode, self._sub_mode)

    def set_sub_mode(self, sub: str) -> None:
        if sub in self.SUB_MODES:
            # サブモード切替時も継続音を止めておく
            self._sound_bank.stop("right_arm_up")
            self._sound_bank.stop("left_arm_up")
            self._prev_right_arm_up = False
            self._prev_left_arm_up = False
            self._sub_mode = sub
            self._detector.reset()
            self._magic_state = self._MAGIC_IDLE
            logger.info(f"Mode4 サブモード: {sub}")

    def toggle_sub_mode(self) -> str:
        try:
            idx = self.SUB_MODES.index(self._sub_mode)
        except ValueError:
            idx = -1
        new_sub = self.SUB_MODES[(idx + 1) % len(self.SUB_MODES)]
        self.set_sub_mode(new_sub)
        return new_sub
