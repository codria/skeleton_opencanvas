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

    # 魔法モードの調整パラメータ（画像座標系。Y は上が 0、下が 1）
    # 構え判定：手首 Y < shoulder_y + (hip_y - shoulder_y) * この比 まで OK。
    # 0.0 = 完全に肩ライン以上、1.0 = 腰まで下げても OK。0.20 で「肩寄り 8 割」。
    _MAGIC_CHARGE_ARM_RATIO: float = 0.20
    # 火球サイズ：両手首距離（画像座標） × このスケール、min/max でクランプ。
    _MAGIC_FIREBALL_SCALE: float = 0.45
    _MAGIC_FIREBALL_MIN: float = 0.03
    _MAGIC_FIREBALL_MAX: float = 0.16
    # 火球サイズはフレーム間 lerp で滑らかに（1.0 で即時、0.2 でゆっくり追従）
    _MAGIC_FIREBALL_LERP: float = 0.25
    # 発射トリガー：charging 中に両手中央が LOOKBACK フレームで
    # LAUNCH_SPEED 以上動いたら発射。そのときの位置差 × VEL_SCALE を初速に。
    # LOOKBACK は 30fps で ~100ms（3 フレーム）を目安。
    # 但し charging 開始から CHARGE_MIN_FRAMES は発射禁止：チャージのために
    # 腕を上げる動作でいきなり撃ってしまうのを防ぐ猶予期間。
    _MAGIC_HAND_HISTORY: int = 10
    _MAGIC_LAUNCH_LOOKBACK: int = 3
    _MAGIC_LAUNCH_SPEED: float = 0.04
    _MAGIC_VEL_SCALE: float = 0.7
    _MAGIC_CHARGE_MIN_FRAMES: int = 8
    _MAGIC_FLY_FRAMES: int = 25

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
        # 発射時の飛翔方向計算のため、charging 中の両手中央位置履歴
        # 直近 _MAGIC_HAND_HISTORY フレーム分保持し、発射時に位置差ベクトルを速度化
        self._magic_hand_history: list[tuple[float, float]] = []

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
        self._magic_hand_history.clear()
        self._prev_right_arm_up = False
        self._prev_left_arm_up = False
        logger.info(f"モード4（体験）開始 サブモード={self._sub_mode}")

    def on_mode_exit(self) -> None:
        # 継続再生中の右腕ドラムロール音は短くフェードして止める
        # （左腕シンバルは短音なので放置で自然に減衰）
        self._sound_bank.fadeout("right_arm_up", ms=150)
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
        右腕（ドラムロール）は長尺音源で「上げてる間だけ鳴らしたい」ので、
        下ろした瞬間に stop する。
        左腕（シンバル）は 1 発鳴らして自然に減衰させる（stop しない）。
        jump / crouch も 1 発鳴らして減衰。
        """
        # 開始側：edge-triggered な event をそのまま再生
        for e in events:
            if self._sound_bank.play(e):
                logger.info(f"[Mode4/楽器] {e} 発火")

        # 終了側：右腕のみ、下ろした瞬間に減衰停止（ぶつ切りは違和感が強いので fadeout）
        cur_r = self._detector.right_arm_up
        if self._prev_right_arm_up and not cur_r:
            if self._sound_bank.fadeout("right_arm_up", ms=300):
                logger.info("[Mode4/楽器] right_arm_up fadeout")
        self._prev_right_arm_up = cur_r
        # 左腕は 1 発鳴らしっぱなしで OK（履歴だけ更新）
        self._prev_left_arm_up = self._detector.left_arm_up

    # --- 魔法サブモード ------------------------------------------------------

    def _update_and_draw_magic(self, results, vx, vy, vw, vh) -> None:
        """魔法の状態機械更新 + 火玉描画。魔法モード独自の判定ロジック：
        - 構え判定：両手首が「肩から腰の 20% 下ライン」より上（腕が疲れない緩めの閾値）
        - 火球サイズ：両手首の距離に比例（大きく広げると大きな火球）
        - 発射トリガー：charging 中に両手中央を LAUNCH_SPEED 以上のスピードで振ったら発射。
          そのスイングベクトル × VEL_SCALE を初速に。構えを解除しても charging は継続。
        """
        # ---- ランドマークから必要情報を抽出 ----
        LS, RS = 11, 12
        LW, RW = 15, 16
        LH, RH = 23, 24
        VIS = 0.5

        charge_pose = False    # 構えポーズが成立しているか
        hand_x = hand_y = None
        hand_width = 0.0
        if results:
            lms = results[0].landmarks
            if (lms[LS].visibility >= VIS and lms[RS].visibility >= VIS and
                    lms[LW].visibility >= VIS and lms[RW].visibility >= VIS and
                    lms[LH].visibility >= VIS and lms[RH].visibility >= VIS):
                shoulder_y = (lms[LS].y + lms[RS].y) / 2
                hip_y = (lms[LH].y + lms[RH].y) / 2
                # 構え許容ライン Y = shoulder_y から hip_y に向かって RATIO 進んだ位置
                # 画像座標は上端 0 / 下端 1 なので、この Y より小さい（上）ならポーズ OK
                thresh_y = shoulder_y + (hip_y - shoulder_y) * self._MAGIC_CHARGE_ARM_RATIO
                if lms[LW].y < thresh_y and lms[RW].y < thresh_y:
                    charge_pose = True
                # 両手中央 & 距離
                hand_x = (lms[LW].x + lms[RW].x) / 2
                hand_y = (lms[LW].y + lms[RW].y) / 2
                dx = lms[LW].x - lms[RW].x
                dy = lms[LW].y - lms[RW].y
                hand_width = (dx * dx + dy * dy) ** 0.5

        # 位置履歴を更新（発射時の速度ベクトル用）
        if hand_x is not None and hand_y is not None:
            self._magic_hand_history.append((hand_x, hand_y))
            if len(self._magic_hand_history) > self._MAGIC_HAND_HISTORY:
                self._magic_hand_history.pop(0)

        # ---- 状態機械 ----
        if self._magic_state == self._MAGIC_IDLE:
            if charge_pose and hand_x is not None:
                self._magic_state = self._MAGIC_CHARGING
                self._magic_x = hand_x
                self._magic_y = hand_y
                self._magic_size = self._MAGIC_FIREBALL_MIN
                self._magic_frames = 0
                # 腕を上げてきた履歴はここで捨て、charging 中の動きだけを
                # 発射スイング判定の対象にする（現フレームだけ再挿入）
                self._magic_hand_history.clear()
                self._magic_hand_history.append((hand_x, hand_y))
                self._sound_bank.play("magic_charge")
                logger.info("[Mode4/魔法] charging 開始")

        elif self._magic_state == self._MAGIC_CHARGING:
            self._magic_frames += 1
            if hand_x is not None:
                self._magic_x = hand_x
                self._magic_y = hand_y
                # 両手幅に応じた目標サイズ（min/max でクランプ）を lerp で滑らかに追従
                target = max(
                    self._MAGIC_FIREBALL_MIN,
                    min(self._MAGIC_FIREBALL_MAX,
                        hand_width * self._MAGIC_FIREBALL_SCALE),
                )
                self._magic_size += (target - self._magic_size) * self._MAGIC_FIREBALL_LERP
            # 猶予フレームを過ぎたらスイング判定 → 閾値超で発射
            launch_vel = None
            if self._magic_frames >= self._MAGIC_CHARGE_MIN_FRAMES:
                launch_vel = self._check_swing_launch()
            if launch_vel is not None:
                self._magic_state = self._MAGIC_FLYING
                self._magic_frames = 0
                self._magic_vx, self._magic_vy = launch_vel
                logger.info(
                    f"[Mode4/魔法] 発射 size={self._magic_size:.3f} "
                    f"v=({self._magic_vx:+.3f}, {self._magic_vy:+.3f})"
                )
            # ポーズ解除でのキャンセルはしない：構えを下ろしても charging は継続、
            # 発射トリガーは常にスイング。

        elif self._magic_state == self._MAGIC_FLYING:
            self._magic_x += self._magic_vx
            self._magic_y += self._magic_vy
            self._magic_frames += 1
            if (self._magic_frames > self._MAGIC_FLY_FRAMES or
                    self._magic_y < 0.02 or self._magic_y > 0.98 or
                    self._magic_x < 0.02 or self._magic_x > 0.98):
                self._magic_state = self._MAGIC_EXPLODE
                self._magic_frames = 0
                self._sound_bank.play("magic_hit")
                logger.info("[Mode4/魔法] 爆発")

        elif self._magic_state == self._MAGIC_EXPLODE:
            # 拡大しながらフェード（火球サイズ * 3 まで膨らむ）
            explode_max = max(0.22, self._magic_size * 3.0)
            self._magic_size = min(explode_max, self._magic_size + 0.012)
            self._magic_frames += 1
            if self._magic_frames > 22:
                self._magic_state = self._MAGIC_IDLE

        # 描画
        if self._magic_state != self._MAGIC_IDLE:
            self._draw_fireball(vx, vy, vw, vh)

    def _check_swing_launch(self) -> tuple[float, float] | None:
        """charging 中に毎フレーム呼ぶ。両手中央の LOOKBACK フレーム間位置差の
        大きさが LAUNCH_SPEED を超えたら (vx, vy) を返す（＝発射）。
        まだ振られていない・履歴不足なら None（＝charging 継続）。
        """
        h = self._magic_hand_history
        if len(h) < 2:
            return None
        n = min(self._MAGIC_LAUNCH_LOOKBACK, len(h) - 1)
        x0, y0 = h[-1 - n]
        x1, y1 = h[-1]
        dx = x1 - x0
        dy = y1 - y0
        speed = (dx * dx + dy * dy) ** 0.5
        if speed < self._MAGIC_LAUNCH_SPEED:
            return None
        return dx * self._MAGIC_VEL_SCALE, dy * self._MAGIC_VEL_SCALE

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
            # サブモード切替時も継続音（右腕ドラムロール）を短くフェードして止める
            self._sound_bank.fadeout("right_arm_up", ms=150)
            self._prev_right_arm_up = False
            self._prev_left_arm_up = False
            self._sub_mode = sub
            self._detector.reset()
            self._magic_state = self._MAGIC_IDLE
            self._magic_hand_history.clear()
            logger.info(f"Mode4 サブモード: {sub}")

    def toggle_sub_mode(self) -> str:
        try:
            idx = self.SUB_MODES.index(self._sub_mode)
        except ValueError:
            idx = -1
        new_sub = self.SUB_MODES[(idx + 1) % len(self.SUB_MODES)]
        self.set_sub_mode(new_sub)
        return new_sub
