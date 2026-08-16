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
import math
import random
from OpenGL.GL import (
    glClearColor, glClear,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glEnable, glDisable, glColor3f, glColor4f, glLineWidth, glPointSize,
    glBegin, glEnd, glVertex2f, glBlendFunc, glHint,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION, GL_MODELVIEW,
    GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, GL_POINTS,
    GL_DEPTH_TEST, GL_LIGHTING, GL_BLEND,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
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
    # エフェクト系
    _MAGIC_EXPLODE_FRAMES: int = 28              # 爆発表示総フレーム
    _MAGIC_CHARGE_SPARK_PROB: float = 0.75       # charging 中の spark 生成確率/frame
    _MAGIC_FLIGHT_SPARK_PROB: float = 1.0        # flying 中の spark 生成確率/frame
    _MAGIC_FLIGHT_TRAIL_LEN: int = 22            # 軌跡ライン頂点数
    _MAGIC_EXPLOSION_PARTICLES: int = 48         # 爆発時パーティクル数
    _MAGIC_SHOCKWAVE_RADIUS: float = 0.28        # 衝撃波リング最大半径（画像座標比）

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

        # エフェクト状態
        # particles: 各要素 = [x, y, vx, vy, life, max_life, size, r, g, b]
        # x/y/vx/vy/size は画像座標比、life はフレーム
        self._particles: list[list[float]] = []
        # 投射中の軌跡ポイント（描画は末尾から古い方へ、alpha fade）
        self._flight_trail: list[tuple[float, float]] = []

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
        self._particles.clear()
        self._flight_trail.clear()
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
            # 軌道 spark を確率生成
            if random.random() < self._MAGIC_CHARGE_SPARK_PROB:
                self._spawn_charge_spark()
            # 猶予フレームを過ぎたらスイング判定 → 閾値超で発射
            launch_vel = None
            if self._magic_frames >= self._MAGIC_CHARGE_MIN_FRAMES:
                launch_vel = self._check_swing_launch()
            if launch_vel is not None:
                self._magic_state = self._MAGIC_FLYING
                self._magic_frames = 0
                self._magic_vx, self._magic_vy = launch_vel
                self._flight_trail.clear()
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
            # 軌跡・尾 spark
            self._flight_trail.append((self._magic_x, self._magic_y))
            if len(self._flight_trail) > self._MAGIC_FLIGHT_TRAIL_LEN:
                self._flight_trail.pop(0)
            if random.random() < self._MAGIC_FLIGHT_SPARK_PROB:
                self._spawn_flight_spark()
            if (self._magic_frames > self._MAGIC_FLY_FRAMES or
                    self._magic_y < 0.02 or self._magic_y > 0.98 or
                    self._magic_x < 0.02 or self._magic_x > 0.98):
                self._magic_state = self._MAGIC_EXPLODE
                self._magic_frames = 0
                self._spawn_explosion_burst()
                self._sound_bank.play("magic_hit")
                logger.info("[Mode4/魔法] 爆発")

        elif self._magic_state == self._MAGIC_EXPLODE:
            self._magic_frames += 1
            if self._magic_frames > self._MAGIC_EXPLODE_FRAMES:
                self._magic_state = self._MAGIC_IDLE
                self._flight_trail.clear()

        # パーティクル更新（全状態共通、EXPLODE 完了後も生き残ってれば描く）
        self._update_particles()

        # ---- 描画 ----
        # IDLE 中で余韻（particle / trail）もなければセットアップ含めて何もしない
        if (self._magic_state == self._MAGIC_IDLE and
                not self._particles and not self._flight_trail):
            return
        self._setup_2d(vx, vy, vw, vh)
        # 尾（FLYING 中と、EXPLODE 直後の余韻）
        if self._flight_trail:
            self._draw_flight_trail(vw, vh)
        # コア（IDLE 以外）
        if self._magic_state == self._MAGIC_CHARGING:
            self._draw_fireball_core(vw, vh, pulsing=True)
            self._draw_charge_rays(vw, vh)
        elif self._magic_state == self._MAGIC_FLYING:
            self._draw_fireball_core(vw, vh, pulsing=False)
        elif self._magic_state == self._MAGIC_EXPLODE:
            self._draw_explosion_flash(vw, vh)
            self._draw_shockwave(vw, vh)
        # パーティクル（最後に上乗せ）
        self._draw_particles(vw, vh)

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

    # --- 描画セットアップ --------------------------------------------------

    def _setup_2d(self, vx, vy, vw, vh) -> None:
        """魔法エフェクト描画用の 2D 座標系 + 加算合成ブレンドを設定する。
        Ortho は左上原点（骨格線と同じ）。加算ブレンドで火っぽい光が重なる。
        """
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # 加算：重なりで白飛び感を出す
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, vw, vh, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(vx, vy, vw, vh)

    # --- 火球コア（レイヤー glow）-------------------------------------------

    def _draw_fireball_core(self, vw: int, vh: int, pulsing: bool) -> None:
        """火球本体：5 層の同心 GL_POINTS で「にじむ光球」を作る。
        pulsing=True なら charging 中のサイズ脈動を掛ける。
        """
        short = min(vw, vh)
        cx = self._magic_x * vw
        cy = self._magic_y * vh
        size = self._magic_size
        if pulsing:
            size *= 1.0 + math.sin(self._magic_frames * 0.28) * 0.10

        # (半径スケール, α, r, g, b)
        layers = (
            (2.6, 0.10, 1.0, 0.35, 0.05),   # 外周 halo（赤めの残光）
            (1.8, 0.20, 1.0, 0.50, 0.10),
            (1.2, 0.35, 1.0, 0.65, 0.20),
            (0.85, 0.60, 1.0, 0.85, 0.45),  # 明るいシェル
            (0.55, 0.95, 1.0, 1.0, 0.85),   # 白熱コア
        )
        for scale, alpha, r, g, b in layers:
            pt = max(4.0, size * short * 2.0 * scale)
            glPointSize(pt)
            glColor4f(r, g, b, alpha)
            glBegin(GL_POINTS)
            glVertex2f(cx, cy)
            glEnd()

    # --- チャージ用エフェクト -----------------------------------------------

    def _draw_charge_rays(self, vw: int, vh: int) -> None:
        """charging 中：中心から放射する 6 本の光線が徐々に回転。"""
        short = min(vw, vh)
        cx = self._magic_x * vw
        cy = self._magic_y * vh
        outer = self._magic_size * short * 1.8
        inner = self._magic_size * short * 0.6
        rot = self._magic_frames * 0.06
        glLineWidth(2.0)
        glColor4f(1.0, 0.7, 0.25, 0.55)
        glBegin(GL_LINES)
        for i in range(6):
            a = rot + math.tau * i / 6
            glVertex2f(cx + math.cos(a) * inner, cy + math.sin(a) * inner)
            glVertex2f(cx + math.cos(a) * outer, cy + math.sin(a) * outer)
        glEnd()

    def _spawn_charge_spark(self) -> None:
        """外周をぐるっと回りつつ中央に吸い込まれる spark を 1 個生成。"""
        angle = random.random() * math.tau
        r_init = self._magic_size * (1.4 + random.random() * 0.6)
        # 接線方向 + 中心向き成分
        tangent = 0.006
        inward = 0.010
        vx = -math.sin(angle) * tangent - math.cos(angle) * inward
        vy = math.cos(angle) * tangent - math.sin(angle) * inward
        # やや彩度違いの暖色
        r = 1.0
        g = 0.55 + random.random() * 0.25
        b = 0.10 + random.random() * 0.15
        self._particles.append([
            self._magic_x + math.cos(angle) * r_init,
            self._magic_y + math.sin(angle) * r_init,
            vx, vy,
            20.0, 20.0, 0.010,
            r, g, b,
        ])

    # --- 飛翔中エフェクト ---------------------------------------------------

    def _draw_flight_trail(self, vw: int, vh: int) -> None:
        """飛翔軌跡：直近位置を alpha フェードするラインで結ぶ。
        末端（＝古い側）ほど暗く細くする効果は太い線 + 加算 blend で見せる。
        """
        n = len(self._flight_trail)
        if n < 2:
            return
        glLineWidth(6.0)
        glBegin(GL_LINE_STRIP)
        for i, (x, y) in enumerate(self._flight_trail):
            t = i / max(1, n - 1)   # 0 = 古い, 1 = 最新
            alpha = t * 0.7
            glColor4f(1.0, 0.55 + t * 0.15, 0.15, alpha)
            glVertex2f(x * vw, y * vh)
        glEnd()

    def _spawn_flight_spark(self) -> None:
        """飛翔中に尾を引く spark を生成。速度は本体と逆向きに小さくブレ。"""
        jitter = 0.006
        speed_back = -0.30
        vx = self._magic_vx * speed_back + (random.random() - 0.5) * jitter
        vy = self._magic_vy * speed_back + (random.random() - 0.5) * jitter
        self._particles.append([
            self._magic_x + (random.random() - 0.5) * 0.008,
            self._magic_y + (random.random() - 0.5) * 0.008,
            vx, vy,
            14.0, 14.0, 0.012,
            1.0, 0.55 + random.random() * 0.25, 0.12,
        ])

    # --- 爆発エフェクト -----------------------------------------------------

    def _spawn_explosion_burst(self) -> None:
        """爆発フレーム 0：放射状にパーティクルを撒く。"""
        n = self._MAGIC_EXPLOSION_PARTICLES
        for i in range(n):
            a = math.tau * i / n + (random.random() - 0.5) * 0.15
            speed = 0.020 + random.random() * 0.025
            # 3 割は白系（＝白熱コア風）、残りは橙〜赤
            if random.random() < 0.30:
                r, g, b = 1.0, 0.95, 0.75
            else:
                r, g, b = 1.0, 0.40 + random.random() * 0.30, 0.10
            self._particles.append([
                self._magic_x, self._magic_y,
                math.cos(a) * speed, math.sin(a) * speed,
                22.0 + random.random() * 6.0, 26.0, 0.016,
                r, g, b,
            ])

    def _draw_shockwave(self, vw: int, vh: int) -> None:
        """爆発中：時間で拡大する光の輪。α は 1→0 でフェード。"""
        short = min(vw, vh)
        t = self._magic_frames / self._MAGIC_EXPLODE_FRAMES
        if t > 1.0:
            return
        radius = self._MAGIC_SHOCKWAVE_RADIUS * t
        alpha = max(0.0, 1.0 - t) * 0.7
        cx = self._magic_x * vw
        cy = self._magic_y * vh
        segments = 48
        # 外リング
        glLineWidth(4.0)
        glColor4f(1.0, 0.85, 0.4, alpha)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            a = math.tau * i / segments
            glVertex2f(cx + math.cos(a) * radius * short,
                       cy + math.sin(a) * radius * short)
        glEnd()
        # 内側にもう一輪（少し遅れて広がる感じを出すために t を後ろにずらす）
        t2 = max(0.0, (self._magic_frames - 4) / self._MAGIC_EXPLODE_FRAMES)
        if t2 > 0.0 and t2 < 1.0:
            radius2 = self._MAGIC_SHOCKWAVE_RADIUS * 0.7 * t2
            alpha2 = max(0.0, 1.0 - t2) * 0.5
            glLineWidth(2.0)
            glColor4f(1.0, 0.95, 0.6, alpha2)
            glBegin(GL_LINE_LOOP)
            for i in range(segments):
                a = math.tau * i / segments
                glVertex2f(cx + math.cos(a) * radius2 * short,
                           cy + math.sin(a) * radius2 * short)
            glEnd()

    def _draw_explosion_flash(self, vw: int, vh: int) -> None:
        """爆発中央の白熱フラッシュ。最初の数フレームで一気に出て急速フェード。"""
        short = min(vw, vh)
        # 序盤ほど明るい：0 で最大、6 フレームで消える
        f = self._magic_frames
        flash_life = 8.0
        if f > flash_life:
            return
        t = f / flash_life
        alpha = (1.0 - t) ** 2
        # フラッシュはコアより大きい
        pt_outer = max(20.0, self._magic_size * short * 4.5 * (1.0 + t * 1.5))
        pt_inner = pt_outer * 0.55
        cx = self._magic_x * vw
        cy = self._magic_y * vh
        glPointSize(pt_outer)
        glColor4f(1.0, 0.7, 0.3, alpha * 0.7)
        glBegin(GL_POINTS)
        glVertex2f(cx, cy)
        glEnd()
        glPointSize(pt_inner)
        glColor4f(1.0, 0.98, 0.85, alpha)
        glBegin(GL_POINTS)
        glVertex2f(cx, cy)
        glEnd()

    # --- パーティクル -------------------------------------------------------

    def _update_particles(self) -> None:
        """全パーティクルを 1 フレーム進める。寿命切れは削除。"""
        if not self._particles:
            return
        survivors: list[list[float]] = []
        for p in self._particles:
            p[0] += p[2]
            p[1] += p[3]
            # ちょい重力・空気抵抗（飛翔粒子っぽくする）
            p[3] += 0.0006
            p[2] *= 0.965
            p[3] *= 0.965
            p[4] -= 1.0
            if p[4] > 0.0:
                survivors.append(p)
        self._particles = survivors

    def _draw_particles(self, vw: int, vh: int) -> None:
        """全パーティクル描画。1 個ずつ glPointSize/glColor を切替（数十個なら OK）。"""
        if not self._particles:
            return
        short = min(vw, vh)
        for p in self._particles:
            alpha = max(0.0, p[4] / p[5])
            pt = max(2.0, p[6] * short)
            glPointSize(pt)
            glColor4f(p[7], p[8], p[9], alpha)
            glBegin(GL_POINTS)
            glVertex2f(p[0] * vw, p[1] * vh)
            glEnd()

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
            self._particles.clear()
            self._flight_trail.clear()
            logger.info(f"Mode4 サブモード: {sub}")

    def toggle_sub_mode(self) -> str:
        try:
            idx = self.SUB_MODES.index(self._sub_mode)
        except ValueError:
            idx = -1
        new_sub = self.SUB_MODES[(idx + 1) % len(self.SUB_MODES)]
        self.set_sub_mode(new_sub)
        return new_sub
