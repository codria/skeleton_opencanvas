"""
mode4_gesture.py
モード4：ジェスチャー体験モード。
背景にカメラ映像（Mode1 同等）、その上に骨格線オーバーレイ、
サブモード（楽器 / 魔法）でジェスチャー連動の効果音・エフェクトを出す。

サブモード:
    instrument : 右腕/左腕/足踏み/しゃがみで別々の音を鳴らす
    magic      : 腕の上げ方で 3 系統の魔法を分岐（両腕=火 / 右腕=吹雪 / 左腕=雷）
                 ・両腕を上げる → 火球チャージ、腕を振ってその方向に発射→着弾爆発
                 ・右腕だけ上げる → 吹雪（腕を上げている間、手先方向へ氷を噴射）
                 ・左腕だけ上げる → 雷（全画面フラッシュ＋稲妻が落ちる一発）
                 片腕→もう片腕を待つ猶予（grace）で「両腕/片腕」を判定するので、
                 同時に上げられなくても両腕魔法を出せる。
"""

from __future__ import annotations
import logging
import math
import random
from OpenGL.GL import (
    glClearColor, glClear,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glEnable, glDisable, glColor3f, glColor4f, glLineWidth, glPointSize,
    glBegin, glEnd, glVertex2f, glBlendFunc, glHint, glGetFloatv,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION, GL_MODELVIEW,
    GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, GL_POINTS, GL_QUADS,
    GL_TRIANGLE_FAN,
    GL_DEPTH_TEST, GL_LIGHTING, GL_BLEND,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
    GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT,
    GL_POINT_SMOOTH, GL_POINT_SMOOTH_HINT,
    GL_ALIASED_POINT_SIZE_RANGE, GL_SMOOTH_POINT_SIZE_RANGE,
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

    # 魔法モードの上位フェーズ（どの系統の魔法を発動中か）
    _PH_IDLE = "idle"          # 待機（腕上げ待ち）
    _PH_RESOLVE = "resolve"    # 片腕検知→もう片腕を待つ猶予中（火/片腕を確定させる）
    _PH_FIRE = "fire"          # 火球（両腕）
    _PH_ICE = "ice"            # 吹雪（右腕のみ・持続）
    _PH_THUNDER = "thunder"    # 雷（左腕のみ・一発）
    _PH_COOLDOWN = "cooldown"  # 発動後、腕が下りるまで再発火を抑止

    # 火球の内部サブ状態（_PH_FIRE の中でのみ使う）
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

    # --- 3 系統分岐（腕の上げ方）---
    # 片腕を検知してから、もう片腕が上がるかを待つ猶予。両腕同時は難しいので
    # この猶予内に 2 本目が上がれば「両腕（火）」、上がらなければ「片腕」に確定。
    _MAGIC_ARM_GRACE_FRAMES: int = 8             # ~250ms @30fps
    # 火球 charging 中、両腕を下ろし続けたら不発にして cooldown に戻す猶予
    _FIRE_ABANDON_FRAMES: int = 20               # ~0.7s
    # 吹雪（右腕のみ・持続）
    _ICE_SPAWN_PER_FRAME: int = 4                # 1 フレームに撒く氷粒子数
    _ICE_SPEED: float = 0.035                    # 噴射初速（画像座標/frame）
    _ICE_SPREAD: float = 0.5                     # 噴射の広がり半角（rad）
    _ICE_LIFE: float = 16.0                      # 氷粒子の寿命（frame）
    _ICE_END_TOLERANCE: int = 4                  # 右腕を下ろして何フレームで終了とみなすか
    # 雷（左腕のみ・一発）。溜め（構え音＋帯電）→ 着弾（フラッシュ＋稲妻）の 2 段。
    # 溜めは実演テンポ優先で短め（~0.5s）。長いと説明の間が空いてしまう。
    _THUNDER_CHARGE_FRAMES: int = 15             # 着弾前の溜めフレーム（~0.5s @30fps）
    # 構え音（雷魔法3）は尺が長いので、溜め終了に音が消え切るよう手前でフェード開始。
    _THUNDER_CHARGE_FADE_FRAMES: int = 6         # 溜め終了の何フレーム手前からフェードするか
    _THUNDER_FRAMES: int = 22                    # 着弾後の表示総フレーム
    _THUNDER_FLASH_FRAMES: int = 10              # 全画面フラッシュの減衰フレーム（着弾起点）
    _THUNDER_BOLT_FLICKER: int = 12             # 稲妻を再生成してちらつかせる期間
    _THUNDER_SEGMENTS: int = 16                  # 稲妻のジグザグ分割数
    _THUNDER_JITTER: float = 0.05                # 稲妻の横ブレ幅（画像座標比）

    def __init__(self, config) -> None:
        self._config = config
        self._sub_mode: str = "instrument"

        # 描画・音・ジェスチャー検出器
        self._camera_overlay = CameraOverlay()
        self._sound_bank = SoundBank("assets/sounds")
        self._detector = GestureDetector()

        # 魔法モード上位フェーズ
        self._magic_phase = self._PH_IDLE
        self._resolve_frames: int = 0   # RESOLVE 猶予の経過フレーム

        # 火球（_PH_FIRE）状態
        self._magic_state = self._MAGIC_IDLE
        self._magic_x: float = 0.5    # 画像座標 0-1
        self._magic_y: float = 0.5
        self._magic_vx: float = 0.0   # 単位: 画像座標/フレーム
        self._magic_vy: float = 0.0
        self._magic_size: float = 0.02   # 半径（画像座標比）
        self._magic_frames: int = 0
        self._fire_abandon: int = 0   # charging 中に両腕を下ろし続けたフレーム数
        # 発射時の飛翔方向計算のため、charging 中の両手中央位置履歴
        # 直近 _MAGIC_HAND_HISTORY フレーム分保持し、発射時に位置差ベクトルを速度化
        self._magic_hand_history: list[tuple[float, float]] = []

        # 吹雪（_PH_ICE）状態
        self._ice_frames: int = 0
        self._ice_lost: int = 0       # 右腕を下ろして経過したフレーム数
        self._ice_hand: tuple[float, float] = (0.5, 0.5)

        # 雷（_PH_THUNDER）状態
        self._thunder_frames: int = 0
        self._thunder_x: float = 0.5  # 着弾（＝左手）位置
        self._thunder_y: float = 0.5
        self._thunder_bolt: list[tuple[float, float]] = []  # 稲妻の折れ線（画像座標）

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
        # ドライバの点サイズ上限を記録（円形グローが点に潰れる不具合の切り分け用）。
        # 大きいグローは三角形ファンで描くので描画自体は上限に依存しないが、
        # 環境調査のためログに残す。
        try:
            aliased = glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE)
            smooth = glGetFloatv(GL_SMOOTH_POINT_SIZE_RANGE)
            logger.info(f"GL point-size range: aliased={tuple(aliased)} "
                        f"smooth={tuple(smooth)}")
        except Exception as e:
            logger.debug(f"点サイズ範囲の取得に失敗: {e}")
        logger.info("Mode4Gesture 初期化完了")

    def _reset_magic(self) -> None:
        """魔法モードの全フェーズ・エフェクト状態を初期化する。"""
        self._magic_phase = self._PH_IDLE
        self._magic_state = self._MAGIC_IDLE
        self._resolve_frames = 0
        self._fire_abandon = 0
        self._ice_frames = 0
        self._ice_lost = 0
        self._thunder_frames = 0
        self._magic_hand_history.clear()
        self._thunder_bolt.clear()
        self._particles.clear()
        self._flight_trail.clear()

    def on_mode_enter(self) -> None:
        # 状態リセット（前回の魔法状態などを持ち越さない）
        self._detector.reset()
        self._reset_magic()
        self._prev_right_arm_up = False
        self._prev_left_arm_up = False
        logger.info(f"モード4（体験）開始 サブモード={self._sub_mode}")

    def on_mode_exit(self) -> None:
        # 継続再生中の右腕ドラムロール音は短くフェードして止める
        # （左腕シンバルは短音なので放置で自然に減衰）
        self._sound_bank.fadeout("right_arm_up", ms=150)
        # 魔法の継続音（火チャージ・氷構え）も念のためフェード
        self._sound_bank.fadeout("magic_fire_charge", ms=150)
        self._sound_bank.fadeout("magic_ice_charge", ms=150)
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
        """魔法モードの上位ディスパッチャ。腕の上げ方で 3 系統に分岐する。
        フェーズ遷移：
            IDLE → RESOLVE →（両腕:FIRE / 右腕:ICE / 左腕:THUNDER）→ COOLDOWN → IDLE
        RESOLVE は「片腕を検知してからもう片腕を待つ猶予」で、両腕同時に上げなくても
        火球（両腕魔法）が出せるようにするための緩衝。COOLDOWN は発動後に腕が下りるまで
        再発火を抑止する。
        """
        info = self._extract_magic_pose(results)

        phase = self._magic_phase
        if phase == self._PH_IDLE:
            self._magic_idle(info)
        elif phase == self._PH_RESOLVE:
            self._magic_resolve(info)
        elif phase == self._PH_FIRE:
            self._update_fire(info)
        elif phase == self._PH_ICE:
            self._update_ice(info)
        elif phase == self._PH_THUNDER:
            self._update_thunder(info)
        elif phase == self._PH_COOLDOWN:
            self._magic_cooldown(info)

        # パーティクルは全フェーズ共通で進める（発動終了後の余韻も描くため）
        self._update_particles()

        # ---- 描画 ----
        active = self._magic_phase in (
            self._PH_RESOLVE, self._PH_FIRE, self._PH_ICE, self._PH_THUNDER)
        if not active and not self._particles and not self._flight_trail:
            return
        self._setup_2d(vx, vy, vw, vh)
        # 雷は全画面フラッシュ＋稲妻を最初に敷く
        if self._magic_phase == self._PH_THUNDER:
            self._draw_thunder(vw, vh)
        # 火の飛翔尾
        if self._flight_trail:
            self._draw_flight_trail(vw, vh)
        # 系統別コア
        if self._magic_phase == self._PH_FIRE:
            if self._magic_state == self._MAGIC_CHARGING:
                self._draw_fireball_core(vw, vh, pulsing=True)
                self._draw_charge_rays(vw, vh)
            elif self._magic_state == self._MAGIC_FLYING:
                self._draw_fireball_core(vw, vh, pulsing=False)
            elif self._magic_state == self._MAGIC_EXPLODE:
                self._draw_explosion_flash(vw, vh)
                self._draw_shockwave(vw, vh)
        elif self._magic_phase == self._PH_ICE:
            self._draw_ice_core(vw, vh)
        # パーティクル（最後に上乗せ）
        self._draw_particles(vw, vh)

    # --- ポーズ抽出 & 腕判定 ------------------------------------------------

    def _extract_magic_pose(self, results):
        """魔法判定に必要な情報を dict で返す。必要ランドマークが揃わなければ None。
        戻り値キー：
            r_up / l_up   : 右腕 / 左腕が「肩〜腰の RATIO 下ライン」より上か
            rc / lc       : 右手首 / 左手首 (x, y)（不可視なら None）
            both_c / span : 両手中央 (x, y) と両手間距離（両手可視時のみ、他は None/0）
            rs            : 右肩 (x, y)（吹雪の噴射方向用）
        """
        if not results:
            return None
        lms = results[0].landmarks
        LS, RS = 11, 12
        LW, RW = 15, 16
        LH, RH = 23, 24
        VIS = 0.5
        # 肩・腰が両方見えないと閾値ラインが引けない
        if not (lms[LS].visibility >= VIS and lms[RS].visibility >= VIS and
                lms[LH].visibility >= VIS and lms[RH].visibility >= VIS):
            return None
        shoulder_y = (lms[LS].y + lms[RS].y) / 2
        hip_y = (lms[LH].y + lms[RH].y) / 2
        thresh_y = shoulder_y + (hip_y - shoulder_y) * self._MAGIC_CHARGE_ARM_RATIO

        rw_vis = lms[RW].visibility >= VIS
        lw_vis = lms[LW].visibility >= VIS
        r_up = rw_vis and lms[RW].y < thresh_y
        l_up = lw_vis and lms[LW].y < thresh_y
        rc = (lms[RW].x, lms[RW].y) if rw_vis else None
        lc = (lms[LW].x, lms[LW].y) if lw_vis else None
        both_c = None
        span = 0.0
        if rw_vis and lw_vis:
            both_c = ((lms[LW].x + lms[RW].x) / 2, (lms[LW].y + lms[RW].y) / 2)
            dx = lms[LW].x - lms[RW].x
            dy = lms[LW].y - lms[RW].y
            span = (dx * dx + dy * dy) ** 0.5
        return {
            "r_up": r_up, "l_up": l_up,
            "rc": rc, "lc": lc, "both_c": both_c, "span": span,
            "rs": (lms[RS].x, lms[RS].y),
        }

    def _magic_idle(self, info) -> None:
        """待機。どちらかの腕が上がったら RESOLVE へ。"""
        if info is None:
            return
        if info["r_up"] or info["l_up"]:
            self._magic_phase = self._PH_RESOLVE
            self._resolve_frames = 0

    def _magic_resolve(self, info) -> None:
        """片腕検知後の猶予。両腕なら火、猶予切れで片腕確定。
        トラッキングが切れたフレームは判定を進めず保留（誤確定を避ける）。
        """
        if info is None:
            return
        self._resolve_frames += 1
        r, l = info["r_up"], info["l_up"]
        if r and l:
            self._start_fire(info)
        elif not r and not l:
            # 猶予中に両腕とも下りた → 中断
            self._magic_phase = self._PH_IDLE
        elif self._resolve_frames >= self._MAGIC_ARM_GRACE_FRAMES:
            if r:
                self._start_ice(info)
            else:
                self._start_thunder(info)
        else:
            # 待機中：上がっている手元に「溜め」の火花を少しだけ出す
            hand = info["rc"] if r else info["lc"]
            if hand is not None and random.random() < 0.5:
                self._spawn_gather_spark(hand[0], hand[1])

    def _magic_cooldown(self, info) -> None:
        """発動後、腕が下りる（またはトラッキング喪失）まで再発火を抑止。"""
        if info is None or (not info["r_up"] and not info["l_up"]):
            self._magic_phase = self._PH_IDLE

    # --- 火球（両腕）--------------------------------------------------------

    def _start_fire(self, info) -> None:
        cx, cy = info["both_c"]
        self._magic_phase = self._PH_FIRE
        self._magic_state = self._MAGIC_CHARGING
        self._magic_x, self._magic_y = cx, cy
        self._magic_size = self._MAGIC_FIREBALL_MIN
        self._magic_frames = 0
        self._fire_abandon = 0
        # 腕を上げてきた履歴は捨て、charging 中の動きだけをスイング判定対象に
        self._magic_hand_history.clear()
        self._magic_hand_history.append((cx, cy))
        self._sound_bank.play("magic_fire_charge")
        logger.info("[Mode4/魔法] 火：charging 開始")

    def _update_fire(self, info) -> None:
        if self._magic_state == self._MAGIC_CHARGING:
            self._magic_frames += 1
            if info is not None and info["both_c"] is not None:
                cx, cy = info["both_c"]
                self._magic_x, self._magic_y = cx, cy
                self._magic_hand_history.append((cx, cy))
                if len(self._magic_hand_history) > self._MAGIC_HAND_HISTORY:
                    self._magic_hand_history.pop(0)
                target = max(
                    self._MAGIC_FIREBALL_MIN,
                    min(self._MAGIC_FIREBALL_MAX,
                        info["span"] * self._MAGIC_FIREBALL_SCALE),
                )
                self._magic_size += (target - self._magic_size) * self._MAGIC_FIREBALL_LERP
            # 両腕を下ろし続けたら不発（片腕が上がっていれば継続）
            arms_down = (info is None) or (not info["r_up"] and not info["l_up"])
            self._fire_abandon = self._fire_abandon + 1 if arms_down else 0
            if random.random() < self._MAGIC_CHARGE_SPARK_PROB:
                self._spawn_charge_spark()
            # 猶予フレーム後にスイング判定
            launch_vel = None
            if self._magic_frames >= self._MAGIC_CHARGE_MIN_FRAMES:
                launch_vel = self._check_swing_launch()
            if launch_vel is not None:
                self._magic_state = self._MAGIC_FLYING
                self._magic_frames = 0
                self._magic_vx, self._magic_vy = launch_vel
                self._flight_trail.clear()
                logger.info(
                    f"[Mode4/魔法] 火：発射 size={self._magic_size:.3f} "
                    f"v=({self._magic_vx:+.3f}, {self._magic_vy:+.3f})"
                )
            elif self._fire_abandon > self._FIRE_ABANDON_FRAMES:
                # 振らずに腕を下ろしたまま → 不発にして cooldown へ
                self._sound_bank.fadeout("magic_fire_charge", ms=200)
                self._magic_state = self._MAGIC_IDLE
                self._magic_phase = self._PH_COOLDOWN
                logger.info("[Mode4/魔法] 火：不発（スイングなし）")

        elif self._magic_state == self._MAGIC_FLYING:
            self._magic_x += self._magic_vx
            self._magic_y += self._magic_vy
            self._magic_frames += 1
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
                self._sound_bank.play("magic_fire_hit")
                logger.info("[Mode4/魔法] 火：爆発")

        elif self._magic_state == self._MAGIC_EXPLODE:
            self._magic_frames += 1
            if self._magic_frames > self._MAGIC_EXPLODE_FRAMES:
                self._magic_state = self._MAGIC_IDLE
                self._magic_phase = self._PH_COOLDOWN
                self._flight_trail.clear()

    # --- 吹雪（右腕のみ・持続）----------------------------------------------

    def _start_ice(self, info) -> None:
        self._magic_phase = self._PH_ICE
        self._ice_frames = 0
        self._ice_lost = 0
        if info["rc"] is not None:
            self._ice_hand = info["rc"]
        self._sound_bank.play("magic_ice_charge")
        logger.info("[Mode4/魔法] 吹雪：開始")

    def _update_ice(self, info) -> None:
        self._ice_frames += 1
        # 右腕が上がっていれば噴射継続、下りて（or 喪失して）一定フレームで終了
        if info is not None and info["r_up"] and info["rc"] is not None:
            self._ice_lost = 0
            hx, hy = info["rc"]
            self._ice_hand = (hx, hy)
            sx, sy = info["rs"]
            self._spawn_ice_burst(hx, hy, hx - sx, hy - sy)
        else:
            self._ice_lost += 1
            if self._ice_lost > self._ICE_END_TOLERANCE:
                # 終了＝「凍結」：着弾音＋手元に氷の弾ける粒を撒いて締める
                self._sound_bank.fadeout("magic_ice_charge", ms=120)
                self._sound_bank.play("magic_ice_hit")
                self._spawn_ice_freeze_burst(*self._ice_hand)
                self._magic_phase = self._PH_COOLDOWN
                logger.info("[Mode4/魔法] 吹雪：凍結（終了）")

    # --- 雷（左腕のみ・一発）------------------------------------------------

    def _start_thunder(self, info) -> None:
        self._magic_phase = self._PH_THUNDER
        self._thunder_frames = 0
        tx, ty = info["lc"] if info["lc"] is not None else (0.5, 0.3)
        self._thunder_x, self._thunder_y = tx, ty
        self._thunder_bolt.clear()   # 溜め中は稲妻なし、着弾時に生成
        self._sound_bank.play("magic_thunder_charge")
        logger.info("[Mode4/魔法] 雷：溜め開始")

    def _update_thunder(self, info) -> None:
        self._thunder_frames += 1
        cf = self._THUNDER_CHARGE_FRAMES
        # 追尾：溜め中は左手位置に着弾点を追従させる（下ろしても最後の位置で撃つ）
        if (self._thunder_frames <= cf and info is not None and
                info["lc"] is not None):
            self._thunder_x, self._thunder_y = info["lc"]

        if self._thunder_frames < cf:
            # 溜め：手元に帯電スパークを集める
            if random.random() < 0.9:
                self._spawn_gather_spark(self._thunder_x, self._thunder_y)
            # 構え音は尺が長いので、溜め終了時刻に消え切るよう手前からフェード開始
            if self._thunder_frames == cf - self._THUNDER_CHARGE_FADE_FRAMES:
                self._sound_bank.fadeout(
                    "magic_thunder_charge",
                    ms=int(self._THUNDER_CHARGE_FADE_FRAMES / 30.0 * 1000),
                )
        elif self._thunder_frames == cf:
            # 着弾：稲妻生成＋全画面フラッシュ（描画側）＋着弾音＋デブリ
            self._gen_thunder_bolt()
            self._sound_bank.play("magic_thunder_hit")
            tx, ty = self._thunder_x, self._thunder_y
            for i in range(16):
                a = math.tau * i / 16 + (random.random() - 0.5) * 0.3
                spd = 0.010 + random.random() * 0.020
                self._particles.append([
                    tx, ty,
                    math.cos(a) * spd, math.sin(a) * spd,
                    16.0 + random.random() * 6.0, 22.0, 0.010,
                    0.7, 0.85, 1.0,
                ])
            logger.info("[Mode4/魔法] 雷：着弾")
        else:
            # 着弾後：稲妻をちらつかせる
            strike_f = self._thunder_frames - cf
            if strike_f <= self._THUNDER_BOLT_FLICKER and strike_f % 2 == 0:
                self._gen_thunder_bolt()
            if strike_f > self._THUNDER_FRAMES:
                self._magic_phase = self._PH_COOLDOWN
                self._thunder_bolt.clear()

    def _gen_thunder_bolt(self) -> None:
        """画面上端から着弾点まで、横ブレしながら落ちるジグザグ折れ線を生成。
        着弾点に近づくほどブレを小さくして狙いを収束させる。"""
        x_top = self._thunder_x + (random.random() - 0.5) * 0.06
        tx, ty = self._thunder_x, self._thunder_y
        n = self._THUNDER_SEGMENTS
        pts: list[tuple[float, float]] = []
        for i in range(n + 1):
            t = i / n
            base_x = x_top + (tx - x_top) * t
            jitter = (random.random() - 0.5) * self._THUNDER_JITTER * (1.0 - t * 0.7)
            pts.append((base_x + jitter, ty * t))
        pts[0] = (x_top, 0.0)
        pts[-1] = (tx, ty)
        self._thunder_bolt = pts

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

    # --- 円形グロー（ドライバの点サイズ上限に依存しない実装）----------------

    def _draw_glow(self, cx: float, cy: float, radius: float,
                   r: float, g: float, b: float, alpha: float,
                   segments: int = 24) -> None:
        """中心が (r,g,b,alpha)・外周が α=0 の放射グラデ円盤を三角形ファンで描く。
        巨大な glPointSize は多くの GPU で最大点サイズにクランプされ「ただの点」に
        潰れるため、大きい円形エフェクトはこの円盤で描く（加算合成前提）。
        """
        if radius <= 0.5:
            return
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(r, g, b, alpha)
        glVertex2f(cx, cy)
        glColor4f(r, g, b, 0.0)
        for i in range(segments + 1):
            a = math.tau * i / segments
            glVertex2f(cx + math.cos(a) * radius, cy + math.sin(a) * radius)
        glEnd()

    # --- 火球コア（レイヤー glow）-------------------------------------------

    def _draw_fireball_core(self, vw: int, vh: int, pulsing: bool) -> None:
        """火球本体：5 層の同心円盤グローで「にじむ光球」を作る。
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
            radius = max(2.0, size * short * scale)
            self._draw_glow(cx, cy, radius, r, g, b, alpha)

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
        # 序盤ほど明るい：0 で最大、8 フレームで消える
        f = self._magic_frames
        flash_life = 8.0
        if f > flash_life:
            return
        t = f / flash_life
        alpha = (1.0 - t) ** 2
        # フラッシュはコアより大きい（半径 = 旧点サイズ/2）
        r_outer = max(10.0, self._magic_size * short * 2.25 * (1.0 + t * 1.5))
        r_inner = r_outer * 0.55
        cx = self._magic_x * vw
        cy = self._magic_y * vh
        self._draw_glow(cx, cy, r_outer, 1.0, 0.7, 0.3, alpha * 0.7)
        self._draw_glow(cx, cy, r_inner, 1.0, 0.98, 0.85, alpha)

    # --- 溜め火花（RESOLVE 中の手元フィードバック）--------------------------

    def _spawn_gather_spark(self, x: float, y: float) -> None:
        """手元の周囲から中心へ吸い込まれる淡い白火花。まだ系統未確定なので中立色。"""
        angle = random.random() * math.tau
        r_init = 0.05 + random.random() * 0.03
        inward = 0.012
        self._particles.append([
            x + math.cos(angle) * r_init,
            y + math.sin(angle) * r_init,
            -math.cos(angle) * inward, -math.sin(angle) * inward,
            12.0, 12.0, 0.007,
            0.85, 0.9, 1.0,
        ])

    # --- 吹雪パーティクル ---------------------------------------------------

    def _spawn_ice_burst(self, hx: float, hy: float, dirx: float, diry: float) -> None:
        """右手先から (dirx, diry) 方向へ、広がりを持たせて氷片を噴射する。"""
        mag = (dirx * dirx + diry * diry) ** 0.5
        if mag < 0.05:
            # 腕がほぼ肩位置＝方向が不定 → 体の外側へ水平やや上に噴射
            base = 0.0 if dirx >= 0 else math.pi
        else:
            base = math.atan2(diry, dirx)
        for _ in range(self._ICE_SPAWN_PER_FRAME):
            ang = base + (random.random() - 0.5) * 2.0 * self._ICE_SPREAD
            spd = self._ICE_SPEED * (0.6 + random.random() * 0.5)
            # 白〜シアンの寒色。少しだけ寿命・サイズをばらす
            g = 0.85 + random.random() * 0.15
            self._particles.append([
                hx, hy,
                math.cos(ang) * spd, math.sin(ang) * spd,
                self._ICE_LIFE * (0.7 + random.random() * 0.6),
                self._ICE_LIFE, 0.006 + random.random() * 0.004,
                0.65 + random.random() * 0.15, g, 1.0,
            ])

    def _spawn_ice_freeze_burst(self, hx: float, hy: float) -> None:
        """吹雪終了（凍結）時に手元で弾ける氷粒。放射状に軽く撒いて締める。"""
        for i in range(20):
            a = math.tau * i / 20 + (random.random() - 0.5) * 0.25
            spd = 0.008 + random.random() * 0.016
            g = 0.88 + random.random() * 0.12
            self._particles.append([
                hx, hy,
                math.cos(a) * spd, math.sin(a) * spd,
                18.0 + random.random() * 6.0, 24.0, 0.008,
                0.7, g, 1.0,
            ])

    def _draw_ice_core(self, vw: int, vh: int) -> None:
        """右手先の寒色グロー（噴射口）。3 層の淡いシアン円盤。"""
        short = min(vw, vh)
        cx = self._ice_hand[0] * vw
        cy = self._ice_hand[1] * vh
        layers = (
            (0.070, 0.18, 0.55, 0.80, 1.0),
            (0.045, 0.30, 0.70, 0.90, 1.0),
            (0.022, 0.75, 0.90, 0.98, 1.0),
        )
        for rad, alpha, r, g, b in layers:
            self._draw_glow(cx, cy, max(2.0, rad * short), r, g, b, alpha)

    # --- 雷エフェクト -------------------------------------------------------

    def _draw_thunder(self, vw: int, vh: int) -> None:
        """雷の描画。溜め中（strike_f<0）は手元に帯電グロー、
        着弾後（strike_f>=0）は全画面フラッシュ＋稲妻。加算合成で白系が強く光る。"""
        short = min(vw, vh)
        strike_f = self._thunder_frames - self._THUNDER_CHARGE_FRAMES
        cx = self._thunder_x * vw
        cy = self._thunder_y * vh

        if strike_f < 0:
            # 溜め：着弾に向けて手元の帯電球が育つ（青白）
            t = (self._thunder_frames + 1) / max(1, self._THUNDER_CHARGE_FRAMES)
            for scale, alpha, r, g, b in (
                (0.10, 0.10 * t, 0.5, 0.7, 1.0),
                (0.06, 0.25 * t, 0.7, 0.85, 1.0),
                (0.03, 0.60 * t, 0.9, 0.97, 1.0),
            ):
                self._draw_glow(cx, cy, max(2.0, scale * short), r, g, b, alpha)
            return

        # 着弾後：全画面フラッシュ（減衰）＋稲妻再生成タイミングで軽く増幅
        fa = max(0.0, 1.0 - strike_f / self._THUNDER_FLASH_FRAMES)
        flick = 0.15 if (strike_f <= self._THUNDER_BOLT_FLICKER and
                         strike_f % 2 == 0) else 0.0
        alpha = min(0.9, fa * 0.8 + flick)
        if alpha > 0.0:
            glColor4f(0.82, 0.9, 1.0, alpha)   # やや青白
            glBegin(GL_QUADS)
            glVertex2f(0.0, 0.0)
            glVertex2f(vw, 0.0)
            glVertex2f(vw, vh)
            glVertex2f(0.0, vh)
            glEnd()
        # 稲妻本体（外グロー＋白コアの 2 パス）
        if self._thunder_bolt and strike_f <= self._THUNDER_BOLT_FLICKER + 4:
            bolt_alpha = max(0.0, 1.0 - strike_f / (self._THUNDER_BOLT_FLICKER + 4))

            def _strip():
                glBegin(GL_LINE_STRIP)
                for x, y in self._thunder_bolt:
                    glVertex2f(x * vw, y * vh)
                glEnd()

            glLineWidth(9.0)
            glColor4f(0.55, 0.72, 1.0, bolt_alpha * 0.5)
            _strip()
            glLineWidth(3.5)
            glColor4f(1.0, 1.0, 1.0, bolt_alpha)
            _strip()
            # 着弾点フラッシュ（円盤グロー）
            self._draw_glow(cx, cy, max(10.0, short * 0.04),
                            0.9, 0.95, 1.0, bolt_alpha)

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
            self._reset_magic()
            logger.info(f"Mode4 サブモード: {sub}")

    def toggle_sub_mode(self) -> str:
        try:
            idx = self.SUB_MODES.index(self._sub_mode)
        except ValueError:
            idx = -1
        new_sub = self.SUB_MODES[(idx + 1) % len(self.SUB_MODES)]
        self.set_sub_mode(new_sub)
        return new_sub
