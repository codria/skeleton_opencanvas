"""
gesture_detector.py
Mode4（体験モード）用のジェスチャー検出。

フレームごとの推定結果（PoseLandmarkResult のリスト）を受け取り、
発火すべきジェスチャーイベント名のリストを返す。
ヒステリシス（ON/OFF の閾値差）と発火後クールダウンで誤発火を抑制。

イベント名:
    right_arm_up   : 右手首が右肩より上がった瞬間
    left_arm_up    : 左手首が左肩より上がった瞬間
    right_step     : 右足首 Y が急に下がった瞬間（右足の着地）
    left_step      : 左足首 Y が急に下がった瞬間（左足の着地）
                     ※ 各足独立に判定。片足だけの足踏みでも連続発火する。
    crouch         : 静止しゃがみ姿勢（腰と足首・腰と肩の縦距離が両方小さい）

状態プロパティ（魔法モード側で参照）:
    right_arm_up   : 右手首が右肩より上に居るか（現在フレーム）
    left_arm_up    : 左手首が左肩より上に居るか
    both_arms_up   : 両手首が両肩より上に居るか
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# MediaPipe Pose Landmarker のインデックス
_NOSE = 0
_LS, _RS = 11, 12   # 肩
_LW, _RW = 15, 16   # 手首
_LH, _RH = 23, 24   # 腰
_LA, _RA = 27, 28   # 足首


@dataclass
class GestureConfig:
    """判定パラメータ。誤発火を抑制するためヒステリシスとクールダウンを使う。"""
    min_visibility: float = 0.5
    # 腕上げ：手首 Y が肩 Y より 0.03 高いと ON、肩と同じ高さまで戻ると OFF
    # （画像座標は上端 0.0 下端 1.0 なので、上に居るとは Y が小さい）
    arm_up_on: float = 0.03
    arm_up_off: float = 0.0
    # 歩行：各足首 Y が前フレームより閾値以上下がった瞬間を「その足の着地」
    # として発火する。片足だけの足踏みでも連続発火するよう独立判定。
    step_delta_threshold: float = 0.02
    # しゃがみ：腰-足首 と 腰-肩 の縦距離が両方これ未満
    crouch_hip_ankle: float = 0.28
    crouch_shoulder_hip: float = 0.20
    # 発火後クールダウン（フレーム数）。ここが大きいほど連打防止
    cooldown_frames: int = 20


@dataclass
class _State:
    right_arm_up: bool = False
    left_arm_up: bool = False
    crouching: bool = False
    cooldowns: dict = field(default_factory=dict)
    # 前フレームの足首 Y（各足独立に Δ 監視するため）
    prev_ra_y: Optional[float] = None
    prev_la_y: Optional[float] = None


class GestureDetector:
    """PoseLandmarkResult 群からジェスチャー event を検出する。"""

    def __init__(self, cfg: Optional[GestureConfig] = None) -> None:
        self._cfg = cfg or GestureConfig()
        self._st = _State()

    def reset(self) -> None:
        """モード切替時などに呼ぶ。前フレーム状態をクリア。"""
        self._st = _State()

    # --- 状態プロパティ（魔法モード側で参照）------------------------------

    @property
    def right_arm_up(self) -> bool:
        return self._st.right_arm_up

    @property
    def left_arm_up(self) -> bool:
        return self._st.left_arm_up

    @property
    def both_arms_up(self) -> bool:
        return self._st.right_arm_up and self._st.left_arm_up

    # --- 検出 --------------------------------------------------------------

    def detect(self, results) -> list[str]:
        """発火する event キーのリスト。空フレームや検出なしなら空。"""
        cfg = self._cfg
        st = self._st

        # クールダウン進行
        for k in list(st.cooldowns.keys()):
            st.cooldowns[k] -= 1
            if st.cooldowns[k] <= 0:
                del st.cooldowns[k]

        if not results:
            return []

        lms = results[0].landmarks
        vis = cfg.min_visibility
        events: list[str] = []

        # 右腕上げ（エッジ検出＋ヒステリシス）
        if lms[_RW].visibility >= vis and lms[_RS].visibility >= vis:
            wy, sy = lms[_RW].y, lms[_RS].y
            if not st.right_arm_up and wy < sy - cfg.arm_up_on:
                st.right_arm_up = True
                self._fire(events, "right_arm_up")
            elif st.right_arm_up and wy > sy - cfg.arm_up_off:
                st.right_arm_up = False

        # 左腕上げ
        if lms[_LW].visibility >= vis and lms[_LS].visibility >= vis:
            wy, sy = lms[_LW].y, lms[_LS].y
            if not st.left_arm_up and wy < sy - cfg.arm_up_on:
                st.left_arm_up = True
                self._fire(events, "left_arm_up")
            elif st.left_arm_up and wy > sy - cfg.arm_up_off:
                st.left_arm_up = False

        # 歩行：各足首 Y の Δ を独立に監視。
        # 前フレームより閾値以上下がった＝その足が着地した瞬間として発火。
        # 相対比較ではないので「右足だけ足踏み」でも right_step が連続で鳴る。
        if lms[_RA].visibility >= vis:
            ra_y = lms[_RA].y
            if st.prev_ra_y is not None:
                delta = ra_y - st.prev_ra_y   # 正 = 下方向（着地）
                if delta > cfg.step_delta_threshold:
                    self._fire(events, "right_step")
            st.prev_ra_y = ra_y
        else:
            st.prev_ra_y = None
        if lms[_LA].visibility >= vis:
            la_y = lms[_LA].y
            if st.prev_la_y is not None:
                delta = la_y - st.prev_la_y
                if delta > cfg.step_delta_threshold:
                    self._fire(events, "left_step")
            st.prev_la_y = la_y
        else:
            st.prev_la_y = None

        # しゃがみ：肩・腰・足首の縦距離が両方小さいか
        hip_y: Optional[float] = None
        if lms[_LH].visibility >= vis and lms[_RH].visibility >= vis:
            hip_y = (lms[_LH].y + lms[_RH].y) / 2
        crouch_now = False
        if (hip_y is not None and
                lms[_LA].visibility >= vis and lms[_RA].visibility >= vis and
                lms[_LS].visibility >= vis and lms[_RS].visibility >= vis):
            ankle_y = (lms[_LA].y + lms[_RA].y) / 2
            shoulder_y = (lms[_LS].y + lms[_RS].y) / 2
            dist_hip_ankle = abs(ankle_y - hip_y)
            dist_shoulder_hip = abs(hip_y - shoulder_y)
            if (dist_hip_ankle < cfg.crouch_hip_ankle and
                    dist_shoulder_hip < cfg.crouch_shoulder_hip):
                crouch_now = True
        # crouch は「入った瞬間」だけ発火（ヒステリシス的）
        if crouch_now and not st.crouching:
            self._fire(events, "crouch")
        st.crouching = crouch_now

        return events

    def _fire(self, events: list[str], key: str) -> bool:
        """クールダウン中でなければ発火して登録＋クールダウン開始。
        戻り値: 実際に発火したら True（呼び出し側で airborne 等の状態を更新する）。
        """
        if key in self._st.cooldowns:
            return False
        events.append(key)
        self._st.cooldowns[key] = self._cfg.cooldown_frames
        return True
