"""
gesture_detector.py
Mode4（体験モード）用のジェスチャー検出。

フレームごとの推定結果（PoseLandmarkResult のリスト）を受け取り、
発火すべきジェスチャーイベント名のリストを返す。
ヒステリシス（ON/OFF の閾値差）と発火後クールダウンで誤発火を抑制。

イベント名:
    right_arm_up   : 右手首が右肩より上がった瞬間
    left_arm_up    : 左手首が左肩より上がった瞬間
    right_step     : 右足首が左足首より下（接地側）に切り替わった瞬間
    left_step      : 左足首が右足首より下（接地側）に切り替わった瞬間
                     ※ その場足踏みでも歩行でも交互に鳴る
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
    # 歩行：足首 Y 差（右-左）が閾値以上に開いたら「その側が下」
    # 拮抗時のチャタリング防止のためヒステリシス。
    step_diff_on: float = 0.03
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
    # 現在どちらの足が下（接地側）と判定されているか: "right" / "left" / None
    foot_down_side: Optional[str] = None


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

        # 歩行：左右足首 Y の差で「今どちらが下（接地側）か」を判定して、
        # 切り替わった瞬間に right_step / left_step を発火する。
        # その場足踏みでも歩行でも交互に鳴る。
        if lms[_RA].visibility >= vis and lms[_LA].visibility >= vis:
            diff = lms[_RA].y - lms[_LA].y   # 正 = 右足首が下
            new_side: Optional[str] = None
            if diff > cfg.step_diff_on:
                new_side = "right"
            elif -diff > cfg.step_diff_on:
                new_side = "left"
            # 拮抗（|diff| <= step_diff_on）のときは切り替えなし＝連打防止
            if new_side is not None and new_side != st.foot_down_side:
                st.foot_down_side = new_side
                self._fire(events, "right_step" if new_side == "right" else "left_step")

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
