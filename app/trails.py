"""
trails.py
両手両足（手首・足首）の移動経路を保持する Trail Buffer。
各部位最大 MAX_POINTS 点を deque で保持し、描画時に古い点ほどフェードさせる。
"""

from __future__ import annotations
from collections import deque
from app.pose_constants import PoseLandmark


# 軌跡を残す対象ランドマーク
TRACKED_POINTS = (
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
)

# 部位別の色 (R, G, B)
TRAIL_COLORS: dict[int, tuple[float, float, float]] = {
    PoseLandmark.LEFT_WRIST:  (0.20, 0.85, 1.00),  # シアン
    PoseLandmark.RIGHT_WRIST: (1.00, 0.40, 0.80),  # マゼンタ
    PoseLandmark.LEFT_ANKLE:  (0.55, 1.00, 0.35),  # ライム
    PoseLandmark.RIGHT_ANKLE: (1.00, 0.75, 0.25),  # オレンジ
}

# 各部位の最大保持点数
MAX_POINTS = 32
# visibility 閾値（これ未満のフレームはバッファに追加しない）
MIN_VIS = 0.4


class TrailBuffer:
    """4 点の移動経路を保持するバッファ。各点は描画系座標 (tp 適用後) を入れる。"""

    def __init__(self) -> None:
        self._max_points: int = MAX_POINTS
        self._buffers: dict[int, deque[tuple[float, float, float]]] = {
            pid: deque(maxlen=self._max_points) for pid in TRACKED_POINTS
        }
        # 「直近フレームで visibility が閾値超え＆検出ありだったか」のフラグ。
        # 軌跡の線は履歴として描き続けるが、現在位置を示す点は
        # 認識失敗時に消したいので、現在の検出状態を別に保持する。
        self._currently_visible: dict[int, bool] = {
            pid: False for pid in TRACKED_POINTS
        }

    def set_max_points(self, n: int) -> None:
        """軌跡の最大保持点数を変更する（既存内容は新 maxlen で引き継ぎ）。"""
        n = max(2, int(n))
        if n == self._max_points:
            return
        self._max_points = n
        self._buffers = {
            pid: deque(buf, maxlen=n) for pid, buf in self._buffers.items()
        }

    def update(self, landmarks, tp) -> None:
        """新しいフレームのランドマークから対象 4 点を取り出して各バッファに追加する。
        tp は (x, y, z) を返す座標変換関数（Mode2/3 共通インターフェース）。
        visibility 不足のときは buffer をクリア（点も線も消える）。
        「描画するかどうか」は buffer の中身だけで決まるよう責務を一箇所に集約。
        """
        for pid in TRACKED_POINTS:
            if pid >= len(landmarks):
                self._currently_visible[pid] = False
                self._buffers[pid].clear()
                continue
            lm = landmarks[pid]
            if lm.visibility < MIN_VIS:
                self._currently_visible[pid] = False
                self._buffers[pid].clear()
                continue
            self._buffers[pid].append(tp(lm))
            self._currently_visible[pid] = True

    def mark_all_invisible(self) -> None:
        """このフレームは検出結果が空。buffer をクリアして点・線とも消す。"""
        for pid in TRACKED_POINTS:
            self._currently_visible[pid] = False
            self._buffers[pid].clear()

    def is_currently_visible(self, pid: int) -> bool:
        return self._currently_visible.get(pid, False)

    def reset(self) -> None:
        """全バッファをクリア（モード切替時など）。"""
        for buf in self._buffers.values():
            buf.clear()
        for pid in TRACKED_POINTS:
            self._currently_visible[pid] = False

    def items(self):
        """(landmark_id, color, points_list) のイテレータを返す。"""
        for pid in TRACKED_POINTS:
            yield pid, TRAIL_COLORS[pid], list(self._buffers[pid])

    def is_empty(self) -> bool:
        return all(len(buf) < 2 for buf in self._buffers.values())
