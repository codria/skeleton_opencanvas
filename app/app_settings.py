"""
app_settings.py
スライダー値・UI トグル等のアプリ全体パラメータを一元管理する。
set() で値を更新すると対応する pyqtSignal が emit され、各 view（ControlPanel）
や計算側（Renderer、Estimator）はそれを subscribe してリアクティブに反映する。

設計原則:
- 値は self._values dict に格納し、key 名で管理（type は dict 値で持つ）
- set/get インターフェースで内部実装の自由度を確保
- key ごとに対応する個別 pyqtSignal を持つ。signal 名は f"{key}_changed"
- to_dict/load_from_dict で永続化と相互運用
"""

from __future__ import annotations
import logging
from typing import Any
from PyQt6.QtCore import QObject, pyqtSignal

from app import user_settings

logger = logging.getLogger(__name__)


class AppSettings(QObject):
    """全 UI パラメータを一元的に保持する Model。

    使い方:
        settings = AppSettings()
        settings.trail_point_size_changed.connect(renderer.set_trail_point_size)
        settings.set("trail_point_size", 8.0)  # → signal が emit され renderer が更新される
    """

    # 値ごとの変更通知シグナル。key 名と一致させる。
    num_poses_changed = pyqtSignal(int)
    smoothing_alpha_changed = pyqtSignal(float)
    graph_scale_changed = pyqtSignal(float)
    graph_visible_changed = pyqtSignal(bool)
    show_bones_changed = pyqtSignal(bool)
    trail_point_size_changed = pyqtSignal(float)
    trail_line_width_changed = pyqtSignal(float)
    trail_max_points_changed = pyqtSignal(int)
    overlay_alpha_changed = pyqtSignal(float)
    mode2_size_scale_changed = pyqtSignal(float)
    mode3_speed_changed = pyqtSignal(float)
    mode3_angle_changed = pyqtSignal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # 既存の user_settings.load() を初期値ソースに。
        # ファイル無しならデフォルトが入る。
        self._values: dict[str, Any] = user_settings.load()

    # --- 基本アクセス ---------------------------------------------------------

    def get(self, key: str) -> Any:
        return self._values[key]

    def set(self, key: str, value: Any) -> None:
        """値を更新し、変更があれば対応 signal を emit する。
        同値なら何もしない（signal 再帰や無駄な再描画を防ぐ）。
        """
        if key not in self._values:
            logger.warning(f"AppSettings.set: 未知のキー {key!r}")
            return
        old = self._values[key]
        if old == value:
            return
        self._values[key] = value
        sig_name = f"{key}_changed"
        sig = getattr(self, sig_name, None)
        if sig is not None:
            sig.emit(value)
        else:
            logger.warning(f"AppSettings: {sig_name} シグナルが未定義")

    # --- 永続化 --------------------------------------------------------------

    def to_dict(self) -> dict:
        """user_settings.save に渡す形式で全値を返す。"""
        return self._values.copy()

    def save(self) -> None:
        """現在値をユーザー設定ファイルに保存する。"""
        user_settings.save(self.to_dict())

    # --- 主要キーを type-safe にアクセスしやすいよう property 化 -------------
    # 「文字列キーを書きたくない」場面用のショートカット。
    # 値を変える時は必ず set() を経由（signal 飛ばすため）。

    @property
    def num_poses(self) -> int:
        return int(self._values["num_poses"])

    @property
    def smoothing_alpha(self) -> float:
        return float(self._values["smoothing_alpha"])

    @property
    def graph_scale(self) -> float:
        return float(self._values["graph_scale"])

    @property
    def graph_visible(self) -> bool:
        return bool(self._values["graph_visible"])

    @property
    def show_bones(self) -> bool:
        return bool(self._values["show_bones"])

    @property
    def trail_point_size(self) -> float:
        return float(self._values["trail_point_size"])

    @property
    def trail_line_width(self) -> float:
        return float(self._values["trail_line_width"])

    @property
    def trail_max_points(self) -> int:
        return int(self._values["trail_max_points"])

    @property
    def overlay_alpha(self) -> float:
        return float(self._values["overlay_alpha"])

    @property
    def mode2_size_scale(self) -> float:
        return float(self._values["mode2_size_scale"])

    @property
    def mode3_speed(self) -> float:
        return float(self._values["mode3_speed"])

    @property
    def mode3_angle(self) -> float:
        return float(self._values["mode3_angle"])
