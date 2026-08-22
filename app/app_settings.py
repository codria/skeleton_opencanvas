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
    trail_visible_changed = pyqtSignal(bool)
    overlay_alpha_changed = pyqtSignal(float)
    mode2_size_scale_changed = pyqtSignal(float)
    mode3_speed_changed = pyqtSignal(float)
    mode3_angle_changed = pyqtSignal(float)
    mannequin_style_changed = pyqtSignal(str)  # "primitive" | "mesh" | "hidden"
    mirror_display_changed = pyqtSignal(bool)
    # 検出エリア。mask=入力マスク領域（全画面基準の x,y,w,h）、
    # filter=マスク内側からの余白 inset(l,r,t,b)。連動値なので専用シグナルでまとめて通知。
    mask_area_changed = pyqtSignal(float, float, float, float)
    filter_inset_changed = pyqtSignal(float, float, float, float)

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
    def trail_visible(self) -> bool:
        return bool(self._values["trail_visible"])

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

    @property
    def mannequin_style(self) -> str:
        return str(self._values["mannequin_style"])

    @property
    def mirror_display(self) -> bool:
        return bool(self._values["mirror_display"])

    def _area(self, prefix: str) -> tuple[float, float, float, float]:
        return (
            float(self._values[f"{prefix}_x"]),
            float(self._values[f"{prefix}_y"]),
            float(self._values[f"{prefix}_w"]),
            float(self._values[f"{prefix}_h"]),
        )

    def _set_area(self, prefix: str, signal, x, y, w, h) -> None:
        new = {f"{prefix}_x": float(x), f"{prefix}_y": float(y),
               f"{prefix}_w": float(w), f"{prefix}_h": float(h)}
        if all(self._values.get(k) == v for k, v in new.items()):
            return
        self._values.update(new)
        signal.emit(float(x), float(y), float(w), float(h))

    @property
    def mask_area(self) -> tuple[float, float, float, float]:
        return self._area("mask_area")

    def set_mask_area(self, x: float, y: float, w: float, h: float) -> None:
        """入力マスク領域を更新し mask_area_changed を 1 回 emit する。"""
        self._set_area("mask_area", self.mask_area_changed, x, y, w, h)

    @property
    def filter_inset(self) -> tuple[float, float, float, float]:
        """マスク領域の内側からの余白 (left, right, top, bottom)。"""
        return (
            float(self._values["filter_inset_l"]),
            float(self._values["filter_inset_r"]),
            float(self._values["filter_inset_t"]),
            float(self._values["filter_inset_b"]),
        )

    def set_filter_inset(self, l: float, r: float, t: float, b: float) -> None:
        new = {"filter_inset_l": float(l), "filter_inset_r": float(r),
               "filter_inset_t": float(t), "filter_inset_b": float(b)}
        if all(self._values.get(k) == v for k, v in new.items()):
            return
        self._values.update(new)
        self.filter_inset_changed.emit(float(l), float(r), float(t), float(b))

    @property
    def filter_area(self) -> tuple[float, float, float, float]:
        """検出後フィルタ領域 (x,y,w,h)。mask_area をマスク内側余白で縮めた矩形。
        必ずマスク領域の内側に収まる。"""
        mx, my, mw, mh = self.mask_area
        l, r, t, b = self.filter_inset
        fx = mx + l
        fy = my + t
        fw = max(0.01, mw - l - r)
        fh = max(0.01, mh - t - b)
        # マスク右端/下端を越えないよう保険（inset が大きすぎる場合）
        fx = min(fx, mx + mw - 0.01)
        fy = min(fy, my + mh - 0.01)
        fw = min(fw, mx + mw - fx)
        fh = min(fh, my + mh - fy)
        return (fx, fy, fw, fh)
