"""
base_mode.py
全モード共通の抽象基底クラス。OpenGL描画インターフェースを定義する。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseMode(ABC):

    @abstractmethod
    def initialize(self) -> None:
        """OpenGLリソースの初期化。GLWidget が初回切替時に一度だけ呼ぶ。
        モードを再度切り替えても再初期化はされない（GLTF・テクスチャ等の重いリソースを保持する）。
        毎回のモード遷移時の処理は on_mode_enter() を使う。"""

    @abstractmethod
    def draw(self, frame: np.ndarray | None,
             results: list,
             width: int, height: int) -> None:
        """paintGL()から呼ばれる描画メソッド。
        frame   : OpenCVのBGRフレーム（Noneの場合あり）
        results : PoseEstimatorの推定結果（空リストの場合あり）
        width   : 描画領域の幅（px）
        height  : 描画領域の高さ（px）
        OpenGLコンテキストはpaintGL()の時点で既にアクティブのため引数不要。
        """

    def on_resize(self, width: int, height: int) -> None:
        """ウィンドウリサイズ時にGLWidget.resizeGL()から呼ばれる（任意実装）。"""

    def on_mode_enter(self) -> None:
        """このモードがアクティブになった時に呼ばれる（任意実装）。"""

    def on_mode_exit(self) -> None:
        """このモードが非アクティブになった時に呼ばれる（任意実装）。"""

    def on_wheel(self, delta_y: int) -> None:
        """マウスホイールイベント（任意実装）。
        delta_y は QWheelEvent.angleDelta().y()。通常 1 ノッチ = ±120。
        """
