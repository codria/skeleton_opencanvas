"""
pose_stream.py
CaptureWorker と表示層（GLWidget・グラフ・トレイル・ControlPanel）の間に挟む
薄いフレームブローカ。worker から流れてくる raw データ（frame, results, fps,
frame_idx, seek_gen）を加工して PoseFrame として再 emit する。

責務:
- 古い seek_gen の emit を弾く（シーク前にキュー入りした古いフレームを破棄）
- 動画ソースなら元動画 fps から t_video（動画内時刻）を計算
- subscriber に PoseFrame dataclass で配る

これにより MainWindow._on_frame_ready に詰まっていた "判断＋計算" が PoseStream
に集約され、表示側は PoseFrame を受け取って使うだけになる。
書き出し（VideoExporter）からも同じ push() で流せるので、ライブと書出のフレーム
処理パイプラインが共通化できる。
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


@dataclass
class PoseFrame:
    """1 フレーム分の推定結果＋メタデータ。subscriber はこれを使うだけで足りる。"""

    frame: np.ndarray              # BGR ndarray
    results: list                  # PoseLandmarkResult のリスト
    fps: float                     # 推定スレッドの実行 fps（表示用）
    frame_idx: int                 # 動画ファイルなら 0-based フレーム ID、カメラは -1
    seek_gen: int                  # シーク世代（古い frame を弾くため）
    t_video: float | None          # 動画内時刻（秒）。カメラ入力なら None
    is_video: bool                 # 元ソースが動画ファイルか


class PoseStream(QObject):
    """フレーム到着の集約レイヤー。worker からの push を加工して subscriber に配る。

    通常は MainWindow が PoseFrame 1 つを購読する。将来的には各 view（グラフ、
    トレイル等）が直接購読する形に分散していけるよう、frame_arrived 1 本に集約。
    """

    frame_arrived = pyqtSignal(object)  # PoseFrame

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current_seek_gen: int = 0
        self._source_fps: float = 30.0
        self._is_video: bool = False

    # --- 状態セット ----------------------------------------------------------

    def set_seek_gen(self, gen: int) -> None:
        """worker.seek_gen が変わったら呼ぶ。これ未満の emit は破棄される。"""
        self._current_seek_gen = int(gen)
        logger.debug(f"PoseStream seek_gen={self._current_seek_gen}")

    def set_source(self, is_video: bool, source_fps: float) -> None:
        """映像ソース切替時に呼ぶ。t_video 計算の分母とフラグを更新。"""
        self._is_video = bool(is_video)
        self._source_fps = max(1.0, float(source_fps))
        logger.debug(f"PoseStream source: is_video={self._is_video} fps={self._source_fps}")

    # --- 入力（worker → ここ）-----------------------------------------------

    def push(self, frame: np.ndarray, results: list,
             fps: float, frame_idx: int, seek_gen: int) -> None:
        """worker.frame_ready から呼ばれる。古い世代は弾いて PoseFrame で配る。"""
        if seek_gen < self._current_seek_gen:
            # シーク前にキューに入っていた古い emit。捨てる。
            return
        t_video: float | None = None
        if frame_idx >= 0:
            # 元動画 fps で時刻を計算。worker の fps 引数（実行 fps）は揺れるので使わない。
            t_video = frame_idx / self._source_fps
        pf = PoseFrame(
            frame=frame,
            results=results,
            fps=fps,
            frame_idx=frame_idx,
            seek_gen=seek_gen,
            t_video=t_video,
            is_video=self._is_video,
        )
        self.frame_arrived.emit(pf)
