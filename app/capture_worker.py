"""
capture_worker.py
カメラキャプチャと骨格推定をバックグラウンドスレッドで実行するワーカー。
カメラ取得スレッドと推定スレッドを分離し、推定処理がカメラ取得をブロックしない構造にする。
"""

from __future__ import annotations
import logging
import time
import threading
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class CaptureWorker(QThread):
    """カメラ取得・骨格推定をバックグラウンドで実行し、結果をシグナルで通知する。

    構造：
      CameraThread  : カメラから常時フレームを取得し最新フレームをバッファに保持
      EstimateLoop  : バッファから最新フレームを取得して骨格推定 → シグナル送信
    """

    # メインスレッドへの通知シグナル（フレーム・推定結果・FPS）
    frame_ready = pyqtSignal(np.ndarray, list, float)

    def __init__(self, camera, estimator, parent=None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._estimator = estimator
        self._running = False

        # カメラスレッドと推定スレッド間の共有バッファ
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

    def _camera_loop(self) -> None:
        """カメラ取得専用スレッド。常時フレームを取得してバッファを更新する。"""
        while self._running:
            frame = self._camera.read_frame()
            if frame is None:
                logger.warning("フレーム取得失敗。スキップします。")
                continue
            with self._frame_lock:
                self._latest_frame = frame

    def run(self) -> None:
        """推定スレッドのメインループ。バッファから最新フレームを取得して推定する。"""
        self._running = True
        logger.info("CaptureWorker 開始")

        # カメラ取得を別スレッドで開始
        camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        camera_thread.start()

        fps = 0.0
        frame_count = 0
        t_start = time.perf_counter()

        while self._running:
            # 最新フレームを取得
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.001)
                continue

            results = self._estimator.estimate(frame)

            # FPS計測（1秒ごとに更新）
            frame_count += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()

            self.frame_ready.emit(frame, results, fps)

        camera_thread.join(timeout=2.0)
        logger.info("CaptureWorker 終了")

    def stop(self) -> None:
        """ワーカーを停止する。"""
        self._running = False
        self.wait()
        logger.info("CaptureWorker 停止完了")


