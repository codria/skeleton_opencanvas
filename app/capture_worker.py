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

    # メインスレッドへの通知シグナル
    # (フレーム, 推定結果, FPS, フレーム ID, シーク世代)
    # frame_idx は動画ファイルでは「そのフレームのインデックス」、カメラでは -1
    # seek_gen は seek/switch_source のたびに +1。メイン側で古い emit を捨てるのに使う
    frame_ready = pyqtSignal(np.ndarray, list, float, int, int)

    def __init__(self, camera, estimator, parent=None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._estimator = estimator
        self._running = False

        # カメラスレッドと推定スレッド間の共有バッファ
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_idx: int = -1
        self._frame_lock = threading.Lock()
        # シーク世代（メインスレッドが古い emit を弾くために使う）
        self._seek_gen: int = 0

    def _camera_loop(self) -> None:
        """カメラ取得専用スレッド。常時フレームを取得してバッファを更新する。
        動画ファイル時は元 fps × speed に合わせて間隔を守る（早送り防止＋速度制御）。
        一時停止時はバッファを更新しない（描画は前フレームを維持）。
        カメラ時は cap.read() のブロックでハードウェア fps に同期する。
        """
        while self._running:
            if self._camera.paused:
                time.sleep(0.03)
                continue

            t0 = time.perf_counter()
            frame = self._camera.read_frame()
            if frame is None:
                # ループ OFF EOF 等で paused=True になる場合あり
                time.sleep(0.01)
                continue
            # read 直後の frame_pos は「次に読むフレーム」を指すので、-1 でこのフレームの ID。
            # カメラ入力時は camera.frame_pos が常に 0 を返すので -1 にして区別する。
            if self._camera.is_video_file:
                cur_idx = max(0, self._camera.frame_pos - 1)
            else:
                cur_idx = -1
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_idx = cur_idx

            if self._camera.is_video_file:
                effective_fps = self._camera.source_fps * self._camera.speed
                target_interval = 1.0 / max(effective_fps, 1.0)
                sleep_time = target_interval - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

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
        # frame_ready emit の最小間隔（メインスレッドのシグナル queue 詰まり防止）
        emit_interval = 1.0 / 30.0
        last_emit_t = 0.0

        while self._running:
            # 一時停止中は推定をスキップ（同じフレームを連続 estimate すると MediaPipe の
            # 内部トラッキング状態が更新され、特に複数人時に person_index がブレてチラつく）
            if self._camera.paused:
                time.sleep(0.05)
                continue

            # 最新フレームを取得（取り出した時点の frame_idx も一緒に渡す）
            with self._frame_lock:
                frame = self._latest_frame
                frame_idx = self._latest_frame_idx

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

            # emit を 30fps に上限制限：推定スレッドは止めずメインスレッドへの通知だけ間引く
            now = time.perf_counter()
            if now - last_emit_t >= emit_interval:
                self.frame_ready.emit(frame, results, fps, frame_idx, self._seek_gen)
                last_emit_t = now

        camera_thread.join(timeout=2.0)
        logger.info("CaptureWorker 終了")

    @property
    def seek_gen(self) -> int:
        return self._seek_gen

    def switch_source(self, source: int | str) -> None:
        """映像ソースを切替し、PoseEstimator のタイムスタンプもジャンプさせる。
        失敗時は Camera 側で SourceOpenError が送出される。"""
        self._camera.switch_source(source)
        # バッファに残った旧ソースのフレームを破棄して、新ソースの初フレームを待つ
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_idx = -1
        self._estimator.reset_timestamp()
        self._seek_gen += 1

    def seek(self, frame_index: int) -> None:
        """動画ファイルの再生位置をジャンプさせ、タイムスタンプもリセットする。"""
        self._camera.seek(frame_index)
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_idx = -1
        self._estimator.reset_timestamp()
        self._seek_gen += 1

    def stop(self) -> None:
        """ワーカーを停止する。"""
        self._running = False
        self.wait()
        logger.info("CaptureWorker 停止完了")


