"""
main_window.py
PyQt6 メインウィンドウ・モード切り替え制御・操作ガイドQLabel管理を担当する。
Step 06改：CaptureWorker でカメラ取得・推定をバックグラウンドスレッド化
"""

from __future__ import annotations
import logging
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QMessageBox
from PyQt6.QtCore import Qt
from app.gl_widget import GLWidget
from app.camera import Camera, CameraNotFoundError
from app.pose_estimator import PoseEstimator
from app.capture_worker import CaptureWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, config, camera: Camera, estimator: PoseEstimator) -> None:
        super().__init__()
        self._config = config
        self._camera = camera
        self._estimator = estimator

        # ウィンドウ基本設定
        self.setWindowTitle("AIスケルトン体験デモ")
        w = config.get("display.width", 1280)
        h = config.get("display.height", 720)
        self.resize(w, h)

        # GLWidget をセントラルウィジェットに設定
        self._gl_widget = GLWidget(config, parent=self)
        self.setCentralWidget(self._gl_widget)

        # バックグラウンドワーカーの起動
        self._worker = CaptureWorker(camera, estimator, parent=self)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.start()

        logger.info(f"MainWindow 初期化完了 size={w}x{h}")

    def _on_frame_ready(self, frame: np.ndarray, results: list, fps: float) -> None:
        """ワーカーからフレーム・推定結果・FPSを受け取りGLWidgetに渡す。"""
        self._gl_widget.update_frame(frame, results, fps)

    def show_error(self, message: str) -> None:
        """QMessageBox.critical() でエラーをポップアップ表示する。"""
        QMessageBox.critical(self, "エラー", message)

    def keyPressEvent(self, event) -> None:
        """キー入力イベント。Step 08 で拡張予定。"""
        key = event.key()
        if key == Qt.Key.Key_Q or key == Qt.Key.Key_Escape:
            logger.info("終了キーが押されました。")
            self.close()
        elif key == Qt.Key.Key_F:
            self._gl_widget.toggle_debug()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """ウィンドウ終了時のリソース解放。"""
        logger.info("アプリ終了処理を開始します。")
        self._worker.stop()
        self._camera.release()
        self._estimator.release()
        event.accept()


