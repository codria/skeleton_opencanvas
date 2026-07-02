"""
main.py
アプリ起動・PyQt6 イベントループ開始・起動時エラーハンドリングを担当する。
"""

import sys
import logging
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMessageBox
from app.config_loader import ConfigLoader, ConfigLoadError
from app.camera import Camera, CameraNotFoundError
from app.pose_estimator import PoseEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    app = QApplication(sys.argv)

    # 設定ファイルの読み込み
    try:
        config = ConfigLoader("config.yaml")
        logger.info("設定ファイル読み込み完了")
    except ConfigLoadError as e:
        QMessageBox.critical(None, "起動エラー", f"設定ファイルの読み込みに失敗しました。\n\n{e}")
        sys.exit(1)

    # カメラの起動
    camera = Camera(
        device_index=config.get("camera.device_index", 0),
        width=config.get("camera.width", 1280),
        height=config.get("camera.height", 720),
        fps=config.get("camera.fps", 30),
    )
    # カメラ起動失敗は fatal にせず warning に留めて続行する（動画ファイル再生モード）。
    # ユーザーは V キーまたは「動画選択」ボタンから動画を開いて使える。
    # C キーまたは「カメラ」ボタンで後から再接続を試みることも可能。
    try:
        camera.start()
    except CameraNotFoundError as e:
        logger.warning(f"カメラ起動失敗（動画ファイル再生モードで続行）: {e}")
        QMessageBox.warning(
            None, "カメラ検出失敗",
            f"カメラを開けませんでした。\n\n{e}\n\n"
            "動画ファイル再生モードで起動します。\n"
            "V キーまたは「動画選択」ボタンから動画を開いてください。"
        )

    # PoseEstimator の初期化
    # モデルファイルの候補を自動検索
    model_path = config.get("pose.model_path", "assets/models/pose_landmarker.task")
    model_candidates = [
        model_path,
        "assets/models/pose_landmarker_heavy.task",
        "assets/models/pose_landmarker_full.task",
        "assets/models/pose_landmarker_lite.task",
    ]
    import os
    resolved_model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            resolved_model_path = candidate
            break
    if resolved_model_path is None:
        QMessageBox.critical(
            None, "起動エラー",
            "MediaPipe モデルファイルが見つかりません。\n\n"
            "以下のいずれかを assets/models/ に配置してください：\n"
            + "\n".join(model_candidates)
        )
        camera.release()
        sys.exit(1)
    logger.info(f"モデルファイル: {resolved_model_path}")

    estimator = PoseEstimator(
        model_path=resolved_model_path,
        num_poses=config.get("pose.num_poses", 3),
        min_detection_confidence=config.get("pose.min_detection_confidence", 0.5),
        min_tracking_confidence=config.get("pose.min_tracking_confidence", 0.5),
        smoothing_alpha=config.get("pose.smoothing_alpha", 0.4),
    )

    # メインウィンドウの起動
    from app.main_window import MainWindow
    window = MainWindow(config, camera, estimator)
    window.show()

    logger.info("アプリ起動完了")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
