"""
pose_estimator.py
MediaPipe Pose Landmarker の初期化・骨格推定・結果の正規化・リソース解放を担当する。
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)


@dataclass
class Landmark:
    x: float          # 画面幅に対する正規化座標 [0.0, 1.0]
    y: float          # 画面高さに対する正規化座標 [0.0, 1.0]
    z: float          # 奥行き（腰を基準とした相対値）
    visibility: float # 可視信頼度 [0.0, 1.0]


@dataclass
class PoseLandmarkResult:
    landmarks: list[Landmark]  # 33点のランドマーク（MediaPipe定義順）
    person_index: int          # 人物インデックス（複数人対応時）


class PoseEstimator:
    def __init__(
        self,
        model_path: str,
        num_poses: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """MediaPipe Pose Landmarker を初期化する。

        Args:
            model_path               : pose_landmarker.task のパス
            num_poses                : 検出する最大人数
            min_detection_confidence : 検出信頼度の閾値
            min_tracking_confidence  : トラッキング信頼度の閾値
        """
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms: int = 0
        logger.info(
            f"PoseEstimator 初期化完了 "
            f"num_poses={num_poses} "
            f"min_detection={min_detection_confidence} "
            f"min_tracking={min_tracking_confidence}"
        )

    def estimate(self, frame: np.ndarray) -> list[PoseLandmarkResult]:
        """BGRフレームを受け取り、全人物のランドマークリストを返す。
        検出できない場合は空リストを返す。

        Args:
            frame : OpenCV BGR フォーマットの numpy 配列

        Returns:
            検出した人物ごとの PoseLandmarkResult のリスト
        """
        # BGR → RGB 変換
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # タイムスタンプをインクリメント（VIDEO モードに必要）
        self._frame_timestamp_ms += 33  # 約30fps想定

        detection_result = self._landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms
        )

        if not detection_result.pose_landmarks:
            return []

        results: list[PoseLandmarkResult] = []
        for person_index, pose_landmarks in enumerate(detection_result.pose_landmarks):
            landmarks = [
                Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility if lm.visibility is not None else 0.0,
                )
                for lm in pose_landmarks
            ]
            results.append(PoseLandmarkResult(
                landmarks=landmarks,
                person_index=person_index,
            ))

        return results

    def release(self) -> None:
        """MediaPipe のリソースを解放する。アプリ終了時に呼ぶ。"""
        self._landmarker.close()
        logger.info("PoseEstimator リソースを解放しました。")


if __name__ == "__main__":
    import sys
    import time
    import os

    # プロジェクトルート（app/の親）をカレントディレクトリに設定
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"作業ディレクトリ: {os.getcwd()}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # モデルファイルの候補を自動検索
    default_candidates = [
        "assets/models/pose_landmarker.task",
        "assets/models/pose_landmarker_heavy.task",
        "assets/models/pose_landmarker_full.task",
        "assets/models/pose_landmarker_lite.task",
    ]
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if model_path is None:
        for candidate in default_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path is None:
        print("❌ モデルファイルが見つかりません。")
        print("以下のいずれかを assets/models/ に配置してください：")
        for c in default_candidates:
            print(f"  {c}")
        sys.exit(1)

    print(f"PoseEstimator テスト開始 (model_path={model_path})")

    # PoseEstimator 初期化
    try:
        t0 = time.perf_counter()
        estimator = PoseEstimator(
            model_path=model_path,
            num_poses=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        t1 = time.perf_counter()
        print(f"初期化時間: {t1 - t0:.2f} 秒")
    except Exception as e:
        print(f"❌ PoseEstimator 初期化失敗: {e}")
        sys.exit(1)

    # カメラからフレームを取得して推定
    sys.path.insert(0, ".")
    from app.camera import Camera, CameraNotFoundError

    cam = Camera(device_index=0, width=1280, height=720, fps=30)
    try:
        cam.start()
        print("カメラ起動完了。10フレーム分の骨格推定を試みます...")
        print("カメラの前に立ってください。")

        success = 0
        for i in range(10):
            frame = cam.read_frame()
            if frame is None:
                print(f"  フレーム {i+1}/10 : フレーム取得失敗")
                continue

            t2 = time.perf_counter()
            results = estimator.estimate(frame)
            t3 = time.perf_counter()
            elapsed_ms = (t3 - t2) * 1000

            if results:
                p = results[0]
                nose = p.landmarks[0]
                print(
                    f"  フレーム {i+1}/10 : {len(results)}人検出 "
                    f"鼻座標=({nose.x:.3f}, {nose.y:.3f}) "
                    f"推定時間={elapsed_ms:.1f}ms"
                )
                success += 1
            else:
                print(f"  フレーム {i+1}/10 : 人物未検出 推定時間={elapsed_ms:.1f}ms")

        print(f"\n結果: 10フレーム中 {success} フレームで人物検出")
        if success > 0:
            print("✅ PoseEstimator OK")
        else:
            print("⚠️  人物が検出されませんでした（カメラの前に立って再試行してください）")

    except CameraNotFoundError as e:
        print(f"❌ カメラエラー: {e}")
    finally:
        cam.release()
        estimator.release()
        print("リソースを解放しました。")
