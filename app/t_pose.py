"""
t_pose.py
T ポーズ（両腕を水平に伸ばした基本姿勢）の MediaPipe ランドマーク 33 点を生成する。
キャラ造形の確認用：実映像なしで Mode2/3 のマネキンを正面から見られる。

座標系：MediaPipe pose_world_landmarks 規約
  - 原点  : 腰の中心
  - x 軸  : 右が +
  - y 軸  : 下が +
  - z 軸  : カメラから奥が +
  - 単位  : メートル
身長 ≒ 1.55m（鼻 -0.66 〜 足首 +0.90）想定。
"""

from __future__ import annotations
from app.pose_constants import PoseLandmark
from app.pose_estimator import Landmark, PoseLandmarkResult


# 33 点の T ポーズ world 座標（メートル、腰原点）
_T_POSE_WORLD: dict[int, tuple[float, float, float]] = {
    # 頭部
    PoseLandmark.NOSE:            ( 0.000, -0.660, -0.050),
    PoseLandmark.LEFT_EYE_INNER:  (-0.025, -0.685, -0.060),
    PoseLandmark.LEFT_EYE:        (-0.045, -0.685, -0.060),
    PoseLandmark.LEFT_EYE_OUTER:  (-0.065, -0.685, -0.050),
    PoseLandmark.RIGHT_EYE_INNER: ( 0.025, -0.685, -0.060),
    PoseLandmark.RIGHT_EYE:       ( 0.045, -0.685, -0.060),
    PoseLandmark.RIGHT_EYE_OUTER: ( 0.065, -0.685, -0.050),
    PoseLandmark.LEFT_EAR:        (-0.085, -0.660,  0.000),
    PoseLandmark.RIGHT_EAR:       ( 0.085, -0.660,  0.000),
    PoseLandmark.MOUTH_LEFT:      (-0.030, -0.595, -0.040),
    PoseLandmark.MOUTH_RIGHT:     ( 0.030, -0.595, -0.040),
    # 上半身（両腕を水平に伸ばす）
    PoseLandmark.LEFT_SHOULDER:   (-0.200, -0.400,  0.000),
    PoseLandmark.RIGHT_SHOULDER:  ( 0.200, -0.400,  0.000),
    PoseLandmark.LEFT_ELBOW:      (-0.450, -0.400,  0.000),
    PoseLandmark.RIGHT_ELBOW:     ( 0.450, -0.400,  0.000),
    PoseLandmark.LEFT_WRIST:      (-0.700, -0.400,  0.000),
    PoseLandmark.RIGHT_WRIST:     ( 0.700, -0.400,  0.000),
    PoseLandmark.LEFT_PINKY:      (-0.785, -0.390,  0.000),
    PoseLandmark.RIGHT_PINKY:     ( 0.785, -0.390,  0.000),
    PoseLandmark.LEFT_INDEX:      (-0.785, -0.410,  0.000),
    PoseLandmark.RIGHT_INDEX:     ( 0.785, -0.410,  0.000),
    PoseLandmark.LEFT_THUMB:      (-0.755, -0.420,  0.020),
    PoseLandmark.RIGHT_THUMB:     ( 0.755, -0.420,  0.020),
    # 下半身（直立、肩幅より狭めの腰幅）
    PoseLandmark.LEFT_HIP:        (-0.100,  0.000,  0.000),
    PoseLandmark.RIGHT_HIP:       ( 0.100,  0.000,  0.000),
    PoseLandmark.LEFT_KNEE:       (-0.100,  0.450,  0.000),
    PoseLandmark.RIGHT_KNEE:      ( 0.100,  0.450,  0.000),
    PoseLandmark.LEFT_ANKLE:      (-0.100,  0.900,  0.000),
    PoseLandmark.RIGHT_ANKLE:     ( 0.100,  0.900,  0.000),
    PoseLandmark.LEFT_HEEL:       (-0.100,  0.925, -0.040),
    PoseLandmark.RIGHT_HEEL:      ( 0.100,  0.925, -0.040),
    PoseLandmark.LEFT_FOOT_INDEX: (-0.100,  0.925,  0.100),
    PoseLandmark.RIGHT_FOOT_INDEX:( 0.100,  0.925,  0.100),
}


def t_pose_world_landmarks() -> list[Landmark]:
    """T ポーズの pose_world_landmarks（33 点、メートル単位、腰原点）を返す。"""
    return [
        Landmark(*_T_POSE_WORLD.get(i, (0.0, 0.0, 0.0)), visibility=1.0)
        for i in range(33)
    ]


def t_pose_image_landmarks(image_origin_x: float = 0.5,
                            image_origin_y: float = 0.55,
                            scale: float = 0.40) -> list[Landmark]:
    """T ポーズの pose_landmarks（画像座標 0〜1）を返す。
    腰中心を (image_origin_x, image_origin_y) に配置し、world (m) を
    scale 倍して画像座標に変換する。
    デフォルトは画面中央やや下に腰、身長 1.55m が画像 0.62 単位に映る程度。
    """
    return [
        Landmark(
            x=image_origin_x + _T_POSE_WORLD.get(i, (0.0, 0.0, 0.0))[0] * scale,
            y=image_origin_y + _T_POSE_WORLD.get(i, (0.0, 0.0, 0.0))[1] * scale,
            z=_T_POSE_WORLD.get(i, (0.0, 0.0, 0.0))[2] * scale,
            visibility=1.0,
        )
        for i in range(33)
    ]


def t_pose_result() -> PoseLandmarkResult:
    """T ポーズの PoseLandmarkResult（image + world、1 人分）を返す。
    Mode2 は image、Mode3 は world を使うので両方含めておく。
    """
    return PoseLandmarkResult(
        landmarks=t_pose_image_landmarks(),
        world_landmarks=t_pose_world_landmarks(),
        person_index=0,
    )
