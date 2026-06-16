"""
pose_constants.py
ランドマークインデックス定数・POSE_CONNECTIONS の定義（全モード共通参照）。
MediaPipe Pose Landmarker の33点定義に準拠。
"""


class PoseLandmark:
    NOSE            = 0
    LEFT_EYE_INNER  = 1
    LEFT_EYE        = 2
    LEFT_EYE_OUTER  = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE       = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR        = 7
    RIGHT_EAR       = 8
    MOUTH_LEFT      = 9
    MOUTH_RIGHT     = 10
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_PINKY      = 17
    RIGHT_PINKY     = 18
    LEFT_INDEX      = 19
    RIGHT_INDEX     = 20
    LEFT_THUMB      = 21
    RIGHT_THUMB     = 22
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28
    LEFT_HEEL       = 29
    RIGHT_HEEL      = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX= 32


# 骨格線の接続定義（モード1・2の描画に使用）
POSE_CONNECTIONS: list[tuple[int, int]] = [
    # 顔（目と口のみ接続）
    (1, 2), (2, 3),     # 左目：内→中→外
    (4, 5), (5, 6),     # 右目：内→中→外
    (9, 10),            # 口：左〜右
    # 上半身
    (11, 12),           # 左肩〜右肩
    (11, 13), (13, 15), # 左腕
    (15, 17), (15, 19), (15, 21),  # 左手
    (17, 19),
    (12, 14), (14, 16), # 右腕
    (16, 18), (16, 20), (16, 22),  # 右手
    (18, 20),
    # 体幹
    (11, 23), (12, 24), (23, 24),  # 肩〜腰
    # 下半身
    (23, 25), (25, 27), # 左脚
    (27, 29), (27, 31), (29, 31),  # 左足
    (24, 26), (26, 28), # 右脚
    (28, 30), (28, 32), (30, 32),  # 右足
]
