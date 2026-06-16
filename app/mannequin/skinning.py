"""
skinning.py
CPUスキニング：ボーン回転行列を計算し頂点を変形する。
MediaPipeのランドマークからボーン回転を推定する。
"""
from __future__ import annotations
import math
import numpy as np
from app.pose_constants import PoseLandmark

MIN_VIS = 0.45

# MediaPipeボーン名 → (始点landmark, 終点landmark)
# モデルのボーン名に部分マッチで対応付ける
BONE_LANDMARK_MAP = {
    "Hips":            (PoseLandmark.LEFT_HIP,          PoseLandmark.RIGHT_HIP),
    "Spine":           (PoseLandmark.LEFT_HIP,          PoseLandmark.LEFT_SHOULDER),
    "Chest":           (PoseLandmark.LEFT_HIP,          PoseLandmark.LEFT_SHOULDER),
    "Upper Chest":     (PoseLandmark.LEFT_SHOULDER,     PoseLandmark.RIGHT_SHOULDER),
    "Neck":            (PoseLandmark.LEFT_SHOULDER,     PoseLandmark.NOSE),
    "Head":            (PoseLandmark.LEFT_SHOULDER,     PoseLandmark.NOSE),
    "Left Shoulder":   (PoseLandmark.LEFT_SHOULDER,     PoseLandmark.LEFT_ELBOW),
    "Left Upper Arm":  (PoseLandmark.LEFT_SHOULDER,     PoseLandmark.LEFT_ELBOW),
    "Left Lower Arm":  (PoseLandmark.LEFT_ELBOW,        PoseLandmark.LEFT_WRIST),
    "Right Shoulder":  (PoseLandmark.RIGHT_SHOULDER,    PoseLandmark.RIGHT_ELBOW),
    "Right Upper Arm": (PoseLandmark.RIGHT_SHOULDER,    PoseLandmark.RIGHT_ELBOW),
    "Right Lower Arm": (PoseLandmark.RIGHT_ELBOW,       PoseLandmark.RIGHT_WRIST),
    "Left Upper Leg":  (PoseLandmark.LEFT_HIP,          PoseLandmark.LEFT_KNEE),
    "Left Lower Leg":  (PoseLandmark.LEFT_KNEE,         PoseLandmark.LEFT_ANKLE),
    "Left Foot":       (PoseLandmark.LEFT_ANKLE,        PoseLandmark.LEFT_FOOT_INDEX),
    "Right Upper Leg": (PoseLandmark.RIGHT_HIP,         PoseLandmark.RIGHT_KNEE),
    "Right Lower Leg": (PoseLandmark.RIGHT_KNEE,        PoseLandmark.RIGHT_ANKLE),
    "Right Foot":      (PoseLandmark.RIGHT_ANKLE,       PoseLandmark.RIGHT_FOOT_INDEX),
}


def _quat_to_mat(q) -> np.ndarray:
    """クォータニオン[x,y,z,w]を4x4回転行列に変換する。"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w), 0],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w), 0],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def _trs_to_mat(t, r, s) -> np.ndarray:
    """TRS（位置・回転・スケール）を4x4行列に変換する。"""
    mat = _quat_to_mat(r)
    mat[0,:3] *= s[0]
    mat[1,:3] *= s[1]
    mat[2,:3] *= s[2]
    mat[0,3], mat[1,3], mat[2,3] = t[0], t[1], t[2]
    return mat


def _vec_to_quat(from_vec, to_vec) -> np.ndarray:
    """from_vecをto_vecに向ける回転クォータニオンを返す。"""
    a = np.array(from_vec, dtype=np.float32)
    b = np.array(to_vec,   dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return np.array([0,0,0,1], dtype=np.float32)
    a /= na
    b /= nb
    dot = float(np.clip(np.dot(a, b), -1, 1))
    if dot > 0.9999:
        return np.array([0,0,0,1], dtype=np.float32)
    if dot < -0.9999:
        perp = np.array([1,0,0], dtype=np.float32)
        if abs(a[0]) > 0.9:
            perp = np.array([0,1,0], dtype=np.float32)
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return np.array([*axis, 0], dtype=np.float32)
    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    angle = math.acos(dot)
    s = math.sin(angle/2)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(angle/2)], dtype=np.float32)


def _lm_vec(lms, idx_a, idx_b) -> np.ndarray | None:
    """2ランドマーク間のベクトルを返す。視認性が低い場合はNone。"""
    la, lb = lms[idx_a], lms[idx_b]
    if la.visibility < MIN_VIS or lb.visibility < MIN_VIS:
        return None
    return np.array([lb.x - la.x, -(lb.y - la.y), -(lb.z - la.z)], dtype=np.float32)


def compute_bone_transforms(model, lms) -> list[np.ndarray]:
    """
    ランドマークからボーンのワールド変換行列リストを計算する。
    戻り値: [bone_count × 4x4 行列]（スキニング行列 = world_mat @ inv_bind）
    """
    n_bones = len(model.bone_names)

    # バインドポーズのノードTRS行列を計算
    node_mats: dict[int, np.ndarray] = {}
    for i, joint_idx in enumerate(model.skin_joints):
        trs = model.node_trs[joint_idx]
        node_mats[joint_idx] = _trs_to_mat(trs['t'], trs['r'], trs['s'])

    # ランドマークからボーン回転を上書き
    for i, bone_name in enumerate(model.bone_names):
        joint_idx = model.bone_nodes[i]
        for key, (idx_a, idx_b) in BONE_LANDMARK_MAP.items():
            if key.lower() not in bone_name.lower():
                continue
            vec = _lm_vec(lms, idx_a, idx_b)
            if vec is None:
                break
            # バインドポーズのボーン方向（ローカルY軸 = ボーン方向）
            trs = model.node_trs[joint_idx]
            bind_mat = _trs_to_mat(trs['t'], trs['r'], trs['s'])
            bind_dir = bind_mat[:3, 1]  # Y軸
            bind_dir_n = np.linalg.norm(bind_dir)
            if bind_dir_n > 1e-6:
                bind_dir /= bind_dir_n

            q = _vec_to_quat(bind_dir, vec)
            rot_mat = _quat_to_mat(q)
            # 平行移動・スケールは保持して回転だけ上書き
            new_mat = rot_mat.copy()
            new_mat[:3, 3] = bind_mat[:3, 3]
            node_mats[joint_idx] = new_mat
            break

    # 親から子へワールド行列を伝播
    world_mats: dict[int, np.ndarray] = {}
    def get_world(node_idx):
        if node_idx in world_mats:
            return world_mats[node_idx]
        local = node_mats.get(node_idx, np.eye(4, dtype=np.float32))
        parent_idx = model.node_parent.get(node_idx)
        if parent_idx is not None and parent_idx in model.skin_joints or \
           parent_idx in [model.bone_nodes[0]]:
            world = get_world(parent_idx) @ local
        else:
            world = local
        world_mats[node_idx] = world
        return world

    for joint_idx in model.skin_joints:
        get_world(joint_idx)

    # スキニング行列 = world_mat @ inv_bind（列優先に転置）
    skin_mats = []
    for i, joint_idx in enumerate(model.skin_joints):
        world = world_mats.get(joint_idx, np.eye(4, dtype=np.float32))
        # inv_bindはGLTF列優先なので転置
        inv = model.inv_bind[i].T
        skin_mats.append(world @ inv)
    return skin_mats


def apply_skinning(model, skin_mats) -> tuple[np.ndarray, np.ndarray]:
    """
    スキニング行列を頂点に適用して変形後の頂点・法線を返す。
    """
    n = len(model.positions)
    out_pos = np.zeros((n, 3), dtype=np.float32)
    out_nor = np.zeros((n, 3), dtype=np.float32)

    # 位置を同次座標に
    pos_h = np.hstack([model.positions, np.ones((n,1), dtype=np.float32)])  # (N,4)
    nor_h = np.hstack([model.normals,   np.zeros((n,1), dtype=np.float32)]) # (N,4)

    for i in range(n):
        j = model.joints[i]   # (4,) bone indices
        w = model.weights[i]  # (4,) weights
        mat = np.zeros((4,4), dtype=np.float32)
        total_w = 0.0
        for k in range(4):
            if w[k] > 0 and j[k] < len(skin_mats):
                m = skin_mats[j[k]]
                if np.isfinite(m).all():
                    mat += w[k] * m
                    total_w += w[k]
        if total_w < 1e-6:
            # ウェイトがない場合はバインドポーズのまま
            out_pos[i] = model.positions[i]
            out_nor[i] = model.normals[i]
            continue
        if total_w < 0.999:
            mat /= total_w  # 正規化
        result_pos = mat @ pos_h[i]
        result_nor = mat @ nor_h[i]
        if np.isfinite(result_pos).all():
            out_pos[i] = result_pos[:3]
        else:
            out_pos[i] = model.positions[i]
        if np.isfinite(result_nor).all():
            out_nor[i] = result_nor[:3]
        else:
            out_nor[i] = model.normals[i]

    # 法線を正規化
    norms = np.linalg.norm(out_nor, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    out_nor /= norms

    return out_pos, out_nor
