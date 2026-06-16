"""
gltf_loader.py
BlockMan.gltfのメッシュ・スケルトン・スキニング情報を読み込む。
"""
from __future__ import annotations
import os
import struct
import logging
import numpy as np
from pygltflib import GLTF2

logger = logging.getLogger(__name__)

TYPE_MAP  = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
COMP_MAP  = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}


class GLTFModel:
    """GLTFモデルのメッシュ・スケルトン情報を保持する。"""

    def __init__(self):
        self.positions:   np.ndarray = None   # (N,3) float32
        self.normals:     np.ndarray = None   # (N,3) float32
        self.joints:      np.ndarray = None   # (N,4) int32
        self.weights:     np.ndarray = None   # (N,4) float32
        self.indices:     np.ndarray = None   # (M,)  int32
        self.inv_bind:    np.ndarray = None   # (B,4,4) float32  逆バインド行列
        self.bone_names:  list[str]  = []     # ボーン名リスト
        self.bone_nodes:  list[int]  = []     # ボーンのノードインデックス
        self.node_parent: dict[int,int] = {}  # ノード→親ノード
        self.node_trs:    dict       = {}     # ノード→{t,r,s}
        self.skin_joints: list[int]  = []     # スキンのジョイントノードインデックス

    @classmethod
    def load(cls, gltf_path: str) -> 'GLTFModel':
        model = cls()
        gltf = GLTF2().load(gltf_path)
        base_dir = os.path.dirname(gltf_path)

        # バッファ読み込み
        buffers = []
        for buf in gltf.buffers:
            path = os.path.join(base_dir, buf.uri)
            with open(path, 'rb') as f:
                buffers.append(f.read())

        def get_accessor(acc_idx, dtype=None):
            acc = gltf.accessors[acc_idx]
            bv  = gltf.bufferViews[acc.bufferView]
            data = buffers[bv.buffer]
            offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
            n    = TYPE_MAP[acc.type]
            fmt  = COMP_MAP[acc.componentType]
            size = struct.calcsize(fmt)
            stride = bv.byteStride or (n * size)
            rows = []
            for i in range(acc.count):
                row = struct.unpack_from(fmt*n, data, offset + i*stride)
                rows.append(row)
            arr = np.array(rows)
            if dtype:
                arr = arr.astype(dtype)
            return arr

        # メッシュ
        prim = gltf.meshes[0].primitives[0]
        attrs = prim.attributes
        model.positions = get_accessor(attrs.POSITION,  np.float32)
        model.normals   = get_accessor(attrs.NORMAL,    np.float32)
        model.joints    = get_accessor(attrs.JOINTS_0,  np.int32)
        model.weights   = get_accessor(attrs.WEIGHTS_0, np.float32)
        model.indices   = get_accessor(prim.indices,    np.int32).flatten()

        # スキン
        skin = gltf.skins[0]
        model.skin_joints = skin.joints
        ibm_flat = get_accessor(skin.inverseBindMatrices, np.float32)
        # GLTFは列優先4x4行列
        model.inv_bind = ibm_flat.reshape(-1, 4, 4)

        # ボーン名・TRS
        for joint_idx in skin.joints:
            node = gltf.nodes[joint_idx]
            model.bone_names.append(node.name or f"bone_{joint_idx}")
            model.bone_nodes.append(joint_idx)
            t = list(node.translation) if node.translation else [0,0,0]
            r = list(node.rotation)    if node.rotation    else [0,0,0,1]
            s = list(node.scale)       if node.scale       else [1,1,1]
            model.node_trs[joint_idx] = {'t': t, 'r': r, 's': s}

        # 親子関係
        for node_idx, node in enumerate(gltf.nodes):
            for child_idx in (node.children or []):
                model.node_parent[child_idx] = node_idx

        # 頂点を正規化（Yが0〜11.8の範囲 → -1〜1に正規化）
        y_min = model.positions[:,1].min()
        y_max = model.positions[:,1].max()
        height = y_max - y_min
        scale = 2.0 / height
        model.positions = (model.positions - [0, y_min + height/2, 0]) * scale
        # inv_bindも同じスケールで調整（平行移動は別途）
        # → スキニング計算時にスケール済み頂点を使う

        logger.info(
            f"GLTFModel loaded: {len(model.positions)}頂点 "
            f"{len(model.indices)}インデックス "
            f"{len(model.bone_names)}ボーン"
        )
        return model
