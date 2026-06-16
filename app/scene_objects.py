"""
scene_objects.py
動画ファイルと同名の YAML から読み込む 3D シーンオブジェクト定義。
座標系は MediaPipe pose_world_landmarks に揃える：
  - 原点：腰の中央
  - x：右 (+)
  - y：下 (+)
  - z：奥 (+、カメラから遠ざかる)
  - 単位：メートル
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BoxObject:
    """直方体オブジェクト。"""
    name: str = "box"
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)   # x, y, z [m]
    size: tuple[float, float, float] = (1.0, 0.05, 0.5) # 幅(x), 高さ(y), 奥行き(z) [m]
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0)   # x, y, z 回転 [度]
    color: tuple[float, float, float] = (0.5, 0.3, 0.2) # R, G, B (0.0〜1.0)


@dataclass
class Scene:
    """1 つの動画に紐づくシーンの内容。"""
    objects: list[BoxObject] = field(default_factory=list)
    # 床アンカー Y 座標（メートル、下向き正）。
    # 指定すると人物の y のみを固定（image_scale と併用可能）。
    floor_y: float | None = None
    # 画像座標 (pose_landmarks) → world 座標 (m) のスケール係数。
    # 「画像 1.0 単位 = world で何メートル」を表す比例定数。
    # 内部では (画像位置 - 原点) × scale で world オフセットに変換する。
    image_scale_x: float | None = None
    image_scale_y: float | None = None
    # 画像内の world 原点（デフォルト画像中央 (0.5, 0.5)）
    image_origin_x: float = 0.5
    image_origin_y: float = 0.5

    @classmethod
    def empty(cls) -> "Scene":
        return cls(objects=[])


def yaml_path_for_video(video_path: str) -> str:
    """動画ファイルパスから対応する .yaml ファイルパスを返す（拡張子置換）。"""
    base, _ = os.path.splitext(video_path)
    return base + ".yaml"


def load_scene(yaml_path: str) -> Scene:
    """YAML ファイルからシーンを読み込む。失敗時は空シーンを返す（warnログ）。"""
    if not os.path.exists(yaml_path):
        return Scene.empty()
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"シーン YAML の読み込み失敗: {yaml_path}: {e}")
        return Scene.empty()

    def _as_optional_float(value, name):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(f"{name} のパースに失敗: {value!r}")
            return None

    floor_y = _as_optional_float(data.get("floor_y"), "floor_y")

    # image_scale を優先、旧 image_size も読む（後方互換、warn ログ）
    image_scale = data.get("image_scale") or {}
    image_scale_x = _as_optional_float(image_scale.get("x"), "image_scale.x")
    image_scale_y = _as_optional_float(image_scale.get("y"), "image_scale.y")
    if image_scale_x is None and image_scale_y is None and "image_size" in data:
        legacy = data.get("image_size") or {}
        legacy_x = _as_optional_float(legacy.get("width_m"), "image_size.width_m")
        legacy_y = _as_optional_float(legacy.get("height_m"), "image_size.height_m")
        if legacy_x is not None or legacy_y is not None:
            logger.warning(
                "YAML の image_size は image_scale にリネームされました。"
                " image_scale: { x: <値>, y: <値> } へ書き換えを推奨します。"
            )
            image_scale_x = legacy_x
            image_scale_y = legacy_y

    image_origin = data.get("image_origin") or {}
    image_origin_x = _as_optional_float(image_origin.get("x"), "image_origin.x")
    image_origin_y = _as_optional_float(image_origin.get("y"), "image_origin.y")
    if image_origin_x is None:
        image_origin_x = 0.5
    if image_origin_y is None:
        image_origin_y = 0.5

    raw_objects = data.get("objects", []) or []
    objects: list[BoxObject] = []
    for i, obj in enumerate(raw_objects):
        try:
            otype = (obj.get("type") or "box").lower()
            if otype != "box":
                logger.warning(f"  [{i}] 未対応の type={otype!r}（現状は box のみ）")
                continue
            objects.append(BoxObject(
                name=str(obj.get("name", f"box{i}")),
                pos=_as_xyz(obj.get("pos", [0, 0, 0])),
                size=_as_xyz(obj.get("size", [1, 0.05, 0.5])),
                rot=_as_xyz(obj.get("rot", [0, 0, 0])),
                color=_as_xyz(obj.get("color", [0.5, 0.3, 0.2])),
            ))
        except Exception as e:
            logger.warning(f"  [{i}] パース失敗: {e}")
    logger.info(
        f"シーン読み込み: {yaml_path} "
        f"({len(objects)} objects, floor_y={floor_y}, "
        f"image_scale=({image_scale_x},{image_scale_y}), "
        f"image_origin=({image_origin_x},{image_origin_y}))"
    )
    return Scene(
        objects=objects,
        floor_y=floor_y,
        image_scale_x=image_scale_x,
        image_scale_y=image_scale_y,
        image_origin_x=image_origin_x,
        image_origin_y=image_origin_y,
    )


def _as_xyz(v) -> tuple[float, float, float]:
    """[x, y, z] のリスト/タプルを 3 要素の float タプルに正規化する。"""
    if v is None:
        return (0.0, 0.0, 0.0)
    if len(v) < 3:
        v = list(v) + [0.0] * (3 - len(v))
    return (float(v[0]), float(v[1]), float(v[2]))
