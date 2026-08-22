"""
user_settings.py
ユーザーがスライダー等で調整した値を JSON で永続化する。
起動時に load() で読み込み、終了時に save() で書き出す。
ファイルがなければデフォルトを返す、壊れていれば warning ログ＋デフォルト。
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# プロジェクトルート直下に置く（git 管理外）
SETTINGS_FILE = Path("user_settings.json")

DEFAULTS: dict = {
    "num_poses": 1,
    "smoothing_alpha": 0.25,
    "graph_scale": 1.0,
    "graph_visible": True,
    "show_bones": False,
    "trail_point_size": 6.0,
    "trail_line_width": 3.0,
    "trail_max_points": 32,
    "trail_visible": True,
    "overlay_alpha": 1.0,
    "mode2_size_scale": 0.28,
    "mode3_speed": 30.0,
    "mode3_angle": 0.0,
    # M キーで循環するマネキン描画スタイル：primitive / mesh / hidden
    "mannequin_style": "primitive",
    # Mode4（体験モード）のサブモード：instrument / magic
    "mode4_sub_mode": "instrument",
    # 画面全体を左右反転（鏡表示）。カメラ入力を推定前に flip するので、
    # 骨格・グラフ・魔法エフェクトすべて自動的に mirror される。
    "mirror_display": False,
    # 端の写り込みを 2 段階で除外する。
    #   ① 入力マスク領域 mask_area（画像正規化 x,y,w,h）：MediaPipe に渡す前に
    #      この外側を黒塗り。全画面からの余白 UI で設定。既定は左右 1/8 除外。
    "mask_area_x": 0.125,
    "mask_area_y": 0.0,
    "mask_area_w": 0.75,
    "mask_area_h": 1.0,
    #   ② 検出後フィルタ：マスク領域の内側からの余白（l/r/t/b）で指定する。
    #      判定領域は必ずマスク領域の内側になる。既定は 0（＝マスクと同じ）。
    "filter_inset_l": 0.0,
    "filter_inset_r": 0.0,
    "filter_inset_t": 0.0,
    "filter_inset_b": 0.0,
}


def load() -> dict:
    """設定を読み込んで dict で返す。欠損キーは DEFAULTS で補完。"""
    if not SETTINGS_FILE.exists():
        logger.info(f"{SETTINGS_FILE} が無いのでデフォルト設定を使用")
        return DEFAULTS.copy()
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"top-level が dict ではない: {type(data)}")
        merged = {**DEFAULTS, **data}
        logger.info(f"{SETTINGS_FILE} 読み込み完了")
        return merged
    except Exception as e:
        logger.warning(f"{SETTINGS_FILE} 読み込み失敗（デフォルト使用）: {e}")
        return DEFAULTS.copy()


def save(values: dict) -> None:
    """設定を書き出す。失敗しても例外は外に出さず warning ログだけ出す
    （アプリ終了処理を止めない）。"""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2, ensure_ascii=False)
        logger.info(f"{SETTINGS_FILE} に設定を保存")
    except Exception as e:
        logger.warning(f"{SETTINGS_FILE} 保存失敗: {e}")
