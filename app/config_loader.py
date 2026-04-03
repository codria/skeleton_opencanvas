"""
config_loader.py
config.yaml の読み込みと値の提供・ランタイム上書きを担当する。
"""

from __future__ import annotations
from typing import Any
import logging
import yaml


class ConfigLoadError(Exception):
    """config.yaml の読み込み・パースに失敗した場合に送出する。"""


class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        """config.yaml を読み込む。失敗時は ConfigLoadError を送出する。"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config: dict = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise ConfigLoadError(f"設定ファイルが見つかりません: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"設定ファイルのパースに失敗しました: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """ドット区切りキーで設定値を取得する。

        例:
            get("camera.device_index")  # → 0
            get("pose.model_path")      # → "assets/models/pose_landmarker.task"

        キーが存在しない場合は default を返す。
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    def set(self, key: str, value: Any) -> None:
        """ランタイムで設定値を上書きする。ファイルへの書き戻しは行わない。

        例:
            set("mode2.background_image", "assets/backgrounds/new.jpg")
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    print(f"ConfigLoader テスト開始 (config_path={config_path})")

    # 正常系：読み込みテスト
    try:
        t0 = time.perf_counter()
        cfg = ConfigLoader(config_path)
        t1 = time.perf_counter()
        print(f"読み込み時間: {t1 - t0:.4f} 秒")

        # get() テスト
        keys_to_check = [
            "camera.device_index",
            "camera.width",
            "camera.height",
            "camera.fps",
            "display.width",
            "display.height",
            "pose.num_poses",
            "pose.model_path",
            "mode2.background_image",
            "mode3.character_model",
            "mode3.rotation_speed",
        ]
        print("\n--- get() テスト ---")
        for key in keys_to_check:
            value = cfg.get(key)
            print(f"  {key}: {value}")

        # 存在しないキー
        missing = cfg.get("存在しない.キー", "DEFAULT")
        print(f"  存在しないキー → {missing}")

        # set() テスト
        print("\n--- set() テスト ---")
        cfg.set("mode2.background_image", "assets/backgrounds/new.jpg")
        print(f"  set後 mode2.background_image: {cfg.get('mode2.background_image')}")
        cfg.set("new_section.new_key", "new_value")
        print(f"  新規セクション new_section.new_key: {cfg.get('new_section.new_key')}")

        print("\n✅ ConfigLoader OK")

    except ConfigLoadError as e:
        print(f"❌ ConfigLoadError: {e}")

    # 異常系：存在しないファイル
    print("\n--- 異常系テスト ---")
    try:
        ConfigLoader("not_exist.yaml")
    except ConfigLoadError as e:
        print(f"  存在しないファイル → ConfigLoadError: OK ({e})")
