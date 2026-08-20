"""
sound_bank.py
Mode4（体験モード）のジェスチャーに応じた効果音を再生するバンク。

音源は assets/sounds/ 以下に配置する（git 管理外、各 PC で再取得）。
ファイル名は「元ファイル名(動作タグ).mp3」形式でも「元ファイル名.mp3」形式でも
どちらでも読める：先頭の (...) と _ より前を「元ファイル名」として比較する。

キー ← 元ファイル名 の対応は SOUND_MAP で定義。同時発火に強い pygame.mixer で
再生する（PyQt6 の QSoundEffect は WAV 専用のため MP3 不可）。
"""

from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# 動作キー → soundeffect-lab.info 上の「元ファイル名」（拡張子なし）。
# 現物ファイルは "元ファイル名(動作).mp3" でも "元ファイル名.mp3" でも読める。
SOUND_MAP: dict[str, str] = {
    # 楽器モード
    "right_arm_up": "ティンパニロール",
    "left_arm_up":  "ロールの閉め",
    "right_step":   "食べ物をパクッ",  # 右足が下（接地側）に切り替わった瞬間
    "left_step":    "可愛い動作",      # 左足が下（接地側）に切り替わった瞬間
    "crouch":       "しょげる",        # 静止しゃがみ姿勢
    # 魔法モード：各系統に「構え（charge）」と「着弾（hit）」の 2 音。
    # 火＝両腕 / 氷＝右腕 / 雷＝左腕。未 DL でも起動可（無音になるだけ）。
    "magic_fire_charge":    "火炎魔法1",
    "magic_fire_hit":       "爆発1",
    "magic_ice_charge":     "氷魔法2",
    "magic_ice_hit":        "氷魔法で凍結",
    "magic_thunder_charge": "雷魔法3",
    "magic_thunder_hit":    "雷魔法4",
}

_SUPPORTED_EXTS = (".mp3", ".wav", ".ogg")


class SoundBank:
    """assets/sounds/ から音源を発見して再生する。
    pygame.mixer 未導入・音源未発見・初期化失敗のいずれでも致命傷にしない
    （ログだけ出して該当キーの play() を no-op にする）。
    """

    def __init__(self, sounds_dir: str = "assets/sounds") -> None:
        self._enabled: bool = False
        self._sounds: dict = {}  # key -> pygame.mixer.Sound

        try:
            import pygame
        except ImportError:
            logger.warning("pygame が入っていないため Mode4 の音は再生されません。"
                            " pip install pygame で追加してください。")
            return

        try:
            pygame.mixer.init()
            self._enabled = True
        except Exception as e:
            logger.warning(f"pygame.mixer 初期化失敗: {e}（音は無効化）")
            return

        self._load(Path(sounds_dir), pygame)

    def _load(self, d: Path, pygame) -> None:
        if not d.is_dir():
            logger.warning(f"音源ディレクトリなし: {d} → 音は無効化")
            return
        found = 0
        for key, base_name in SOUND_MAP.items():
            path = self._find(d, base_name)
            if path is None:
                logger.warning(f"音源未発見: key={key} base={base_name!r}")
                continue
            try:
                self._sounds[key] = pygame.mixer.Sound(str(path))
                found += 1
                logger.info(f"音源読込: {key:14s} <- {path.name}")
            except Exception as e:
                logger.warning(f"音源読込失敗 {path.name}: {e}")
        logger.info(f"SoundBank: {found}/{len(SOUND_MAP)} 音源読込完了")

    @staticmethod
    def _find(d: Path, base_name: str) -> Path | None:
        """ファイル名の (…) や _ より前が base_name と一致する mp3/wav を返す。
        例: base_name='ティンパニロール' なら
            'ティンパニロール(右手UP).mp3'
            'ティンパニロール（右手UP）.mp3'
            'ティンパニロール_v2.mp3'
            'ティンパニロール.mp3'
          いずれも hit する。
        """
        for f in d.iterdir():
            if f.suffix.lower() not in _SUPPORTED_EXTS:
                continue
            stem = f.stem
            for sep in ("(", "(", "_"):
                stem = stem.split(sep)[0]
            if stem.strip() == base_name:
                return f
        return None

    def play(self, key: str) -> bool:
        if not self._enabled:
            return False
        s = self._sounds.get(key)
        if s is None:
            return False
        try:
            s.play()
            return True
        except Exception as e:
            logger.warning(f"音再生失敗 key={key}: {e}")
            return False

    def stop(self, key: str) -> bool:
        """key に対応する音源を即時停止する（同時再生されている全チャンネル分）。
        ぶつ切りで止まるので、自然に止めたい場合は fadeout() を使う。
        """
        if not self._enabled:
            return False
        s = self._sounds.get(key)
        if s is None:
            return False
        try:
            s.stop()
            return True
        except Exception as e:
            logger.warning(f"音停止失敗 key={key}: {e}")
            return False

    def fadeout(self, key: str, ms: int = 300) -> bool:
        """key に対応する音源を ms ミリ秒かけて減衰停止する。
        pygame.mixer.Sound.fadeout() の薄いラッパー。
        """
        if not self._enabled:
            return False
        s = self._sounds.get(key)
        if s is None:
            return False
        try:
            s.fadeout(int(ms))
            return True
        except Exception as e:
            logger.warning(f"音フェード失敗 key={key}: {e}")
            return False

    def release(self) -> None:
        if not self._enabled:
            return
        try:
            import pygame
            pygame.mixer.quit()
        except Exception:
            pass
