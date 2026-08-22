"""
camera.py
映像ソース（カメラデバイス or 動画ファイル）の接続・フレーム取得・起動時疎通確認を担当する。
ランタイムでのソース切替（カメラ⇄動画）と動画 EOF 時のループ再生に対応する。
"""

from __future__ import annotations
import logging
import platform
import threading
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# プラットフォーム別カメラバックエンド：
#   Windows : DirectShow (CAP_DSHOW)
#   macOS   : AVFoundation (CAP_AVFOUNDATION)
#   Linux 他: デフォルト (V4L2 等)
_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    _CAMERA_BACKEND = cv2.CAP_DSHOW
elif _SYSTEM == "Darwin":
    _CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
else:
    _CAMERA_BACKEND = cv2.CAP_ANY


class CameraNotFoundError(Exception):
    """起動時にカメラデバイスが見つからない場合に送出する。"""


class SourceOpenError(Exception):
    """ランタイムでの映像ソース切替に失敗した場合に送出する（軽微エラー）。"""


class Camera:
    def __init__(self, device_index: int, width: int, height: int, fps: int) -> None:
        """カメラデバイス番号・解像度・FPS を受け取る。

        Args:
            device_index : 初期カメラデバイス番号（0, 1, 2 ...）
            width        : キャプチャ幅（px、デバイスのみ有効）
            height       : キャプチャ高さ（px、デバイスのみ有効）
            fps          : 目標フレームレート（デバイスのみ有効）
        """
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        # 現在のソース：int（デバイス番号）or str（ファイルパス）
        self._source: int | str = device_index
        # read_frame / switch_source の競合防止
        self._lock = threading.Lock()
        # 動画再生制御
        self._paused = False
        self._loop = True
        self._speed = 1.0

    @property
    def device_index(self) -> int:
        """起動時に指定されたカメラデバイス番号（カメラ復帰用）。"""
        return self._device_index

    @property
    def is_video_file(self) -> bool:
        """現在のソースが動画ファイルか。"""
        return isinstance(self._source, str)

    @property
    def source_fps(self) -> float:
        """現在のソースの fps（cap から取得した実値）。
        不明な場合は構築時の self._fps を返す。"""
        if self._cap is None:
            return float(self._fps)
        v = self._cap.get(cv2.CAP_PROP_FPS)
        return v if v and v > 0 else float(self._fps)

    # --- 動画再生制御（動画ファイル時のみ意味を持つ） -----------------------

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def toggle_paused(self) -> bool:
        self._paused = not self._paused
        return self._paused

    @property
    def loop(self) -> bool:
        return self._loop

    def set_loop(self, loop: bool) -> None:
        self._loop = loop

    def toggle_loop(self) -> bool:
        self._loop = not self._loop
        return self._loop

    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.1, float(speed))

    @property
    def frame_pos(self) -> int:
        """動画ファイルの現在フレーム位置。カメラ時は 0。"""
        if self._cap is None or not self.is_video_file:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def frame_count(self) -> int:
        """動画ファイルの総フレーム数。カメラ時は 0。"""
        if self._cap is None or not self.is_video_file:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def seek(self, frame_index: int) -> None:
        """動画ファイルの再生位置を frame_index にジャンプさせる。"""
        with self._lock:
            if self._cap is None or not self.is_video_file:
                return
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = max(0, min(frame_index, max(total - 1, 0)))
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def start(self) -> None:
        """初期ソース（カメラ）を起動する。
        指定 device_index が使えない場合は近傍の番号（0..probe_max）を自動で試す。
        いずれも駄目なら CameraNotFoundError を送出する。
        """
        try:
            used = self._open_camera_auto(self._device_index)
            self._device_index = used   # カメラ復帰（C キー）でも使えた番号を使う
        except SourceOpenError as e:
            raise CameraNotFoundError(str(e)) from e

    @staticmethod
    def _probe_candidates(preferred: int, probe_max: int = 3) -> list[int]:
        """カメラ番号の探索順。preferred を先頭に 0..probe_max を重複なく続ける。"""
        seen: set[int] = set()
        out: list[int] = []
        for i in [preferred, *range(probe_max + 1)]:
            if i >= 0 and i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _open_camera_auto(self, preferred: int) -> int:
        """preferred から順にカメラ番号を試し、最初に開けた番号を返す。
        指定番号以外で開けた場合は、config 更新を促す警告を出す。
        全滅なら SourceOpenError。"""
        candidates = self._probe_candidates(preferred)
        last_err: Exception | None = None
        for i in candidates:
            try:
                self._open(i)
            except SourceOpenError as e:
                last_err = e
                logger.info(f"カメラ device_index={i} は使用不可: {e}")
                continue
            if i != preferred:
                logger.warning(
                    f"カメラ device_index={preferred} が使えないため {i} で起動しました。"
                    f" 次回から探索を省くなら config.yaml の camera.device_index を "
                    f"{i} にしてください。"
                )
            return i
        raise SourceOpenError(
            f"使用可能なカメラが見つかりません（試行 {candidates}）。最後のエラー: {last_err}"
        )

    def switch_source(self, source: int | str) -> None:
        """ランタイムで入力ソースを切替する。
        失敗時は SourceOpenError を送出し、既存ソースを継続する。
        """
        with self._lock:
            old_cap = self._cap
            old_source = self._source
            try:
                self._open(source)
            except SourceOpenError:
                # _open 内で失敗した場合は self._cap / self._source は変えていない
                raise
            else:
                if old_cap is not None and old_cap is not self._cap:
                    old_cap.release()
                logger.info(
                    f"映像ソース切替: {old_source!r} → {source!r}"
                )

    def _open(self, source: int | str) -> None:
        """指定ソースを開いて self._cap / self._source を更新する。
        失敗時は SourceOpenError を送出し、self の状態は変更しない。
        """
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)
            label = f"動画ファイル {source}"
        else:
            cap = cv2.VideoCapture(source, _CAMERA_BACKEND)
            label = f"カメラ device_index={source}"

        if not cap.isOpened():
            raise SourceOpenError(f"{label} を開けません。")

        if not isinstance(source, str):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)

        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise SourceOpenError(f"{label} からフレームを取得できません。")

        # 動画ファイルは先頭から再生し直す
        if isinstance(source, str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._cap = cap
        self._source = source
        # ソース切替時に再生制御を初期状態へリセット
        self._paused = False
        self._loop = True
        self._speed = 1.0

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"{label} を開きました 解像度={actual_w}x{actual_h} FPS={actual_fps:.1f}"
        )

    def read_frame(self) -> np.ndarray | None:
        """最新フレームをBGR numpy配列で返す。
        動画ファイルは EOF 到達時に先頭へ巻き戻してループ再生する。
        取得できない場合は None を返す（例外は送出しない）。
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                logger.warning("映像ソースが起動していません。")
                return None

            ret, frame = self._cap.read()
            if ret:
                return frame

            # 動画ファイル EOF: ループ ON なら先頭へ、OFF なら一時停止状態にする
            if isinstance(self._source, str):
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if ret:
                        return frame
                else:
                    self._paused = True
                    return None

            logger.warning("フレームの取得に失敗しました。")
            return None

    def release(self) -> None:
        """映像ソースのリソースを解放する。"""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                logger.info("映像ソースのリソースを解放しました。")


if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    device_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"カメラ接続テスト開始 (device_index={device_index})")
    print("終了するには Ctrl+C を押してください。")

    cam = Camera(device_index=device_index, width=1280, height=720, fps=30)
    try:
        t0 = time.perf_counter()
        cam.start()
        t1 = time.perf_counter()
        print(f"カメラ起動時間: {t1 - t0:.2f} 秒")

        print("フレーム取得を10回試みます...")
        success = 0
        t2 = time.perf_counter()
        for i in range(10):
            frame = cam.read_frame()
            if frame is not None:
                success += 1
                print(f"  フレーム {i+1}/10 : OK (shape={frame.shape})")
            else:
                print(f"  フレーム {i+1}/10 : 取得失敗")
        t3 = time.perf_counter()

        print(f"\n結果: 10回中 {success} 回成功")
        print(f"フレーム取得10回の合計時間: {t3 - t2:.2f} 秒")
        print(f"平均フレーム取得時間: {(t3 - t2) / 10 * 1000:.1f} ms/frame")
        if success == 10:
            print("✅ カメラ接続 OK")
        else:
            print("⚠️  一部フレームの取得に失敗しました")
    except CameraNotFoundError as e:
        print(f"❌ {e}")
    finally:
        t4 = time.perf_counter()
        cam.release()
        t5 = time.perf_counter()
        print(f"カメラ解放時間: {t5 - t4:.2f} 秒")
        print("カメラを解放しました。")
