"""
camera.py
カメラデバイス接続・解像度およびFPS設定・フレーム取得・起動時疎通確認を担当する。
"""

from __future__ import annotations
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraNotFoundError(Exception):
    """起動時にカメラデバイスが見つからない場合に送出する。"""


class Camera:
    def __init__(self, device_index: int, width: int, height: int, fps: int) -> None:
        """カメラデバイス番号・解像度・FPS を受け取る。

        Args:
            device_index : カメラデバイス番号（0, 1, 2 ...）
            width        : キャプチャ幅（px）
            height       : キャプチャ高さ（px）
            fps          : 目標フレームレート
        """
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        """カメラを起動する。
        カメラが見つからない・映像が取得できない場合は CameraNotFoundError を送出する。
        """
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            raise CameraNotFoundError(
                f"カメラが見つかりません（device_index={self._device_index}）。"
                "カメラを接続し直すか、config.yaml の camera.device_index を変更してください。"
            )

        # 解像度・FPS を設定
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # 疎通確認：実際にフレームが取得できるか検証
        ret, _ = self._cap.read()
        if not ret:
            self._cap.release()
            raise CameraNotFoundError(
                f"カメラ（device_index={self._device_index}）からフレームを取得できません。"
            )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"カメラ起動完了 device_index={self._device_index} "
            f"解像度={actual_w}x{actual_h} FPS={actual_fps:.1f}"
        )

    def read_frame(self) -> np.ndarray | None:
        """最新フレームをBGR numpy配列で返す。
        取得できない場合は None を返す（例外は送出しない）。
        """
        if self._cap is None or not self._cap.isOpened():
            logger.warning("カメラが起動していません。")
            return None

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("フレームの取得に失敗しました。")
            return None

        return frame

    def release(self) -> None:
        """カメラリソースを解放する。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("カメラリソースを解放しました。")


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
