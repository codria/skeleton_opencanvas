"""
pose_estimator.py
MediaPipe Pose Landmarker の初期化・骨格推定・結果の正規化・リソース解放を担当する。
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)


@dataclass
class Landmark:
    x: float          # 画面幅に対する正規化座標 [0.0, 1.0]
    y: float          # 画面高さに対する正規化座標 [0.0, 1.0]
    z: float          # 奥行き（腰を基準とした相対値）
    visibility: float # 可視信頼度 [0.0, 1.0]


@dataclass
class PoseLandmarkResult:
    landmarks: list[Landmark]              # 画像座標のランドマーク（x,y∈[0,1], z 近似）
    world_landmarks: list[Landmark]        # 真の 3D 座標（腰原点・メートル単位、y は下が +）
    person_index: int                       # 人物インデックス（複数人対応時）


class PoseEstimator:
    def __init__(
        self,
        model_path: str,
        num_poses: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        smoothing_alpha: float = 0.4,
        center_priority: bool = False,
        mask_area: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        filter_area: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ) -> None:
        """MediaPipe Pose Landmarker を初期化する。

        Args:
            model_path               : pose_landmarker.task のパス
            num_poses                : ユーザーが最終的に扱いたい人数
            min_detection_confidence : 検出信頼度の閾値
            min_tracking_confidence  : トラッキング信頼度の閾値
            smoothing_alpha          : EMA の追従係数。
                                       1.0 で平滑化なし、0.0 で動かない。
                                       0.4 が初期値（手足のブルブルを抑える）。
            center_priority          : True にすると内部的に多めに検出して、
                                       画面中央に近い順にソートしてから num_poses に
                                       絞る。「外縁部にちらっと写ってる人」を
                                       選ばせたくない用途向け。
        """
        # 再作成用に保存
        self._model_path = model_path
        self._num_poses = num_poses
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._center_priority = bool(center_priority)
        # 検出エリア（画像正規化 x, y, w, h）。画面端の写り込みを 2 段階で除外する。
        # 各段は独立した領域を持つ（先方の要望が「入力で消す」「判定で外す」どちらでも
        # それぞれ個別に調整できる）：
        #   ① 入力マスク mask_area  : MediaPipe に渡す前にこの外を黒塗り（検出させない）
        #   ② 検出後フィルタ filter_area : 残った結果から重心がこの外の人物を捨てる
        # (0,0,1,1) ならその段は無効（全画面）。
        self._mask_area = self._sanitize_area(mask_area)
        self._filter_area = self._sanitize_area(filter_area)
        # estimate と set_num_poses の排他用
        self._lock = threading.Lock()

        self._landmarker = self._create_landmarker(self._effective_num_poses(num_poses))
        self._frame_timestamp_ms: int = 0
        self._smoothing_alpha: float = max(0.05, min(1.0, smoothing_alpha))
        # person_index → 直前フレームの平滑化済み (image_landmarks, world_landmarks)
        self._smoothed: dict[int, tuple[list[Landmark], list[Landmark]]] = {}
        logger.info(
            f"PoseEstimator 初期化完了 "
            f"num_poses={num_poses} "
            f"min_detection={min_detection_confidence} "
            f"min_tracking={min_tracking_confidence} "
            f"smoothing_alpha={self._smoothing_alpha} "
            f"center_priority={self._center_priority}"
        )

    def _effective_num_poses(self, user_num_poses: int) -> int:
        """MediaPipe に渡す実効的な num_poses。
        center_priority=True のときは、後段で中央近い順に絞るため多めに検出する。
        """
        if self._center_priority:
            return max(3, user_num_poses)
        return user_num_poses

    def set_smoothing_alpha(self, alpha: float) -> None:
        """EMA の追従係数を変更する。"""
        self._smoothing_alpha = max(0.05, min(1.0, alpha))
        logger.info(f"smoothing_alpha={self._smoothing_alpha}")

    @staticmethod
    def _sanitize_area(area: tuple[float, float, float, float]
                       ) -> tuple[float, float, float, float]:
        """検出エリアを [0,1] に収め、w/h が正になるよう整える。"""
        x, y, w, h = area
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        w = max(0.01, min(1.0 - x, float(w)))
        h = max(0.01, min(1.0 - y, float(h)))
        return (x, y, w, h)

    def set_mask_area(self, x: float, y: float, w: float, h: float) -> None:
        """① 入力マスク領域（画像正規化 x,y,w,h）を変更する。"""
        with self._lock:
            self._mask_area = self._sanitize_area((x, y, w, h))
            self._smoothed.clear()
        logger.info(f"mask_area={self._mask_area}")

    def set_filter_area(self, x: float, y: float, w: float, h: float) -> None:
        """② 検出後フィルタ領域（画像正規化 x,y,w,h）を変更する。"""
        with self._lock:
            self._filter_area = self._sanitize_area((x, y, w, h))
            self._smoothed.clear()
        logger.info(f"filter_area={self._filter_area}")

    @property
    def mask_area(self) -> tuple[float, float, float, float]:
        return self._mask_area

    @property
    def filter_area(self) -> tuple[float, float, float, float]:
        return self._filter_area

    @staticmethod
    def _mask_frame_to_area(rgb, area: tuple[float, float, float, float]):
        """① 入力マスク：エリア矩形の外側を黒 (0) で塗りつぶす（in-place）。
        MediaPipe にエリア外を見せないことで、端の人物をそもそも検出させない。
        area が全画面 (0,0,1,1) なら何もしない。rgb は (H, W, 3) を想定。
        """
        ax, ay, aw, ah = area
        if (ax, ay, aw, ah) == (0.0, 0.0, 1.0, 1.0):
            return rgb
        h, w = rgb.shape[:2]
        x0 = max(0, min(w, int(round(ax * w))))
        x1 = max(0, min(w, int(round((ax + aw) * w))))
        y0 = max(0, min(h, int(round(ay * h))))
        y1 = max(0, min(h, int(round((ay + ah) * h))))
        if x0 > 0:
            rgb[:, :x0] = 0
        if x1 < w:
            rgb[:, x1:] = 0
        if y0 > 0:
            rgb[:y0, :] = 0
        if y1 < h:
            rgb[y1:, :] = 0
        return rgb

    _FILTER_VIS = 0.5   # 重心計算に使う可視ランドマークの visibility 閾値

    @classmethod
    def _person_centroid(cls, plm):
        """人物の代表点＝可視ランドマークの重心 (x, y) を返す。
        腰など特定点だと上半身しか映らない時に破綻するので、可視点全体の平均を使う。
        可視点が無ければ全点平均、ランドマーク自体が無ければ None。
        """
        xs, ys = [], []
        for lm in plm:
            v = lm.visibility if lm.visibility is not None else 0.0
            if v >= cls._FILTER_VIS:
                xs.append(lm.x)
                ys.append(lm.y)
        if not xs:   # 可視点なし → 全点平均で代替
            xs = [lm.x for lm in plm]
            ys = [lm.y for lm in plm]
        if not xs:
            return None
        return sum(xs) / len(xs), sum(ys) / len(ys)

    @classmethod
    def _filter_by_area(cls, pose_landmarks_list, world_lists,
                        area: tuple[float, float, float, float]):
        """② 検出後フィルタ：各人物の代表点（可視ランドマークの重心）が area 矩形外の
        人物を除外する。
        戻り値: (filtered_pose_list, filtered_world_list, changed)。
        area が全画面 (0,0,1,1) なら素通り（changed=False）。
        """
        ax, ay, aw, ah = area
        if (ax, ay, aw, ah) == (0.0, 0.0, 1.0, 1.0):
            return pose_landmarks_list, world_lists, False
        kept = []
        for i, plm in enumerate(pose_landmarks_list):
            if not plm:
                continue
            c = cls._person_centroid(plm)
            if c is None:
                continue
            cx, cy = c
            if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                kept.append(i)
        if len(kept) == len(pose_landmarks_list):
            return pose_landmarks_list, world_lists, False
        filtered_pl = [pose_landmarks_list[i] for i in kept]
        filtered_wl = [world_lists[i] for i in kept if i < len(world_lists)]
        return filtered_pl, filtered_wl, True

    @property
    def smoothing_alpha(self) -> float:
        return self._smoothing_alpha

    @property
    def num_poses(self) -> int:
        return self._num_poses

    def _create_landmarker(self, num_poses: int):
        base_options = mp_python.BaseOptions(model_asset_path=self._model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    def set_num_poses(self, num_poses: int) -> None:
        """num_poses を変更する。PoseLandmarker を再作成するため 1〜2 秒程度ブロックする。
        estimate と排他されるので、推定スレッドはこの間ブロックされる。
        """
        n = max(1, min(10, int(num_poses)))
        with self._lock:
            if n == self._num_poses:
                return
            try:
                self._landmarker.close()
            except Exception as e:
                logger.warning(f"古い PoseLandmarker close 失敗: {e}")
            self._landmarker = self._create_landmarker(self._effective_num_poses(n))
            self._num_poses = n
            # トラッキング連続性を切る
            self._frame_timestamp_ms += 10_000
            self._smoothed.clear()
            logger.info(f"PoseLandmarker 再作成: num_poses={n}")

    def estimate(self, frame: np.ndarray) -> list[PoseLandmarkResult]:
        """BGRフレームを受け取り、全人物のランドマークリストを返す。
        検出できない場合は空リストを返す。

        Args:
            frame : OpenCV BGR フォーマットの numpy 配列

        Returns:
            検出した人物ごとの PoseLandmarkResult のリスト
        """
        # set_num_poses 中に landmarker を作り替えるので、推論～smoothing 全体を排他
        with self._lock:
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
            # ① 入力マスク：エリア外を黒塗り（rgb_frame は反転コピーなので
            #    in-place で塗っても表示用の元 frame には影響しない）。
            self._mask_frame_to_area(rgb_frame, self._mask_area)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self._frame_timestamp_ms += 33  # 約30fps想定
            detection_result = self._landmarker.detect_for_video(
                mp_image, self._frame_timestamp_ms
            )

            if not detection_result.pose_landmarks:
                self._smoothed.clear()
                return []

            results: list[PoseLandmarkResult] = []
            active_keys: set[int] = set()
            a = self._smoothing_alpha
            world_lists = detection_result.pose_world_landmarks or []

            pose_landmarks_list = list(detection_result.pose_landmarks)
            world_lists = list(world_lists)

            # 検出エリア外の人物を除外（代表点=NOSE がエリア矩形の外なら捨てる）。
            # 画面端に写り込んだ別人を判定対象から外す。全画面(0,0,1,1)なら素通り。
            pose_landmarks_list, world_lists, changed = self._filter_by_area(
                pose_landmarks_list, world_lists, self._filter_area
            )
            if changed:
                # 集合が変わったフレームは EMA を新規扱いにする
                self._smoothed.clear()
            if not pose_landmarks_list:
                self._smoothed.clear()
                return []

            # center_priority: 画面中央（0.5, 0.5）に NOSE が近い順にソートしてから
            # num_poses に絞る。外縁部にちらっと写った人を選ばせない用途向け。
            if self._center_priority and len(pose_landmarks_list) > self._num_poses:
                def _center_dist(pose_lm):
                    if not pose_lm:
                        return float("inf")
                    nose = pose_lm[0]  # index 0 = NOSE
                    dx = nose.x - 0.5
                    dy = nose.y - 0.5
                    return dx * dx + dy * dy
                order = sorted(
                    range(len(pose_landmarks_list)),
                    key=lambda i: _center_dist(pose_landmarks_list[i]),
                )[: self._num_poses]
                pose_landmarks_list = [pose_landmarks_list[i] for i in order]
                world_lists = [
                    world_lists[i] for i in order if i < len(world_lists)
                ]
                # トラッキング連続性は person_index に紐付いているので、順序が
                # 変わったフレームは EMA を新規扱いにする。
                self._smoothed.clear()

            for person_index, pose_landmarks in enumerate(pose_landmarks_list):
                raw_image = [
                    Landmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility if lm.visibility is not None else 0.0,
                    )
                    for lm in pose_landmarks
                ]
                raw_world: list[Landmark] = []
                if person_index < len(world_lists):
                    raw_world = [
                        Landmark(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=lm.visibility if lm.visibility is not None else 0.0,
                        )
                        for lm in world_lists[person_index]
                    ]

                prev = self._smoothed.get(person_index)
                if prev is None or len(prev[0]) != len(raw_image):
                    smoothed_image = raw_image
                    smoothed_world = raw_world
                else:
                    prev_image, prev_world = prev
                    smoothed_image = [
                        Landmark(
                            x=a * new.x + (1 - a) * old.x,
                            y=a * new.y + (1 - a) * old.y,
                            z=a * new.z + (1 - a) * old.z,
                            visibility=new.visibility,
                        )
                        for new, old in zip(raw_image, prev_image)
                    ]
                    if raw_world and len(prev_world) == len(raw_world):
                        smoothed_world = [
                            Landmark(
                                x=a * new.x + (1 - a) * old.x,
                                y=a * new.y + (1 - a) * old.y,
                                z=a * new.z + (1 - a) * old.z,
                                visibility=new.visibility,
                            )
                            for new, old in zip(raw_world, prev_world)
                        ]
                    else:
                        smoothed_world = raw_world

                self._smoothed[person_index] = (smoothed_image, smoothed_world)
                active_keys.add(person_index)
                results.append(PoseLandmarkResult(
                    landmarks=smoothed_image,
                    world_landmarks=smoothed_world,
                    person_index=person_index,
                ))

            for key in list(self._smoothed.keys()):
                if key not in active_keys:
                    del self._smoothed[key]

            return results

    def reset_timestamp(self) -> None:
        """フレームタイムスタンプを大きく前進させ、平滑化状態もクリアする。
        MediaPipe VIDEO モードは単調増加を求めるため 0 には戻さず、
        10 秒分ジャンプさせて前ソースとの連続性を切る。
        映像ソース切替（カメラ⇄動画）時に呼ぶ。
        """
        self._frame_timestamp_ms += 10_000
        self._smoothed.clear()
        logger.info("PoseEstimator タイムスタンプを 10s ジャンプし、平滑化状態をクリアしました。")

    def release(self) -> None:
        """MediaPipe のリソースを解放する。アプリ終了時に呼ぶ。"""
        self._landmarker.close()
        logger.info("PoseEstimator リソースを解放しました。")


if __name__ == "__main__":
    import sys
    import time
    import os

    # プロジェクトルート（app/の親）をカレントディレクトリに設定
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"作業ディレクトリ: {os.getcwd()}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # モデルファイルの候補を自動検索
    default_candidates = [
        "assets/models/pose_landmarker.task",
        "assets/models/pose_landmarker_heavy.task",
        "assets/models/pose_landmarker_full.task",
        "assets/models/pose_landmarker_lite.task",
    ]
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if model_path is None:
        for candidate in default_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path is None:
        print("❌ モデルファイルが見つかりません。")
        print("以下のいずれかを assets/models/ に配置してください：")
        for c in default_candidates:
            print(f"  {c}")
        sys.exit(1)

    print(f"PoseEstimator テスト開始 (model_path={model_path})")

    # PoseEstimator 初期化
    try:
        t0 = time.perf_counter()
        estimator = PoseEstimator(
            model_path=model_path,
            num_poses=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        t1 = time.perf_counter()
        print(f"初期化時間: {t1 - t0:.2f} 秒")
    except Exception as e:
        print(f"❌ PoseEstimator 初期化失敗: {e}")
        sys.exit(1)

    # カメラからフレームを取得して推定
    sys.path.insert(0, ".")
    from app.camera import Camera, CameraNotFoundError

    cam = Camera(device_index=0, width=1280, height=720, fps=30)
    try:
        cam.start()
        print("カメラ起動完了。10フレーム分の骨格推定を試みます...")
        print("カメラの前に立ってください。")

        success = 0
        for i in range(10):
            frame = cam.read_frame()
            if frame is None:
                print(f"  フレーム {i+1}/10 : フレーム取得失敗")
                continue

            t2 = time.perf_counter()
            results = estimator.estimate(frame)
            t3 = time.perf_counter()
            elapsed_ms = (t3 - t2) * 1000

            if results:
                p = results[0]
                nose = p.landmarks[0]
                print(
                    f"  フレーム {i+1}/10 : {len(results)}人検出 "
                    f"鼻座標=({nose.x:.3f}, {nose.y:.3f}) "
                    f"推定時間={elapsed_ms:.1f}ms"
                )
                success += 1
            else:
                print(f"  フレーム {i+1}/10 : 人物未検出 推定時間={elapsed_ms:.1f}ms")

        print(f"\n結果: 10フレーム中 {success} フレームで人物検出")
        if success > 0:
            print("✅ PoseEstimator OK")
        else:
            print("⚠️  人物が検出されませんでした（カメラの前に立って再試行してください）")

    except CameraNotFoundError as e:
        print(f"❌ カメラエラー: {e}")
    finally:
        cam.release()
        estimator.release()
        print("リソースを解放しました。")
