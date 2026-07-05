"""
extract_csv.py
mp4 動画から MediaPipe Pose のランドマークを抽出して CSV に書き出す CLI ツール。
GUI は起動しない（PyQt6 に依存しない）。

使い方:
    python -m app.tools.extract_csv INPUT.mp4 OUTPUT.csv [options]

例:
    python -m app.tools.extract_csv sample.mp4 sample.csv
    python -m app.tools.extract_csv sample.mp4 sample.csv --format wide
    python -m app.tools.extract_csv sample.mp4 sample.csv --num-poses 2

出力データは MediaPipe の raw 出力（EMA 平滑化は適用しない）。
平滑化は後処理で任意に適用できる。
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import time

import cv2

# app モジュールを import できるようリポジトリルートを sys.path に入れる
# （`python -m app.tools.extract_csv` なら不要だが、直接呼びでも動くようにする）
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)


# MediaPipe Pose Landmarker (33 点) のインデックス→名前対応
LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# main.py と同じ候補で自動フォールバックする
DEFAULT_MODEL_CANDIDATES = [
    "assets/models/pose_landmarker.task",
    "assets/models/pose_landmarker_heavy.task",
    "assets/models/pose_landmarker_full.task",
    "assets/models/pose_landmarker_lite.task",
]


# --- ヘルパ ------------------------------------------------------------------

def _resolve_model(explicit: str | None) -> str:
    """--model 指定があればそれ、なければ既知の候補から先に見つかったものを返す。"""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"指定モデルが見つかりません: {explicit}")
        return explicit
    for c in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "MediaPipe モデルファイルが見つかりません。"
        f"以下のいずれかを配置するか --model で指定してください: {DEFAULT_MODEL_CANDIDATES}"
    )


def _open_video(path: str) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total


def _format_eta(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


# --- CSV 出力形式 ------------------------------------------------------------
# Long: 1 行 = 1 ランドマーク（複数人・後処理・pandas 向け）
# Wide: 1 行 = 1 人物、列がランドマーク別（Excel・単一人物向け）

def _write_long_header(w: csv.writer) -> None:
    w.writerow([
        "frame_idx", "t_sec", "person", "lm_idx", "lm_name",
        # 画像座標系（x,y: 正規化 [0,1]、z: 腰基準の推定奥行き）
        "x", "y", "z", "visibility",
        # World 座標系（腰原点・メートル、y は下が +）
        "wx", "wy", "wz", "w_visibility",
    ])


def _write_long_rows(w: csv.writer, frame_idx: int, t_sec: float, results) -> int:
    """1 行 = 1 ランドマーク。書いた行数を返す。"""
    n_written = 0
    for r in results:
        n_lm = len(r.landmarks)
        n_wlm = len(r.world_landmarks)
        for i in range(n_lm):
            lm = r.landmarks[i]
            wlm = r.world_landmarks[i] if i < n_wlm else None
            row = [
                frame_idx, f"{t_sec:.4f}", r.person_index, i,
                LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"LM_{i}",
                f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}", f"{lm.visibility:.4f}",
            ]
            if wlm is not None:
                row.extend([
                    f"{wlm.x:.6f}", f"{wlm.y:.6f}", f"{wlm.z:.6f}", f"{wlm.visibility:.4f}",
                ])
            else:
                row.extend(["", "", "", ""])
            w.writerow(row)
            n_written += 1
    return n_written


def _write_wide_header(w: csv.writer) -> None:
    header = ["frame_idx", "t_sec", "person"]
    for name in LANDMARK_NAMES:
        header.extend([
            f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis",
            f"{name}_wx", f"{name}_wy", f"{name}_wz", f"{name}_wvis",
        ])
    w.writerow(header)


def _write_wide_rows(w: csv.writer, frame_idx: int, t_sec: float, results) -> int:
    """1 行 = 1 人物。書いた行数を返す。"""
    n_written = 0
    for r in results:
        row: list = [frame_idx, f"{t_sec:.4f}", r.person_index]
        n_lm = len(r.landmarks)
        n_wlm = len(r.world_landmarks)
        for i in range(len(LANDMARK_NAMES)):
            if i < n_lm:
                lm = r.landmarks[i]
                row.extend([
                    f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}", f"{lm.visibility:.4f}",
                ])
            else:
                row.extend(["", "", "", ""])
            if i < n_wlm:
                wlm = r.world_landmarks[i]
                row.extend([
                    f"{wlm.x:.6f}", f"{wlm.y:.6f}", f"{wlm.z:.6f}", f"{wlm.visibility:.4f}",
                ])
            else:
                row.extend(["", "", "", ""])
        w.writerow(row)
        n_written += 1
    return n_written


# --- メイン ------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.tools.extract_csv",
        description="mp4 動画から MediaPipe Pose の各フレームランドマークを CSV に抽出する。",
    )
    p.add_argument("input", help="入力動画ファイル (mp4/mov/avi/mkv/webm など)")
    p.add_argument("output", help="出力 CSV ファイル")
    p.add_argument("--model", default=None,
                   help="MediaPipe モデルパス（省略時は assets/models/ から自動検索）")
    p.add_argument("--num-poses", type=int, default=1,
                   help="検出する最大人数（default: 1）")
    p.add_argument("--format", choices=["long", "wide"], default="long",
                   help="CSV 形式。long=1行1ランドマーク（複数人・pandas 向け, default）／"
                        "wide=1行1人物・列がランドマーク別（Excel 向け）")
    p.add_argument("--min-detection", type=float, default=0.5,
                   help="検出信頼度閾値（default: 0.5）")
    p.add_argument("--min-tracking", type=float, default=0.5,
                   help="トラッキング信頼度閾値（default: 0.5）")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="DEBUG レベルログを出力")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.exists(args.input):
        logger.error(f"入力ファイルが見つかりません: {args.input}")
        return 2

    try:
        model_path = _resolve_model(args.model)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 2
    logger.info(f"モデル: {model_path}")

    # 生値を出したいので smoothing_alpha=1.0（EMA を実質無効化 → new をそのまま採用）。
    estimator = PoseEstimator(
        model_path=model_path,
        num_poses=args.num_poses,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        smoothing_alpha=1.0,
    )

    cap, fps, total = _open_video(args.input)
    logger.info(
        f"入力: {args.input}  fps={fps:.2f}  total_frames={total}  format={args.format}"
    )

    write_header = _write_long_header if args.format == "long" else _write_wide_header
    write_rows = _write_long_rows if args.format == "long" else _write_wide_rows

    frame_idx = 0
    n_rows = 0
    t_start = time.perf_counter()
    last_report_t = t_start
    try:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            write_header(writer)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = estimator.estimate(frame)
                t_sec = frame_idx / max(fps, 1.0)
                if results:
                    n_rows += write_rows(writer, frame_idx, t_sec, results)
                frame_idx += 1

                # 進捗は 0.5 秒に 1 回だけ更新する（stdout flood を避ける）
                now = time.perf_counter()
                if now - last_report_t >= 0.5:
                    elapsed = now - t_start
                    if total > 0 and frame_idx > 0:
                        pct = frame_idx / total * 100
                        eta = elapsed / frame_idx * (total - frame_idx)
                        print(
                            f"\r  {frame_idx}/{total} ({pct:5.1f}%)  "
                            f"ETA {_format_eta(eta)}  ",
                            end="", flush=True,
                        )
                    else:
                        print(f"\r  {frame_idx} frames", end="", flush=True)
                    last_report_t = now

        elapsed = time.perf_counter() - t_start
        # \r で 1 行に上書きしていた進捗を改行して締める
        print(
            f"\r完了: {frame_idx} frames / {n_rows} rows / "
            f"{elapsed:.1f}s → {args.output}"
        )
    finally:
        cap.release()
        estimator.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
