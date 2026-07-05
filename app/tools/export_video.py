"""
export_video.py
CLI から Mode2 の動画書き出しを行う。ウィンドウは表示せず (WA_DontShowOnScreen)、
既存の VideoExporter をそのまま流用する。
user_settings.json / config.yaml の永続化パラメータを使うので、GUI で好みの
見た目に調整して閉じておけば、その設定で複数動画を一括処理できる。

Usage:
    python -m app.tools.export_video INPUT.mp4
    python -m app.tools.export_video INPUT.mp4 -o OUT.mp4
    python -m app.tools.export_video INPUT.mp4 --sample
    python -m app.tools.export_video INPUT.mp4 --frames 500
    python -m app.tools.export_video INPUT.mp4 --no-audio
    for f in *.mp4; do python -m app.tools.export_video "$f"; done
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# `python -m app.tools.export_video` でなくても動くよう、リポジトリルートを path に。
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)


# main.py と同じ候補で自動フォールバック
DEFAULT_MODEL_CANDIDATES = [
    "assets/models/pose_landmarker.task",
    "assets/models/pose_landmarker_heavy.task",
    "assets/models/pose_landmarker_full.task",
    "assets/models/pose_landmarker_lite.task",
]


def _resolve_model(config_model_path: str) -> str:
    candidates = [config_model_path] + DEFAULT_MODEL_CANDIDATES
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"MediaPipe モデルが見つかりません。試した候補: {candidates}"
    )


def _default_output_path(input_path: str, is_sample: bool) -> str:
    """--output 省略時のデフォルト。<name>_export.mp4 / <name>_sample.mp4。"""
    p = Path(input_path)
    suffix = "_sample" if is_sample else "_export"
    return str(p.with_stem(p.stem + suffix).with_suffix(".mp4"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.tools.export_video",
        description=(
            "ヘッドレスで Mode2 の動画書き出しを行う。GUI ウィンドウは表示しない。"
            " user_settings.json / config.yaml の永続化パラメータを流用する。"
        ),
    )
    p.add_argument("input", help="入力動画ファイル (mp4/mov/avi/mkv/webm)")
    p.add_argument("-o", "--output", default=None,
                   help="出力 MP4 パス。省略で <input>_export.mp4 "
                        "（--sample 時は <input>_sample.mp4）")
    p.add_argument("--sample", action="store_true",
                   help="先頭 300 フレームだけ書き出す（確認用）")
    p.add_argument("--frames", type=int, default=None,
                   help="任意フレーム数まで書き出す（--sample より優先）")
    p.add_argument("--no-audio", action="store_true",
                   help="ffmpeg 音声 mux をスキップして無音で出力")
    p.add_argument("--csv", default=None,
                   help="動画と同時に CSV も書き出す（フレームごとのランドマーク）。"
                        "同時実行なので MediaPipe 推定コストは 1 回で済む。"
                        "指定時は smoothing_alpha=1.0（生値）に強制される。")
    p.add_argument("--csv-format", choices=["long", "wide"], default="wide",
                   help="--csv 時の出力形式（default: wide、Excel 向け）")
    p.add_argument("--config", default="config.yaml",
                   help="システム設定ファイル（default: config.yaml）")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG ログ")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.exists(args.input):
        logger.error(f"入力動画が見つかりません: {args.input}")
        return 2

    # 出力パス決定
    out_path = args.output if args.output else _default_output_path(args.input, args.sample)

    # フレーム制限（--frames が最優先、次に --sample、無指定なら全フレーム）
    if args.frames is not None:
        max_frames: int | None = args.frames
    elif args.sample:
        max_frames = 300
    else:
        max_frames = None

    logger.info(f"入力: {args.input}")
    logger.info(f"出力: {out_path}")
    logger.info(f"フレーム制限: {max_frames if max_frames is not None else '（全フレーム）'}")

    # ---- QApplication をまず立てる（ヘッドレスモード）----
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # 依存モジュールは QApplication 後にインポート
    from app.config_loader import ConfigLoader
    from app.camera import Camera, SourceOpenError
    from app.pose_estimator import PoseEstimator
    from app.main_window import MainWindow
    from app.video_export import VideoExporter

    # ---- Config ----
    try:
        config = ConfigLoader(args.config)
    except Exception as e:
        logger.error(f"設定ファイル読込失敗 ({args.config}): {e}")
        return 2

    # ---- Camera を動画ファイルで開く（start() は使わずに switch_source）----
    camera = Camera(
        device_index=config.get("camera.device_index", 0),
        width=config.get("camera.width", 1280),
        height=config.get("camera.height", 720),
        fps=config.get("camera.fps", 30),
    )
    try:
        camera.switch_source(args.input)
    except SourceOpenError as e:
        logger.error(f"動画を開けません: {e}")
        return 3

    # ---- PoseEstimator ----
    model_path = _resolve_model(
        config.get("pose.model_path", "assets/models/pose_landmarker.task")
    )
    logger.info(f"モデル: {model_path}")
    # --csv 指定時は生値優先（EMA を効かせずに CSV を出す）。
    # 平滑化を後処理でかけたいユースケースが多いので、この方が扱いやすい。
    smoothing_alpha = 1.0 if args.csv else config.get("pose.smoothing_alpha", 0.25)
    estimator = PoseEstimator(
        model_path=model_path,
        num_poses=config.get("pose.num_poses", 1),
        min_detection_confidence=config.get("pose.min_detection_confidence", 0.5),
        min_tracking_confidence=config.get("pose.min_tracking_confidence", 0.5),
        smoothing_alpha=smoothing_alpha,
        center_priority=config.get("pose.center_priority", False),
    )

    # ---- MainWindow をヘッドレスに作って GL context を確立 ----
    # WA_DontShowOnScreen: 画面には出さないが Qt 内部では show 扱いになり、
    # QOpenGLWidget.initializeGL() が呼ばれる。
    mw = MainWindow(config, camera, estimator)
    try:
        mw.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    except Exception:
        pass
    mw.show()
    # イベントループを回して initializeGL を走らせる
    for _ in range(3):
        app.processEvents()

    # 書き出しは自前で動画フレームを回すので、ライブ用の worker は止める
    try:
        mw._worker.stop()
    except Exception as e:
        logger.warning(f"worker.stop 失敗: {e}")

    # Mode2 に切替（Mode2 が生成されて AppSettings が renderer に反映される）
    if mw._current_mode_id != 2:
        mw.switch_mode(2)
        for _ in range(3):
            app.processEvents()

    if mw._mode2 is None:
        logger.error("Mode2 の初期化に失敗しました")
        mw.close()
        return 4

    # ---- 書出実行 ----
    last_report = [0.0]

    def on_progress(cur: int, total: int) -> None:
        # 0.3 秒に 1 回だけ更新
        now = time.perf_counter()
        if now - last_report[0] < 0.3 and cur != total:
            return
        last_report[0] = now
        if total > 0:
            pct = cur * 100 / max(total, 1)
            print(f"\r  {cur}/{total} ({pct:5.1f}%)  ", end="", flush=True)
        else:
            print(f"\r  {cur} frames", end="", flush=True)

    graph_widgets = None
    if mw._graph_x is not None:
        graph_widgets = [mw._graph_x, mw._graph_y]

    # --- CSV 併走の準備（--csv 指定時のみ）---
    # graph_update_cb は VideoExporter が毎フレーム呼ぶフック。
    # ここに CSV writer をぶら下げて、同じ推定結果を CSV にも書き出す。
    # MediaPipe 推定は 1 回で済むので extract_csv を別プロセスで走らせる
    # よりトータル約 2 倍速い。
    csv_file = None
    csv_writer = None
    csv_write_rows = None
    csv_frame_idx = [0]
    if args.csv:
        import csv as csv_mod
        from app.tools.extract_csv import (
            _write_long_header, _write_long_rows,
            _write_wide_header, _write_wide_rows,
        )
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv_mod.writer(csv_file)
        if args.csv_format == "wide":
            _write_wide_header(csv_writer)
            csv_write_rows = _write_wide_rows
        else:
            _write_long_header(csv_writer)
            csv_write_rows = _write_long_rows
        logger.info(f"CSV も併走で書き出し: {args.csv} (format={args.csv_format})")

    original_graph_cb = mw._append_graphs_for_export

    def combined_cb(results, t_video: float) -> None:
        # まずは既存のグラフ更新
        original_graph_cb(results, t_video)
        # CSV 併走
        if csv_writer is not None:
            if results:
                csv_write_rows(csv_writer, csv_frame_idx[0], t_video, results)
            csv_frame_idx[0] += 1

    t_start = time.perf_counter()
    exporter = VideoExporter(
        input_path=args.input,
        output_path=out_path,
        estimator=estimator,
        mode2=mw._mode2,
        gl_widget=mw._gl_widget,
        progress_cb=on_progress,
        cancel_check=lambda: False,
        max_frames=max_frames,
        graph_widgets=graph_widgets,
        graph_update_cb=combined_cb,
    )
    # --no-audio: 音声 mux をスキップ（メソッド差し替え）
    if args.no_audio:
        exporter._mux_audio_from_source = lambda *a, **kw: False

    try:
        written, total = exporter.run()
    except Exception as e:
        logger.exception(f"書出エラー: {e}")
        if csv_file is not None:
            csv_file.close()
        mw.close()
        estimator.release()
        camera.release()
        return 5

    if csv_file is not None:
        csv_file.close()
        logger.info(f"CSV 完了: {csv_frame_idx[0]} frames → {args.csv}")

    elapsed = time.perf_counter() - t_start
    print(f"\r完了: {written}/{total} frames / {elapsed:.1f}s → {out_path}")

    # クリーンアップ
    mw.close()
    estimator.release()
    camera.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
