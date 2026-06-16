"""
video_export.py
入力動画を頭から読み、各フレームに対して MediaPipe で推定、Mode2 の見た目を
QOpenGLFramebufferObject にオフスクリーンレンダリングして cv2.VideoWriter で
MP4 に書き出す。

メインスレッドで処理する（GLWidget の OpenGL context を借りるため）。
1 フレーム書き終えるごとに QApplication.processEvents() で UI 応答性を保つ。
"""

from __future__ import annotations
import logging
import cv2
import numpy as np
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QOpenGLContext, QImage, QPainter
from PyQt6.QtOpenGL import QOpenGLFramebufferObject
from OpenGL.GL import glViewport, glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT

logger = logging.getLogger(__name__)


class ExportCancelled(Exception):
    """ユーザーがエクスポートをキャンセルした際に送出する。"""


class VideoExporter:
    """Mode2 描画を動画全フレームに対し処理して MP4 出力する。"""

    def __init__(self, input_path: str, output_path: str,
                 estimator, mode2,
                 gl_widget,
                 progress_cb=None,
                 cancel_check=None,
                 max_frames: int | None = None,
                 graph_widgets=None,
                 graph_update_cb=None) -> None:
        """
        Args:
            input_path      : 入力動画ファイルのパス
            output_path     : 出力 MP4 ファイルのパス
            estimator       : PoseEstimator インスタンス
            mode2           : Mode2Mannequin インスタンス（draw メソッドを使う）
            gl_widget       : GLWidget（OpenGL context を借りる）
            progress_cb     : callable(current_frame, total_frames) — 進捗通知
            cancel_check    : callable() -> bool — True を返したらキャンセル
            max_frames      : 書き出す最大フレーム数（None で全フレーム）
            graph_widgets   : 動画右上に上から順に重ねる QWidget のリスト
                              （None または空で焼き込まない）。
                              isVisible() が False のものはスキップ。
            graph_update_cb : callable(results, t_video_sec) — 毎フレーム呼ばれ、
                              書出側でグラフバッファを進めるためのフック。
        """
        self._input_path = input_path
        self._output_path = output_path
        self._estimator = estimator
        self._mode2 = mode2
        self._gl_widget = gl_widget
        self._progress_cb = progress_cb or (lambda c, t: None)
        self._cancel_check = cancel_check or (lambda: False)
        self._max_frames = max_frames
        self._graph_widgets = list(graph_widgets) if graph_widgets else []
        self._graph_update_cb = graph_update_cb

    def run(self) -> tuple[int, int]:
        """エクスポート実行。(処理済みフレーム数, 全フレーム数) を返す。"""
        cap = cv2.VideoCapture(self._input_path)
        if not cap.isOpened():
            raise RuntimeError(f"入力動画を開けません: {self._input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # max_frames が指定されていれば、進捗・終了判定用の総数を上書き
        effective_total = total if self._max_frames is None else min(total or self._max_frames, self._max_frames)
        logger.info(
            f"VideoExporter 開始: {self._input_path} "
            f"{width}x{height} {fps:.2f}fps total={total} "
            f"target={effective_total} → {self._output_path}"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self._output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"出力動画を作成できません: {self._output_path}")

        # 書出中は GLWidget の自動再描画を止める。
        # processEvents 中に GLWidget.paintEvent が走ると context が切り替わって
        # FBO バインド状態や texture が壊れ、数フレームで描画破綻するため。
        # 注意: setUpdatesEnabled は子ウィジェットにも伝播するので、
        # grab() の対象となるグラフは個別に updates を復活させる。
        gl_updates_were_enabled = self._gl_widget.updatesEnabled()
        self._gl_widget.setUpdatesEnabled(False)
        graph_prev_updates = []
        for w in self._graph_widgets:
            if w is None:
                graph_prev_updates.append(None)
                continue
            graph_prev_updates.append(w.updatesEnabled())
            w.setUpdatesEnabled(True)
        logger.info(
            f"VideoExporter graphs={len(self._graph_widgets)} "
            f"visible=[{','.join(str(w.isVisible() if w else False) for w in self._graph_widgets)}] "
            f"sizes=[{','.join(f'{w.width()}x{w.height()}' if w else '-' for w in self._graph_widgets)}]"
        )

        # GLWidget の OpenGL context を借りて、メインスレッドで描画
        self._gl_widget.makeCurrent()
        try:
            fbo = QOpenGLFramebufferObject(
                width, height,
                QOpenGLFramebufferObject.Attachment.CombinedDepthStencil,
            )

            frame_idx = 0
            try:
                while True:
                    if self._cancel_check():
                        raise ExportCancelled()
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = self._estimator.estimate(frame)

                    # グラフバッファ更新（ライブと違い worker emit が止まっているので自前で）
                    # 動画内時刻 = フレーム番号 / 元動画 fps
                    if self._graph_update_cb is not None:
                        try:
                            self._graph_update_cb(results, frame_idx / fps)
                        except Exception as e:
                            logger.warning(f"graph_update_cb failed: {e}")

                    # 念のため毎フレーム context を確保
                    # （processEvents 等で context が動かされても復旧できる）
                    self._gl_widget.makeCurrent()

                    # FBO にバインドして Mode2 を描画
                    fbo.bind()
                    try:
                        glViewport(0, 0, width, height)
                        glClearColor(0.0, 0.0, 0.0, 1.0)
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                        # Mode2 の draw は frame, results, width, height を受け取る
                        self._mode2.draw(frame, results, width, height)
                        # Bキー有効時はボーンオーバーレイを重ねる（ライブ表示と同様）
                        if getattr(self._gl_widget, "show_bones", False):
                            self._gl_widget.draw_bone_overlay(
                                results, frame, width, height
                            )
                    finally:
                        fbo.release()

                    # FBO → QImage → numpy(BGR) → VideoWriter
                    img = fbo.toImage()
                    arr = _qimage_to_bgr_array(img)
                    # グラフを右上にアルファ合成（ライブ表示と同様レイアウト）
                    if self._graph_widgets:
                        _overlay_graphs(arr, self._graph_widgets)
                    writer.write(arr)

                    frame_idx += 1
                    self._progress_cb(frame_idx, effective_total)
                    # UI 応答性のため定期的にイベントを処理
                    # （GLWidget の paint は setUpdatesEnabled(False) で抑制済み）
                    if frame_idx % 5 == 0:
                        QCoreApplication.processEvents()
                    # max_frames に到達したら終了
                    if self._max_frames is not None and frame_idx >= self._max_frames:
                        break
            finally:
                fbo = None
        finally:
            self._gl_widget.doneCurrent()
            self._gl_widget.setUpdatesEnabled(gl_updates_were_enabled)
            for w, prev in zip(self._graph_widgets, graph_prev_updates):
                if w is not None and prev is not None:
                    w.setUpdatesEnabled(prev)
            cap.release()
            writer.release()

        logger.info(f"VideoExporter 完了: {frame_idx}/{effective_total} フレーム書き出し")
        return frame_idx, effective_total


def _qimage_to_bgra_array(img: QImage) -> np.ndarray:
    """QImage を numpy BGRA 配列に変換する。アルファ合成用。"""
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    ptr = img.bits()
    ptr.setsize(h * bpl)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl // 4, 4)
    rgba = arr[:, :w, :]
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return np.ascontiguousarray(bgra)


def _render_graph_to_qimage(w) -> QImage | None:
    """TimeSeriesGraph（pg.PlotWidget をラップした QWidget）を QImage に描画する。
    QOpenGLWidget の子 QWidget は grab() がうまく動かない既知の挙動があるため、
    PlotWidget（QGraphicsView 継承）の render(painter) で viewport 全体を描画する。
    pyqtgraph.GraphicsScene は QGraphicsScene.render の keyword 引数を受け付けないため、
    scene.render ではなく QGraphicsView.render を使う。
    """
    plot = getattr(w, "_plot", None) or w
    if not hasattr(plot, "render") or not hasattr(plot, "viewport"):
        return None
    gw, gh = w.width(), w.height()
    if gw <= 0 or gh <= 0:
        return None
    img = QImage(gw, gh, QImage.Format.Format_ARGB32)
    img.fill(0)  # 透明
    painter = QPainter(img)
    try:
        # QGraphicsView.render(painter) は viewport を target rect に描画する
        plot.render(painter)
    finally:
        painter.end()
    return img


def _overlay_graphs(arr: np.ndarray, widgets) -> None:
    """動画フレーム（BGR）の右上にグラフ widget を貼り付ける。
    QGraphicsScene を直接 render することで QOpenGLWidget の子問題を回避。
    ライブ表示のレイアウト（右上、上から順、間隔 8px）に合わせる。
    arr は in-place で書き換える。
    """
    margin = 10
    spacing = 8
    fh, fw = arr.shape[:2]
    y_off = margin
    for w in widgets:
        if w is None or not w.isVisible():
            continue
        gw_req, gh_req = w.width(), w.height()
        if gw_req <= 0 or gh_req <= 0:
            continue
        gimg = _render_graph_to_qimage(w)
        if gimg is None or gimg.isNull():
            logger.warning(f"graph render returned null ({gw_req}x{gh_req})")
            continue
        bgr = _qimage_to_bgr_array(gimg)
        gh, gw = bgr.shape[0], bgr.shape[1]
        # 配置位置（右上、frame 範囲内にクリップ）
        x = fw - gw - margin
        y = y_off
        if x < 0 or y < 0 or x + gw > fw or y + gh > fh:
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            gw = min(gw, fw - x)
            gh = min(gh, fh - y)
            bgr = bgr[:gh, :gw]
        if gw <= 0 or gh <= 0:
            continue
        arr[y:y + gh, x:x + gw] = bgr
        y_off += gh + spacing


def _qimage_to_bgr_array(img: QImage) -> np.ndarray:
    """QImage を numpy BGR 配列に変換する。
    QOpenGLFramebufferObject.toImage() は内部で OpenGL の Y 軸反転を
    補正済みの QImage（上が +Y）を返すため、ここでは追加の flipud は
    行わない（行うと天地逆になる）。
    bytesPerLine が w*4 と一致しない（4 バイト境界 padding）ケースも
    考慮して構築する。
    """
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    ptr = img.bits()
    ptr.setsize(h * bpl)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl // 4, 4)
    # padding 列がある場合は w 列だけ採用
    rgba = arr[:, :w, :]
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(bgr)
