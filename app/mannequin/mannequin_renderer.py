"""
mannequin_renderer.py
BlockMan.gltfを読み込み、骨格データに合わせてスキニング描画する。
正射影（モード2）と透視投影（モード3）を切り替え可能。
"""
from __future__ import annotations
import math
import logging
import numpy as np
from app.mannequin.gltf_loader import GLTFModel

from OpenGL.GL import (
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glEnable, glDisable, glColor3f, glColorMaterial,
    glClear, glLightfv, glMaterialfv, glNormal3f,
    glPushMatrix, glPopMatrix, glTranslatef, glRotatef, glScalef,
    glBegin, glEnd, glVertex3f,
    glEnableClientState, glDisableClientState,
    glVertexPointer, glNormalPointer, glDrawElements,
    GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL,
    GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR, GL_SHININESS,
    GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
    GL_PROJECTION, GL_MODELVIEW,
    GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
    GL_VERTEX_ARRAY, GL_NORMAL_ARRAY,
    GL_FLOAT, GL_UNSIGNED_INT, GL_TRIANGLES,
    GL_FRONT, GL_BACK, GL_NORMALIZE,
    glColor4f, glLineWidth, glPointSize, glBlendFunc,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINE_STRIP, GL_POINTS,
    GL_POINT_SMOOTH, GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT, GL_NICEST,
    glHint,
)
from OpenGL.GLU import gluPerspective
from app.mannequin.primitives import (
    draw_sphere, draw_capsule, draw_tapered_cylinder, draw_rounded_prism,
)
from app.pose_constants import PoseLandmark
from app.trails import TrailBuffer

logger = logging.getLogger(__name__)

MODEL_COLOR = (0.85, 0.82, 0.78)  # アイボリー
PRIMITIVE_COLOR = (0.95, 0.78, 0.55)  # 暖色オレンジ
HEAD_COLOR = (0.95, 0.85, 0.70)

# Mode3 透視投影の視点距離（メートル単位、world_landmarks 用）
# 視野は約 距離 * 2 * tan(22.5°) ≈ 距離 * 0.83
# 視点 2.5m → 縦視野 ≈ 2.07m で身長 1.7m を余裕で収める
VIEW_DISTANCE = 2.5

# プリミティブ人型描画の寸法
PRIM_JOINT_R = 0.030
PRIM_BONE_R = 0.022
PRIM_HEAD_R = 0.070
# 身長基準スケール：描画空間で身長を約 0.9 に揃える（視野 ±0.5〜±0.75 内に収める）
PRIM_HEIGHT_TARGET = 0.9
# 身長が取れない時のフォールバック：肩幅基準
PRIM_SHOULDER_FALLBACK = 0.40
PRIM_MIN_VIS = 0.4

# Mode2 raw 座標描画用（MediaPipe 0〜1 スケールに対するパーツ比率）
RAW_JOINT_R = 0.012   # 画像幅の 1.2%
RAW_BONE_R = 0.008    # 画像幅の 0.8%
RAW_HEAD_R = 0.030    # 画像幅の 3%
# world (m) サイズを Mode2 画像座標 [0,1] に換算する係数（元 0.4 から 0.7 倍 = 0.28）
RAW_SIZE_SCALE = 0.28

# 頭部の縦長楕円体
HEAD_RX = 0.075   # 横半径
HEAD_RY = 0.100   # 縦半径（長辺）
HEAD_RZ = 0.085   # 奥行き半径
NECK_R = 0.045    # 首の太さ
NECK_BASE_RATIO = 0.20  # 首根本：両肩中点 → 頭部中心 の何 % 頭側に上げるか

# 腰原点での足元 Y 座標（メートル、下が正）
WORLD_FOOT_Y = 0.95

# 部位別の関節半径（球サイズ、メートル）。隣接骨の端半径と揃えてある。
JOINT_RADII = {
    PoseLandmark.LEFT_SHOULDER:  0.045,
    PoseLandmark.RIGHT_SHOULDER: 0.045,
    PoseLandmark.LEFT_HIP:       0.042,
    PoseLandmark.RIGHT_HIP:      0.042,
    PoseLandmark.LEFT_ELBOW:     0.040,
    PoseLandmark.RIGHT_ELBOW:    0.040,
    PoseLandmark.LEFT_KNEE:      0.040,
    PoseLandmark.RIGHT_KNEE:     0.040,
    PoseLandmark.LEFT_WRIST:     0.035,
    PoseLandmark.RIGHT_WRIST:    0.035,
    PoseLandmark.LEFT_ANKLE:     0.035,
    PoseLandmark.RIGHT_ANKLE:    0.035,
}

# 骨（始点ランドマーク, 終点ランドマーク, 始点半径, 終点半径）。
# 接合点の関節半径と揃えて段差ゼロ。
# 胴 (LS-RS, LH-RH, LS-LH, RS-RH) は _draw_torso_core で 5 頂点版を描画するため、
# ここには含めない。
TAPERED_BONES = [
    # 腕（肩 0.045 → 肘 0.040 → 手首 0.035）
    (PoseLandmark.LEFT_SHOULDER,  PoseLandmark.LEFT_ELBOW,  0.045, 0.040),
    (PoseLandmark.LEFT_ELBOW,     PoseLandmark.LEFT_WRIST,  0.040, 0.035),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, 0.045, 0.040),
    (PoseLandmark.RIGHT_ELBOW,    PoseLandmark.RIGHT_WRIST, 0.040, 0.035),
    # 脚（腰 0.042 → 膝 0.040 → 足首 0.035）
    (PoseLandmark.LEFT_HIP,    PoseLandmark.LEFT_KNEE,   0.042, 0.040),
    (PoseLandmark.LEFT_KNEE,   PoseLandmark.LEFT_ANKLE,  0.040, 0.035),
    (PoseLandmark.RIGHT_HIP,   PoseLandmark.RIGHT_KNEE,  0.042, 0.040),
    (PoseLandmark.RIGHT_KNEE,  PoseLandmark.RIGHT_ANKLE, 0.040, 0.035),
]

# 指の骨（手首から指先、テーパード）
FINGER_BONES = [
    (PoseLandmark.LEFT_WRIST,  PoseLandmark.LEFT_INDEX,  0.014, 0.008),
    (PoseLandmark.LEFT_WRIST,  PoseLandmark.LEFT_PINKY,  0.014, 0.008),
    (PoseLandmark.LEFT_WRIST,  PoseLandmark.LEFT_THUMB,  0.012, 0.007),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX, 0.014, 0.008),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY, 0.014, 0.008),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB, 0.012, 0.007),
]

# 旧互換（raw 描画で参照する PRIM_JOINTS / PRIM_BONES を残す）
PRIM_JOINTS = list(JOINT_RADII.keys())
PRIM_BONES = [(a, b) for (a, b, _, _) in TAPERED_BONES]


class MannequinRenderer:
    """BlockMan GLTFモデルをスキニング描画する共用レンダラー。"""

    def __init__(self) -> None:
        self._model: GLTFModel | None = None
        self._rotation_y = 0.0
        # 描画スタイル: "primitive"（球＋カプセル・動く）or "mesh"（BlockMan・Tポーズ固定）
        self._style: str = "primitive"
        # マネキン表示倍率（マネキン自体のサイズ）
        self._scale_factor: float = 1.0
        # カメラ視点距離（メートル、Mode3 透視投影用）
        self._view_distance: float = VIEW_DISTANCE
        # Mode2 ortho 描画時の viewport アスペクト比（楕円体スケールの y 補正用、Mode3 では None）
        self._image_view_aspect: float | None = None
        # Mode2 マネキンサイズ係数（カメラと被写体の近さに応じてランタイム調整）
        self._raw_size_scale: float = RAW_SIZE_SCALE
        # トレイル（両手両足の軌跡）
        self._trail_buffer = TrailBuffer()
        self._trail_mode: str | None = None   # "ortho" or "perspective"
        self._trail_point_size: float = 6.0   # 点サイズ（px、0 で点描画オフ）
        self._trail_line_width: float = 3.0   # 線太さ（px、0 で線描画オフ）
        # マネキン本体の可視性。トレイル可視性とは独立で、両方任意の組合せ可。
        self._mannequin_visible: bool = True
        self._trail_visible: bool = True

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    def set_scale_factor(self, factor: float) -> None:
        self._scale_factor = max(0.3, min(3.0, float(factor)))

    def adjust_scale(self, delta: float) -> float:
        """スケール係数を増減する。delta>0 で拡大（近づく感覚）。"""
        self.set_scale_factor(self._scale_factor + delta)
        return self._scale_factor

    @property
    def view_distance(self) -> float:
        return self._view_distance

    def set_view_distance(self, d: float) -> None:
        """Mode3 カメラ視点距離（メートル）を設定。0.3〜20m でクランプ。"""
        self._view_distance = max(0.3, min(20.0, float(d)))

    def adjust_view_distance(self, delta: float) -> float:
        """カメラ視点距離を増減する。delta>0 で遠ざかる。"""
        self.set_view_distance(self._view_distance + delta)
        return self._view_distance

    @property
    def raw_size_scale(self) -> float:
        return self._raw_size_scale

    def set_raw_size_scale(self, scale: float) -> None:
        """Mode2 のマネキンサイズ係数を設定する。0.05〜1.0 でクランプ。"""
        self._raw_size_scale = max(0.05, min(1.0, float(scale)))

    # --- トレイル設定 -------------------------------------------------------

    def set_trail_point_size(self, size: float) -> None:
        """軌跡の点サイズ（px）。0 で点描画をオフ。"""
        self._trail_point_size = max(0.0, float(size))

    def set_trail_line_width(self, width: float) -> None:
        """軌跡の線太さ（px）。0 で線描画をオフ。"""
        self._trail_line_width = max(0.0, float(width))

    def set_trail_max_points(self, n: int) -> None:
        """軌跡の最大保持点数（点数が多いほど長い軌跡）。"""
        self._trail_buffer.set_max_points(n)

    def reset_trail(self) -> None:
        """シーク・モード切替等の時系列ジャンプで連続性が崩れる時に呼ぶ。"""
        self._trail_buffer.reset()

    @property
    def mannequin_visible(self) -> bool:
        return self._mannequin_visible

    def set_mannequin_visible(self, visible: bool) -> None:
        """マネキン本体（関節・骨・頭・足）の表示／非表示を切り替える。
        トレイル表示とは独立。
        """
        self._mannequin_visible = bool(visible)

    def toggle_mannequin_visible(self) -> bool:
        self._mannequin_visible = not self._mannequin_visible
        return self._mannequin_visible

    @property
    def trail_visible(self) -> bool:
        return self._trail_visible

    def set_trail_visible(self, visible: bool) -> None:
        """トレイル（軌跡＋現在位置点）の表示／非表示を切り替える。
        マネキン本体の表示とは独立。
        """
        self._trail_visible = bool(visible)

    def toggle_trail_visible(self) -> bool:
        self._trail_visible = not self._trail_visible
        return self._trail_visible


    @property
    def primitive_foot_y(self) -> float:
        """draw_perspective（Mode3）で使う座標系でのマネキン足元 Y 座標。
        world_landmarks ベース：腰原点で足元は y ≈ 0.95m。
        scale_factor でズームすると床も連動して動く。"""
        return WORLD_FOOT_Y * self._scale_factor

    @property
    def style(self) -> str:
        return self._style

    # M キーの循環対象。この順で回る。
    _STYLE_CYCLE = ("primitive", "mesh", "hidden")

    def set_style(self, style: str) -> None:
        if style not in self._STYLE_CYCLE:
            raise ValueError(f"unknown style: {style!r}")
        self._style = style
        logger.info(f"MannequinRenderer スタイル: {style}")

    def toggle_style(self) -> str:
        """primitive → mesh → hidden → primitive の順で循環する。"""
        try:
            idx = self._STYLE_CYCLE.index(self._style)
        except ValueError:
            idx = -1
        self.set_style(self._STYLE_CYCLE[(idx + 1) % len(self._STYLE_CYCLE)])
        return self._style

    def load_model(self, gltf_path: str) -> None:
        """GLTFモデルを読み込む。initialize()後に呼ぶ。"""
        try:
            self._model = GLTFModel.load(gltf_path)
            logger.info(f"MannequinRenderer: モデル読み込み完了 {gltf_path}")
        except Exception as e:
            logger.error(f"MannequinRenderer: モデル読み込み失敗 {e}")
            self._model = None

    def setup_lighting(self) -> None:
        """ライティングを設定する。"""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)  # glScalef した楕円体の法線を正規化
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, -3.0, 4.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.95, 0.95, 0.95, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.40, 0.40, 0.45, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.85, 0.85, 0.85, 1.0])
        # マテリアルのスペキュラハイライト（金属より柔らかい樹脂・スポーツウェア風）
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.55, 0.55, 0.55, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, [32.0])

    def update_rotation(self, delta: float) -> None:
        self._rotation_y = (self._rotation_y + delta) % 360.0

    def set_rotation_y(self, angle_deg: float) -> None:
        """視点回転角度を直接設定する（角度スライダー用）。"""
        self._rotation_y = float(angle_deg) % 360.0

    @property
    def rotation_y(self) -> float:
        """現在の Y 軸回転角度（度）。Mode3 の床グリッド視点と同期するために公開。"""
        return self._rotation_y

    def draw_ortho(self, results, view_x, view_y, view_w, view_h) -> None:
        """正射影でモデルを描画する（モード2用）。
        投影は MediaPipe 座標系 (x∈[0,1], y∈[0,1]) そのまま使い、
        人物の画面位置にマネキンを追従させる（ローカル正規化なし）。
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, 1.0, 1.0, 0.0, -10.0, 10.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(view_x, view_y, view_w, view_h)

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

        # 楕円体の y スケールを viewport アスペクトで補正するために保存
        self._image_view_aspect = view_w / max(view_h, 1)
        # モード切替検知（座標系が変わるのでトレイルバッファをクリア）
        if self._trail_mode != "ortho":
            self._trail_buffer.reset()
            self._trail_mode = "ortho"

        # Mode2 用 trail 座標変換（raw 画像座標を z 軽縮）
        def _tp_raw(lm):
            return (lm.x, lm.y, -lm.z * 0.5)

        try:
            if not results:
                # 検出なしフレームは点を消すために全部 invisible
                self._trail_buffer.mark_all_invisible()
            else:
                for result in results:
                    # トレイルはマネキン可視性と独立に常時更新する
                    self._trail_buffer.update(result.landmarks, _tp_raw)
                    if self._mannequin_visible:
                        self._draw_model(result.landmarks, coord="raw")
        finally:
            self._image_view_aspect = None

        # トレイル描画（マネキン非表示でも表示する）
        if self._trail_visible:
            self._draw_trails()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

    def draw_perspective(self, results, view_x, view_y, view_w, view_h) -> None:
        """透視投影でモデルを描画する（モード3用）。
        pose_world_landmarks（腰原点・メートル単位）が利用可能なら真の 3D 描画、
        無ければ画像座標ベースのローカル正規化にフォールバック。
        呼び出し側（Mode3D.draw）が事前に depth buffer をクリアしている前提で、
        ここではクリアしない（直前に描画されたシーンオブジェクトとの Z テストを保つため）。
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, view_w / max(view_h, 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(view_x, view_y, view_w, view_h)

        from OpenGL.GLU import gluLookAt
        # 視点は腰の高さ（y=0）から円運動。up=(0,-1,0) で MediaPipe Y(下が+) と一致
        d = self._view_distance
        gluLookAt(
            math.sin(math.radians(self._rotation_y)) * d,
            0.0,
            math.cos(math.radians(self._rotation_y)) * d,
            0, 0, 0,
            0, -1, 0,
        )

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        # モード切替検知（座標系が変わるのでトレイルバッファをクリア）
        if self._trail_mode != "perspective":
            self._trail_buffer.reset()
            self._trail_mode = "perspective"

        # Mode3 用 trail 座標変換（world 座標を scale_factor 倍）
        s = self._scale_factor

        def _tp_world(lm, _s=s):
            return (lm.x * _s, lm.y * _s, lm.z * _s)

        if not results:
            # 検出なしフレームは点を消すために全部 invisible
            self._trail_buffer.mark_all_invisible()
        else:
            for result in results:
                world = getattr(result, "world_landmarks", None)
                if world:
                    # トレイルはマネキン可視性と独立に常時更新する
                    self._trail_buffer.update(world, _tp_world)
                    if self._mannequin_visible:
                        self._draw_model(world, coord="world")
                else:
                    if self._mannequin_visible:
                        self._draw_model(result.landmarks, coord="local")

        # トレイル描画（マネキン非表示でも表示する）
        if self._trail_visible:
            self._draw_trails()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

    def _draw_model(self, lms, coord: str = "local") -> None:
        """スタイル × 座標系に応じてマネキンを描画する。
        coord: "world"（実3D・Mode3）/ "raw"（画像座標・Mode2）/ "local"（体中心正規化・フォールバック）
        style="hidden" ならマネキン本体は描画しない（トレイル等は別経路で描画される）。
        """
        if self._style == "hidden":
            return
        if self._style == "primitive":
            if coord == "world":
                self._draw_primitive_world(lms)
            elif coord == "raw":
                self._draw_primitive_raw(lms)
            else:
                self._draw_primitive_mannequin(lms)
        else:
            self._draw_mesh_mannequin(lms)

    def _draw_primitive_mannequin(self, lms) -> None:
        """関節を球、骨をカプセルで描画する。MediaPipe ランドマークに連動して動く。
        身長（鼻〜足首）基準でスケール正規化し、体中心を原点に置く。
        身長が取れない時は肩幅基準にフォールバック。
        """
        ls = lms[PoseLandmark.LEFT_SHOULDER]
        rs = lms[PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < PRIM_MIN_VIS or rs.visibility < PRIM_MIN_VIS:
            return

        # 横方向は肩中点を原点に
        cx = (ls.x + rs.x) / 2

        # 縦方向は身長（鼻〜両足首中点）を基準に
        nose = lms[PoseLandmark.NOSE]
        la = lms[PoseLandmark.LEFT_ANKLE]
        ra = lms[PoseLandmark.RIGHT_ANKLE]

        nose_y = nose.y if nose.visibility >= PRIM_MIN_VIS else None
        if la.visibility >= PRIM_MIN_VIS and ra.visibility >= PRIM_MIN_VIS:
            foot_y = (la.y + ra.y) / 2
        elif la.visibility >= PRIM_MIN_VIS:
            foot_y = la.y
        elif ra.visibility >= PRIM_MIN_VIS:
            foot_y = ra.y
        else:
            foot_y = None

        if nose_y is not None and foot_y is not None and foot_y - nose_y > 0.1:
            # 身長基準
            height = foot_y - nose_y
            cy = (nose_y + foot_y) / 2
            scale = PRIM_HEIGHT_TARGET * self._scale_factor / height
        else:
            # フォールバック：肩幅基準・肩中点を原点
            shoulder_w = abs(ls.x - rs.x)
            if shoulder_w < 0.01:
                return
            cy = (ls.y + rs.y) / 2
            scale = PRIM_SHOULDER_FALLBACK * self._scale_factor / shoulder_w

        def tp(lm):
            # 原点合わせ＋スケール正規化。
            # MediaPipe Y は下が +、描画系も up=(0,-1,0) のため反転不要。
            # MediaPipe Z はカメラ手前が負・奥が正と逆向きなので反転。
            return (
                (lm.x - cx) * scale,
                (lm.y - cy) * scale,
                -lm.z * scale,
            )

        # パーツ太さもスケールに連動させる（位置だけ広がって細く見えるのを防ぐ）
        joint_r = PRIM_JOINT_R * self._scale_factor
        bone_r = PRIM_BONE_R * self._scale_factor
        head_r = PRIM_HEAD_R * self._scale_factor

        glColor3f(*PRIMITIVE_COLOR)

        # 関節（球）
        for idx in PRIM_JOINTS:
            lm = lms[idx]
            if lm.visibility < PRIM_MIN_VIS:
                continue
            x, y, z = tp(lm)
            glPushMatrix()
            glTranslatef(x, y, z)
            draw_sphere(joint_r, 12, 8)
            glPopMatrix()

        # 骨（カプセル）
        for a, b in PRIM_BONES:
            la, lb = lms[a], lms[b]
            if la.visibility < PRIM_MIN_VIS or lb.visibility < PRIM_MIN_VIS:
                continue
            self._draw_bone_capsule(tp(la), tp(lb), bone_r)

        # 頭（鼻位置に大きめの球）
        nose = lms[PoseLandmark.NOSE]
        if nose.visibility >= PRIM_MIN_VIS:
            x, y, z = tp(nose)
            glColor3f(*HEAD_COLOR)
            glPushMatrix()
            glTranslatef(x, y, z)
            draw_sphere(head_r, 16, 12)
            glPopMatrix()

    def _draw_primitive_world(self, lms) -> None:
        """pose_world_landmarks ベースの実 3D 描画（Mode3）。"""
        s = self._scale_factor

        def tp(lm):
            return (lm.x * s, lm.y * s, lm.z * s)

        self._draw_rich_body(lms, tp, s)

    def _draw_rich_body(self, lms, tp, size_scale: float) -> None:
        """部位別関節・テーパード骨・胴中央・首・指・頭目・足を共通描画する。
        Mode2/3 で共有。size_scale は world (m) サイズを描画系単位に換算する係数。
        """
        s = size_scale
        glColor3f(*PRIMITIVE_COLOR)

        # 胴 axis を計算（後段の torso / neck / head で共有）
        body_axis = self._compute_body_axis(lms, tp)

        # 関節（部位別サイズの球）
        for idx, base_r in JOINT_RADII.items():
            if idx >= len(lms):
                continue
            lm = lms[idx]
            if lm.visibility < PRIM_MIN_VIS:
                continue
            x, y, z = tp(lm)
            glPushMatrix()
            glTranslatef(x, y, z)
            draw_sphere(base_r * s, 14, 10)
            glPopMatrix()

        # 体幹側面・四肢の骨（テーパード）
        for a, b, r1, r2 in TAPERED_BONES:
            if a >= len(lms) or b >= len(lms):
                continue
            la, lb = lms[a], lms[b]
            if la.visibility < PRIM_MIN_VIS or lb.visibility < PRIM_MIN_VIS:
                continue
            self._draw_tapered_bone(tp(la), tp(lb), r1 * s, r2 * s)

        # 胴中央（両肩中点 → 両腰中点）で空洞を埋める
        self._draw_torso_core(lms, tp, s, body_axis)

        # 首（首根本 → 頭部下端）
        self._draw_neck(lms, tp, s, body_axis)

        # 指（手首から先）
        for a, b, r1, r2 in FINGER_BONES:
            if a >= len(lms) or b >= len(lms):
                continue
            la, lb = lms[a], lms[b]
            if la.visibility < PRIM_MIN_VIS or lb.visibility < PRIM_MIN_VIS:
                continue
            self._draw_tapered_bone(tp(la), tp(lb), r1 * s, r2 * s)
            tx, ty, tz = tp(lb)
            glPushMatrix()
            glTranslatef(tx, ty, tz)
            draw_sphere(r2 * s, 8, 6)
            glPopMatrix()

        # 頭部
        self._draw_head(lms, tp, s, body_axis)

        # 足
        self._draw_feet(lms, tp, s)

    @staticmethod
    def _midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)

    @staticmethod
    def _visible(lms, *indices) -> bool:
        for i in indices:
            if i >= len(lms) or lms[i].visibility < PRIM_MIN_VIS:
                return False
        return True

    def _compute_body_axis(self, lms, tp) -> dict | None:
        """胴 4 頂点から center / forward / right / up を計算する。
        forward は 4 三角形 (center, P_i, P_{i+1}) の外積合計の符号反転（体の前向き）。
        right は LS→RS を forward と直交化、up = forward × right。
        4 頂点 visible でなければ None。
        """
        if not self._visible(
            lms,
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
        ):
            return None

        ls = tp(lms[PoseLandmark.LEFT_SHOULDER])
        rs = tp(lms[PoseLandmark.RIGHT_SHOULDER])
        rh = tp(lms[PoseLandmark.RIGHT_HIP])
        lh = tp(lms[PoseLandmark.LEFT_HIP])
        pts = (ls, rs, rh, lh)

        cx = (ls[0] + rs[0] + rh[0] + lh[0]) / 4.0
        cy = (ls[1] + rs[1] + rh[1] + lh[1]) / 4.0
        cz = (ls[2] + rs[2] + rh[2] + lh[2]) / 4.0

        nx_sum = ny_sum = nz_sum = 0.0
        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            ax, ay, az = a[0] - cx, a[1] - cy, a[2] - cz
            bx, by, bz = b[0] - cx, b[1] - cy, b[2] - cz
            nx_sum += ay * bz - az * by
            ny_sum += az * bx - ax * bz
            nz_sum += ax * by - ay * bx
        n_len = math.sqrt(nx_sum * nx_sum + ny_sum * ny_sum + nz_sum * nz_sum)
        if n_len < 1e-9:
            return None
        forward = (-nx_sum / n_len, -ny_sum / n_len, -nz_sum / n_len)

        rrx, rry, rrz = rs[0] - ls[0], rs[1] - ls[1], rs[2] - ls[2]
        dot = rrx * forward[0] + rry * forward[1] + rrz * forward[2]
        rpx = rrx - dot * forward[0]
        rpy = rry - dot * forward[1]
        rpz = rrz - dot * forward[2]
        r_len = math.sqrt(rpx * rpx + rpy * rpy + rpz * rpz)
        if r_len < 1e-9:
            return None
        right = (rpx / r_len, rpy / r_len, rpz / r_len)

        up = (
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        )

        return {
            'center': (cx, cy, cz),
            'forward': forward,
            'right': right,
            'up': up,
            'pts': pts,
        }

    def _compute_neck_base(self, lms, tp, axis, s: float) -> tuple[float, float, float] | None:
        """首根本 = 両肩中点 + (head_center - 両肩中点) × NECK_BASE_RATIO。
        胴の 5 頂点目として、また首ボーンの胴側終点として共有される。
        """
        if axis is None:
            return None
        if not self._visible(
            lms,
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.NOSE,
        ):
            return None
        nose_pt = tp(lms[PoseLandmark.NOSE])
        head_center = self._compute_head_center(lms, tp, nose_pt, axis, s)
        sm = self._midpoint(tp(lms[PoseLandmark.LEFT_SHOULDER]),
                            tp(lms[PoseLandmark.RIGHT_SHOULDER]))
        return (
            sm[0] + (head_center[0] - sm[0]) * NECK_BASE_RATIO,
            sm[1] + (head_center[1] - sm[1]) * NECK_BASE_RATIO,
            sm[2] + (head_center[2] - sm[2]) * NECK_BASE_RATIO,
        )

    def _compute_head_center(self, lms, tp, nose_pt, axis, s: float) -> tuple[float, float, float]:
        """NOSE は顔の前面表面と仮定し、頭部中心 = NOSE - face_forward × HEAD_RZ × s。
        face_forward は両耳 → 鼻方向、両耳が visibility 低い時は胴 forward をフォールバック。
        """
        face_forward = None
        if (PoseLandmark.LEFT_EAR < len(lms)
                and PoseLandmark.RIGHT_EAR < len(lms)
                and lms[PoseLandmark.LEFT_EAR].visibility >= PRIM_MIN_VIS
                and lms[PoseLandmark.RIGHT_EAR].visibility >= PRIM_MIN_VIS):
            le = tp(lms[PoseLandmark.LEFT_EAR])
            re = tp(lms[PoseLandmark.RIGHT_EAR])
            em = self._midpoint(le, re)
            fx, fy, fz = nose_pt[0] - em[0], nose_pt[1] - em[1], nose_pt[2] - em[2]
            f_len = math.sqrt(fx * fx + fy * fy + fz * fz)
            if f_len > 1e-6:
                face_forward = (fx / f_len, fy / f_len, fz / f_len)
        if face_forward is None and axis is not None:
            face_forward = axis['forward']
        if face_forward is None:
            return nose_pt
        rz = HEAD_RZ * s
        return (
            nose_pt[0] - face_forward[0] * rz,
            nose_pt[1] - face_forward[1] * rz,
            nose_pt[2] - face_forward[2] * rz,
        )

    def _draw_torso_core(self, lms, tp, s: float, axis: dict | None) -> None:
        """胴の 5 頂点版描画（neck_base, LS, LH, RH, RS）。
        - 5 シリンダー：neck_base→LS / neck_base→RS / LS→LH / RS→RH / LH→RH
        - 各頂点に対応する関節半径分 forward 方向にシフトした 5 頂点で前後面を 5 三角形ずつ
        - neck_base 位置にスフィア追加
        肩バー (LS-RS) は廃止し、neck_base 経由で繋がる。
        """
        if axis is None:
            return
        neck_base = self._compute_neck_base(lms, tp, axis, s)
        if neck_base is None:
            return

        ls, rs, rh, lh = axis['pts']
        forward = axis['forward']

        # CCW 順（前面から見て反時計回り）：neck_base 上中央 → LS 左上 → LH 左下 → RH 右下 → RS 右上
        pts5 = (neck_base, ls, lh, rh, rs)
        radii5 = (
            NECK_R * s,
            JOINT_RADII[PoseLandmark.LEFT_SHOULDER] * s,
            JOINT_RADII[PoseLandmark.LEFT_HIP] * s,
            JOINT_RADII[PoseLandmark.RIGHT_HIP] * s,
            JOINT_RADII[PoseLandmark.RIGHT_SHOULDER] * s,
        )

        front_pts = [
            (p[0] + forward[0] * r, p[1] + forward[1] * r, p[2] + forward[2] * r)
            for p, r in zip(pts5, radii5)
        ]
        back_pts = [
            (p[0] - forward[0] * r, p[1] - forward[1] * r, p[2] - forward[2] * r)
            for p, r in zip(pts5, radii5)
        ]
        front_center = (
            sum(p[0] for p in front_pts) / 5.0,
            sum(p[1] for p in front_pts) / 5.0,
            sum(p[2] for p in front_pts) / 5.0,
        )
        back_center = (
            sum(p[0] for p in back_pts) / 5.0,
            sum(p[1] for p in back_pts) / 5.0,
            sum(p[2] for p in back_pts) / 5.0,
        )

        glColor3f(*PRIMITIVE_COLOR)

        # 5 シリンダー（5 辺）
        bones = [
            (pts5[0], pts5[1], radii5[0], radii5[1]),  # neck_base → LS
            (pts5[1], pts5[2], radii5[1], radii5[2]),  # LS → LH
            (pts5[2], pts5[3], radii5[2], radii5[3]),  # LH → RH
            (pts5[3], pts5[4], radii5[3], radii5[4]),  # RH → RS
            (pts5[4], pts5[0], radii5[4], radii5[0]),  # RS → neck_base
        ]
        for p1, p2, r1, r2 in bones:
            self._draw_tapered_bone(p1, p2, r1, r2)

        # neck_base スフィア（半径 NECK_R）
        glPushMatrix()
        glTranslatef(*neck_base)
        draw_sphere(NECK_R * s, 14, 10)
        glPopMatrix()

        # 前面：法線 +forward、CCW 巻き順
        glNormal3f(forward[0], forward[1], forward[2])
        glBegin(GL_TRIANGLES)
        for i in range(5):
            glVertex3f(*front_center)
            glVertex3f(*front_pts[i])
            glVertex3f(*front_pts[(i + 1) % 5])
        glEnd()

        # 背面：法線 -forward、巻き順反転
        glNormal3f(-forward[0], -forward[1], -forward[2])
        glBegin(GL_TRIANGLES)
        for i in range(5):
            glVertex3f(*back_center)
            glVertex3f(*back_pts[(i + 1) % 5])
            glVertex3f(*back_pts[i])
        glEnd()

    def _draw_neck(self, lms, tp, s: float, axis: dict | None) -> None:
        """首：胴の 5 頂点目 (neck_base) と頭部楕円体への接合点を結ぶシリンダー。

        接合点計算：
          1. 首の進行方向 u = (head_center − neck_base) を単位化
          2. head_center から −u 方向に楕円体表面までの距離
             surf_dist = 1 / √(uₓ²/rx² + u_y²/ry² + u_z²/rz²)
          3. 首半径 NECK_R 分だけ更に楕円体内側に押し込む（表面での円柱貫入を避ける）
        胴側 neck_base は _draw_torso_core と同じ点を使うのでギャップ無し。
        """
        if axis is None:
            return
        neck_base = self._compute_neck_base(lms, tp, axis, s)
        if neck_base is None:
            return

        nose_pt = tp(lms[PoseLandmark.NOSE])
        head_center = self._compute_head_center(lms, tp, nose_pt, axis, s)
        rx, ry, rz = HEAD_RX * s, HEAD_RY * s, HEAD_RZ * s

        dx = head_center[0] - neck_base[0]
        dy = head_center[1] - neck_base[1]
        dz = head_center[2] - neck_base[2]
        d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d_len < 1e-6:
            return
        ux, uy, uz = dx / d_len, dy / d_len, dz / d_len

        denom = (ux * ux) / (rx * rx) + (uy * uy) / (ry * ry) + (uz * uz) / (rz * rz)
        if denom < 1e-12:
            return
        surf_dist = 1.0 / math.sqrt(denom)

        # 首が楕円体に NECK_R × s 分めり込むことで側面のすき間/段差を消す
        embed = NECK_R * s
        head_bottom = (
            head_center[0] - ux * (surf_dist - embed),
            head_center[1] - uy * (surf_dist - embed),
            head_center[2] - uz * (surf_dist - embed),
        )

        self._draw_tapered_bone(neck_base, head_bottom, NECK_R * s, NECK_R * s)

    def _draw_head(self, lms, tp, s: float, axis: dict | None) -> None:
        """頭部を縦長楕円体（rx, ry, rz）で描画する。
        中心は NOSE（顔の前面表面）から face_forward 方向に -HEAD_RZ × s 押し下げた点。
        """
        if PoseLandmark.NOSE >= len(lms):
            return
        nose = lms[PoseLandmark.NOSE]
        if nose.visibility < PRIM_MIN_VIS:
            return

        nose_pt = tp(nose)
        head_center = self._compute_head_center(lms, tp, nose_pt, axis, s)
        hcx, hcy, hcz = head_center

        rx = HEAD_RX * s
        ry = HEAD_RY * s
        rz = HEAD_RZ * s

        # Mode2 ortho 描画時：viewport の x/y 引き伸ばし比を打ち消すため y スケールに aspect を掛ける
        # （ortho [0,1]×[0,1] が view_w×view_h にマップされ x 方向が view_aspect 倍引き伸ばされるため）
        y_scale = ry / rx
        if self._image_view_aspect is not None:
            y_scale *= self._image_view_aspect

        glColor3f(*HEAD_COLOR)
        glPushMatrix()
        glTranslatef(hcx, hcy, hcz)
        glScalef(1.0, y_scale, rz / rx)
        draw_sphere(rx, 22, 16)
        glPopMatrix()

        glColor3f(*PRIMITIVE_COLOR)

    def _draw_trails(self) -> None:
        """4 部位（両手両足）の移動経路を線＋点でフェードアウト描画する。
        マネキンの上に重ねるため LIGHTING / DEPTH_TEST を OFF にして BLEND ON。
        """
        if self._trail_buffer.is_empty():
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        try:
            # 線（アンチエイリアス付き）
            # GL_LINE_STRIP は太線の接続部 (joint) の隙間処理がないため
            # 急な角で線分が割れて見えることがあるが、運用上は許容する。
            if self._trail_line_width > 0.0:
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                glLineWidth(self._trail_line_width)
                for _pid, color, pts in self._trail_buffer.items():
                    n = len(pts)
                    if n < 2:
                        continue
                    r, g, b = color
                    glBegin(GL_LINE_STRIP)
                    for i, (x, y, z) in enumerate(pts):
                        alpha = (i + 1) / n
                        glColor4f(r, g, b, alpha)
                        glVertex3f(x, y, z)
                    glEnd()
            # 点（軌跡全体ではなく「現在の手足の位置」だけ描く）
            # 認識失敗時は trail_buffer.update が buffer を空にしてくれるので、
            # ここでは buffer の中身だけ見ればよい（描画判定は buffer 側で一元管理）。
            if self._trail_point_size > 0.0:
                glEnable(GL_POINT_SMOOTH)
                glPointSize(self._trail_point_size)
                glBegin(GL_POINTS)
                for pid, color, pts in self._trail_buffer.items():
                    if not pts:
                        continue
                    r, g, b = color
                    x, y, z = pts[-1]
                    glColor4f(r, g, b, 1.0)
                    glVertex3f(x, y, z)
                glEnd()
        finally:
            glDisable(GL_LINE_SMOOTH)
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)

    def _draw_feet(self, lms, tp, s: float) -> None:
        """両足：ANKLE → FOOT_INDEX を本体と同じ色でテーパード描画する。"""
        feet_pairs = [
            (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
            (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
        ]
        glColor3f(*PRIMITIVE_COLOR)
        for ank_i, toe_i in feet_pairs:
            if ank_i >= len(lms) or toe_i >= len(lms):
                continue
            ank, toe = lms[ank_i], lms[toe_i]
            if ank.visibility < PRIM_MIN_VIS or toe.visibility < PRIM_MIN_VIS:
                continue
            # 足首 0.035 → つま先 0.025 でテーパード（足首関節と滑らかに繋がる）
            self._draw_tapered_bone(tp(ank), tp(toe), 0.035 * s, 0.025 * s)

    @staticmethod
    def _draw_tapered_bone(p1, p2, r1: float, r2: float) -> None:
        """p1 → p2 を結ぶテーパード円錐台を描画する。両端で半径が変わる。
        例外で glPopMatrix が呼ばれないと matrix stack が累積してオーバーフローするので、
        ゼロ除算ガード＋ try/finally で必ず pop されるようにする。
        """
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        vz = p2[2] - p1[2]
        length = math.sqrt(vx * vx + vy * vy + vz * vz)
        if length < 1e-6:
            return

        glPushMatrix()
        try:
            glTranslatef(p1[0], p1[1], p1[2])
            dz = vz / length
            if dz < 0.99999:
                if dz > -0.99999:
                    ax, ay = -vy, vx
                    an = math.sqrt(ax * ax + ay * ay)
                    if an > 1e-9:
                        angle = math.degrees(math.acos(max(-1.0, min(1.0, dz))))
                        glRotatef(angle, ax / an, ay / an, 0.0)
                else:
                    glRotatef(180.0, 1.0, 0.0, 0.0)
            draw_tapered_cylinder(r1, r2, length, 14)
        finally:
            glPopMatrix()

    def _draw_primitive_raw(self, lms) -> None:
        """MediaPipe 画像座標 (x,y∈[0,1]) をそのまま使うリッチ描画。Mode2 用。
        ローカル正規化せず、人物の画面位置にマネキンが追従する。
        サイズは world (m) を画像比率に換算（self._raw_size_scale）して描画する。
        """
        size_scale = self._scale_factor * self._raw_size_scale

        def tp(lm):
            return (lm.x, lm.y, -lm.z * 0.5)

        self._draw_rich_body(lms, tp, size_scale)

    @staticmethod
    def _draw_bone_capsule(p1, p2, radius: float) -> None:
        """p1 → p2 を結ぶカプセルを描画する。Z 軸カプセルを回転して向きを合わせる。
        例外時の pop 漏れ対策で try/finally + ゼロ除算ガード。
        """
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        vz = p2[2] - p1[2]
        length = math.sqrt(vx * vx + vy * vy + vz * vz)
        if length < 1e-6:
            return

        glPushMatrix()
        try:
            glTranslatef(p1[0], p1[1], p1[2])
            dz = vz / length
            if dz < 0.99999:
                if dz > -0.99999:
                    ax, ay = -vy, vx
                    an = math.sqrt(ax * ax + ay * ay)
                    if an > 1e-9:
                        angle = math.degrees(math.acos(max(-1.0, min(1.0, dz))))
                        glRotatef(angle, ax / an, ay / an, 0.0)
                else:
                    glRotatef(180.0, 1.0, 0.0, 0.0)
            draw_capsule(radius, length)
        finally:
            glPopMatrix()

    def _draw_mesh_mannequin(self, lms) -> None:
        """BlockMan.gltf を Tポーズ固定で描画する（スキニング未適用）。
        両肩中点を基準に位置・スケールを合わせる。
        """
        if self._model is None:
            return

        # 肩が見えていることを確認
        ls = lms[PoseLandmark.LEFT_SHOULDER]
        rs = lms[PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < 0.3 or rs.visibility < 0.3:
            return

        # MediaPipeの肩中点
        mp_shoulder_cx = (ls.x + rs.x) / 2 - 0.5
        mp_shoulder_cy = (ls.y + rs.y) / 2 - 0.5

        # MediaPipeの肩幅
        mp_shoulder_w = abs(ls.x - rs.x)
        if mp_shoulder_w < 0.01:
            return

        # バインドポーズ頂点（Y反転してMediaPipe座標系に合わせる）
        verts = self._model.positions.copy()
        verts[:,1] = -verts[:,1]
        norms = self._model.normals.copy()
        norms[:,1] = -norms[:,1]

        # モデルの肩幅・肩Y位置（ボーンウェイトから計算した実測値）
        MODEL_SHOULDER_W = 1.358   # 肩関節の実際の幅（手先ではなく肩骨）
        MODEL_SHOULDER_Y = -0.409  # Y反転後の肩のY座標（Left/Right平均）

        # スケール = MediaPipe肩幅 / モデル肩幅 × ユーザー指定の倍率
        scale = mp_shoulder_w / MODEL_SHOULDER_W * self._scale_factor
        verts *= scale

        # 肩中点をMediaPipeの肩位置に合わせる
        model_shoulder_x = 0.0  # モデルはX対称なので中心=0
        model_shoulder_y = MODEL_SHOULDER_Y * scale

        verts[:,0] += mp_shoulder_cx - model_shoulder_x
        verts[:,1] += mp_shoulder_cy - model_shoulder_y

        # OpenGL描画
        glColor3f(*MODEL_COLOR)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        verts_c = np.ascontiguousarray(verts, dtype=np.float32)
        norms_c = np.ascontiguousarray(norms, dtype=np.float32)
        idx_c   = np.ascontiguousarray(self._model.indices, dtype=np.uint32)

        glVertexPointer(3, GL_FLOAT, 0, verts_c)
        glNormalPointer(GL_FLOAT, 0, norms_c)
        glDrawElements(GL_TRIANGLES, len(idx_c), GL_UNSIGNED_INT, idx_c)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
