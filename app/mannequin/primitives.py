"""
primitives.py
球・カプセル・円柱などのOpenGL3Dプリミティブ描画関数。
"""

from __future__ import annotations
import math
from OpenGL.GL import (
    glBegin, glEnd, glVertex3f, glNormal3f,
    GL_TRIANGLE_STRIP, GL_TRIANGLES, GL_TRIANGLE_FAN, GL_QUAD_STRIP,
)
from OpenGL.GLU import gluSphere, gluNewQuadric, gluCylinder, gluDisk

# 毎フレーム gluNewQuadric() を呼ぶと GPU 内部リソースが累積して
# 動画書出のような高速連続描画で描画が壊れる。モジュールスコープで
# 1 個だけ作って使い回す。
_QUADRIC = None


def _get_quadric():
    """共有 quadric オブジェクトを返す（遅延初期化）。
    OpenGL context がアクティブな状態で初回呼び出しすること。
    """
    global _QUADRIC
    if _QUADRIC is None:
        _QUADRIC = gluNewQuadric()
    return _QUADRIC


def draw_sphere(radius: float, slices: int = 12, stacks: int = 8) -> None:
    """球を描画する。"""
    gluSphere(_get_quadric(), radius, slices, stacks)


def draw_cylinder(radius: float, height: float,
                  slices: int = 12) -> None:
    """Z軸方向の円柱を描画する（両端キャップなし）。"""
    gluCylinder(_get_quadric(), radius, radius, height, slices, 1)


def draw_tapered_cylinder(base_r: float, top_r: float, height: float,
                          slices: int = 14) -> None:
    """Z軸方向の円錐台（両端半径違い）を両端キャップ付きで描画する。
    base が z=0（法線 -Z）、top が z=height（法線 +Z）。
    """
    from OpenGL.GL import glPushMatrix, glPopMatrix, glTranslatef, glRotatef
    q = _get_quadric()
    gluCylinder(q, base_r, top_r, height, slices, 1)
    # base キャップ：法線を -Z に向けるため X 軸 180° 回転
    if base_r > 0:
        glPushMatrix()
        glRotatef(180.0, 1.0, 0.0, 0.0)
        gluDisk(q, 0.0, base_r, slices, 1)
        glPopMatrix()
    # top キャップ：z=height に移動、法線は +Z（デフォルト）
    if top_r > 0:
        glPushMatrix()
        glTranslatef(0.0, 0.0, height)
        gluDisk(q, 0.0, top_r, slices, 1)
        glPopMatrix()


def draw_capsule(radius: float, height: float,
                 slices: int = 12, stacks: int = 6) -> None:
    """カプセル（円柱 + 両端半球）をZ軸方向に描画する。
    height はカプセル全体の長さ（球の直径を含む）。
    """
    from OpenGL.GL import glPushMatrix, glPopMatrix, glTranslatef, glRotatef
    shaft = max(0.0, height - radius * 2)
    q = _get_quadric()

    # 胴体（円柱）
    gluCylinder(q, radius, radius, shaft, slices, 1)

    # 上端キャップ（半球）
    glPushMatrix()
    glTranslatef(0, 0, shaft)
    _draw_hemisphere(radius, slices, stacks)
    glPopMatrix()

    # 下端キャップ（半球・反転）
    glPushMatrix()
    glRotatef(180, 1, 0, 0)
    _draw_hemisphere(radius, slices, stacks)
    glPopMatrix()

    # 両端ディスク
    gluDisk(q, 0, radius, slices, 1)
    glPushMatrix()
    glTranslatef(0, 0, shaft)
    gluDisk(q, 0, radius, slices, 1)
    glPopMatrix()


def draw_box(size_x: float, size_y: float, size_z: float) -> None:
    """原点中心の直方体を描画する（6面 × 法線付き）。
    size_x/y/z はそれぞれ各軸方向の全長（半幅ではない）。
    """
    hx = size_x / 2.0
    hy = size_y / 2.0
    hz = size_z / 2.0

    glBegin(GL_TRIANGLES)
    # +X 面
    glNormal3f(1, 0, 0)
    for vx, vy, vz in [(+hx, -hy, -hz), (+hx, +hy, -hz), (+hx, +hy, +hz),
                       (+hx, -hy, -hz), (+hx, +hy, +hz), (+hx, -hy, +hz)]:
        glVertex3f(vx, vy, vz)
    # -X 面
    glNormal3f(-1, 0, 0)
    for vx, vy, vz in [(-hx, -hy, -hz), (-hx, +hy, +hz), (-hx, +hy, -hz),
                       (-hx, -hy, -hz), (-hx, -hy, +hz), (-hx, +hy, +hz)]:
        glVertex3f(vx, vy, vz)
    # +Y 面
    glNormal3f(0, 1, 0)
    for vx, vy, vz in [(-hx, +hy, -hz), (+hx, +hy, +hz), (+hx, +hy, -hz),
                       (-hx, +hy, -hz), (-hx, +hy, +hz), (+hx, +hy, +hz)]:
        glVertex3f(vx, vy, vz)
    # -Y 面
    glNormal3f(0, -1, 0)
    for vx, vy, vz in [(-hx, -hy, -hz), (+hx, -hy, -hz), (+hx, -hy, +hz),
                       (-hx, -hy, -hz), (+hx, -hy, +hz), (-hx, -hy, +hz)]:
        glVertex3f(vx, vy, vz)
    # +Z 面
    glNormal3f(0, 0, 1)
    for vx, vy, vz in [(-hx, -hy, +hz), (+hx, -hy, +hz), (+hx, +hy, +hz),
                       (-hx, -hy, +hz), (+hx, +hy, +hz), (-hx, +hy, +hz)]:
        glVertex3f(vx, vy, vz)
    # -Z 面
    glNormal3f(0, 0, -1)
    for vx, vy, vz in [(-hx, -hy, -hz), (+hx, +hy, -hz), (+hx, -hy, -hz),
                       (-hx, -hy, -hz), (-hx, +hy, -hz), (+hx, +hy, -hz)]:
        glVertex3f(vx, vy, vz)
    glEnd()


def _build_rounded_quad_contour(p1, p2, p3, p4,
                                  corner_r: float, n_arc: int = 6):
    """4 頂点 (2D ローカル座標, CCW) の四辺形を corner_r で角丸化した
    輪郭点列 (CCW) を返す。各 1/4 円弧は n_arc + 1 点で離散化する。

    p1〜p4 は (x, y) のタプル。コーナーが 180° (一直線) の時は丸めず通過。
    """
    pts = [p1, p2, p3, p4]
    contour: list[tuple[float, float]] = []
    for i in range(4):
        p = pts[i]
        prev_pt = pts[(i - 1) % 4]
        next_pt = pts[(i + 1) % 4]

        # 入りベクトル：p から prev へ
        ex_in = prev_pt[0] - p[0]
        ey_in = prev_pt[1] - p[1]
        l_in = math.hypot(ex_in, ey_in)
        if l_in < 1e-9:
            contour.append(p)
            continue
        ex_in /= l_in
        ey_in /= l_in

        # 出ベクトル：p から next へ
        ex_out = next_pt[0] - p[0]
        ey_out = next_pt[1] - p[1]
        l_out = math.hypot(ex_out, ey_out)
        if l_out < 1e-9:
            contour.append(p)
            continue
        ex_out /= l_out
        ey_out /= l_out

        # 角丸の始点 / 終点（頂点から両側の辺方向へ corner_r 進んだ点）
        start_x = p[0] + corner_r * ex_in
        start_y = p[1] + corner_r * ey_in
        end_x = p[0] + corner_r * ex_out
        end_y = p[1] + corner_r * ey_out

        # 二等分線方向と挟角から円弧中心への距離を求める
        bx = ex_in + ex_out
        by = ey_in + ey_out
        bl = math.hypot(bx, by)
        if bl < 1e-9:
            # 180° 直角でない場合（180° 一直線）は丸めず頂点を通過
            contour.append(p)
            continue
        bx /= bl
        by /= bl
        cos_half = bl / 2.0  # |ein + eout| = 2 cos(挟角/2)
        sin_half = math.sqrt(max(0.0, 1.0 - cos_half * cos_half))
        if sin_half < 1e-6:
            contour.append(p)
            continue
        center_dist = corner_r / sin_half
        cx = p[0] + center_dist * bx
        cy = p[1] + center_dist * by

        # 始点〜終点を結ぶ円弧
        ang_start = math.atan2(start_y - cy, start_x - cx)
        ang_end = math.atan2(end_y - cy, end_x - cx)
        d_ang = ang_end - ang_start
        if d_ang > math.pi:
            d_ang -= 2 * math.pi
        elif d_ang < -math.pi:
            d_ang += 2 * math.pi

        for k in range(n_arc + 1):
            t = k / n_arc
            a = ang_start + d_ang * t
            contour.append((cx + corner_r * math.cos(a),
                            cy + corner_r * math.sin(a)))

    return contour


def draw_rounded_prism(p1, p2, p3, p4,
                        center: tuple[float, float, float],
                        right: tuple[float, float, float],
                        up: tuple[float, float, float],
                        forward: tuple[float, float, float],
                        depth: float, corner_r: float,
                        n_arc: int = 6) -> None:
    """4 頂点 (世界座標) を「前面」とする角丸プリズム（前後押し出し）を描画する。

    胴のローカル軸 (right, up, forward) と中心 center を渡す。
    forward 方向に depth/2 オフセットして背面を作る。
    前後面は平面、側面は前後の対応する角丸輪郭点を quad strip で繋ぐ。

    Args:
        p1〜p4 : 前面の 4 頂点 (世界座標、CCW で回るとよい)
        center : 胴中心 (世界座標)
        right, up, forward : 胴のローカル軸 (単位ベクトル)
        depth  : 前後の厚み
        corner_r : 角丸半径
    """
    # 4 頂点を center 基準のローカル 2D (right, up 平面) に投影
    def to_local(p):
        dx, dy, dz = p[0] - center[0], p[1] - center[1], p[2] - center[2]
        x = dx * right[0] + dy * right[1] + dz * right[2]
        y = dx * up[0] + dy * up[1] + dz * up[2]
        return (x, y)

    contour_2d = _build_rounded_quad_contour(
        to_local(p1), to_local(p2), to_local(p3), to_local(p4),
        corner_r, n_arc=n_arc,
    )
    n = len(contour_2d)
    if n < 3:
        return

    half_d = depth / 2.0
    fx, fy, fz = forward
    rx, ry, rz = right
    ux, uy, uz = up

    # ローカル 2D → 世界座標（前面 = -forward * half_d）
    front_3d = []
    back_3d = []
    for lx, ly in contour_2d:
        wx = center[0] + lx * rx + ly * ux
        wy = center[1] + lx * ry + ly * uy
        wz = center[2] + lx * rz + ly * uz
        front_3d.append((wx - fx * half_d, wy - fy * half_d, wz - fz * half_d))
        back_3d.append((wx + fx * half_d, wy + fy * half_d, wz + fz * half_d))

    cx_f = center[0] - fx * half_d
    cy_f = center[1] - fy * half_d
    cz_f = center[2] - fz * half_d
    cx_b = center[0] + fx * half_d
    cy_b = center[1] + fy * half_d
    cz_b = center[2] + fz * half_d

    # 前面（法線 -forward）TRIANGLE_FAN
    glNormal3f(-fx, -fy, -fz)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(cx_f, cy_f, cz_f)
    for p in front_3d:
        glVertex3f(*p)
    glVertex3f(*front_3d[0])
    glEnd()

    # 背面（法線 +forward、巻き順反転）
    glNormal3f(fx, fy, fz)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(cx_b, cy_b, cz_b)
    for p in reversed(back_3d):
        glVertex3f(*p)
    glVertex3f(*back_3d[-1])
    glEnd()

    # 側面 quad strip：各輪郭点で「ローカル中心からの方向」を外向き法線として使う
    glBegin(GL_QUAD_STRIP)
    for i in range(n + 1):
        idx = i % n
        lx, ly = contour_2d[idx]
        l = math.hypot(lx, ly)
        if l < 1e-9:
            nx_l, ny_l = 0.0, 1.0
        else:
            nx_l, ny_l = lx / l, ly / l
        nx = nx_l * rx + ny_l * ux
        ny = nx_l * ry + ny_l * uy
        nz = nx_l * rz + ny_l * uz
        glNormal3f(nx, ny, nz)
        glVertex3f(*back_3d[idx])
        glVertex3f(*front_3d[idx])
    glEnd()


def _draw_hemisphere(radius: float, slices: int, stacks: int) -> None:
    """+Z方向の半球を描画する。"""
    for i in range(stacks // 2):
        lat0 = math.pi * i / stacks
        lat1 = math.pi * (i + 1) / stacks
        z0 = math.cos(lat0) * radius
        z1 = math.cos(lat1) * radius
        r0 = math.sin(lat0) * radius
        r1 = math.sin(lat1) * radius

        glBegin(GL_TRIANGLE_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * j / slices
            c, s = math.cos(lng), math.sin(lng)
            glNormal3f(c * math.sin(lat1), s * math.sin(lat1), math.cos(lat1))
            glVertex3f(c * r1, s * r1, z1)
            glNormal3f(c * math.sin(lat0), s * math.sin(lat0), math.cos(lat0))
            glVertex3f(c * r0, s * r0, z0)
        glEnd()
