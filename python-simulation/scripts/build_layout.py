"""Sinh hình học ngã tư Lê Văn Lương × Hoàng Đạo Thúy / Hoàng Minh Giám cho mô phỏng Python độc lập.

Mọi số đo nằm trong ARMS / CORNER_R bên dưới. Đo thực địa xong thì sửa số ở đây rồi chạy lại:
    python scripts/build_layout.py
Đầu ra (UTM 48N / EPSG:32648, mét):
    python-simulation/simulation/python_simulation/includes/*.geojson
    gama-simulation/simulation/includes/*.geojson

Hệ tọa độ đo: mét, gốc O = tâm 4 nút OSM của ngã tư (21.005326 N, 105.804685 E), x = Đông, y = Bắc.
Mỗi nhánh: s = khoảng cách dọc trục nhánh tính từ O (hướng ra ngoài), t = lệch ngang, dương về phía TRÁI
khi nhìn ra ngoài (= phía chiều VÀO vì xe đi bên phải). Nguồn: ảnh Esri World Imagery 11/2025 (0,28 m/px);
riêng nhánh SE đo trên ảnh Esri Wayback 06/2023 rồi quy đổi về cùng khung (xem README).
"""
import json, math, os
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, Point, mapping
from shapely.ops import unary_union

SUBLANE_W = 1.05  # m, bề rộng 1 sublane (~3 sublane = 1 làn to 3.15 m; hướng 4 làn ≈ 12 sublane)
ARM_LEN = 150.0   # m, chiều dài mỗi nhánh đưa vào mô phỏng, tính từ O

ARMS = {  # bearing: phương vị trục nhánh (độ, từ Bắc theo chiều kim đồng hồ)
    #        c: tâm dải phân cách (t), med: rộng dải phân cách, w_in / w_out: rộng lòng đường chiều vào / ra (bó vỉa–bó vỉa)
    #        cw: vạch qua đường [mép trong, mép ngoài] (s), stop: vạch dừng chiều vào (s), nose: mũi dải phân cách (s)
    'NE': dict(name='Lê Văn Lương (hướng Láng Hạ)', bearing=53.4, c=-2.10, med=7.8, w_in=13.3, w_out=12.7, cw=(14.1, 17.7), stop=19.8, nose=17.4),
    'SE': dict(name='Hoàng Đạo Thúy (hướng Nhân Hòa)', bearing=128.5, c=-6.11, med=2.5, w_in=13.5, w_out=9.2, cw=(23.3, 26.5), stop=28.5, nose=28.3),
    'SW': dict(name='Lê Văn Lương (hướng Khuất Duy Tiến)', bearing=232.8, c=1.05, med=7.9, w_in=12.8, w_out=12.4, cw=(19.6, 22.9), stop=24.9, nose=21.8),
    'NW': dict(name='Hoàng Minh Giám (hướng Trung Hòa)', bearing=316.4, c=1.40, med=2.6, w_in=11.9, w_out=9.9, cw=(20.1, 23.4), stop=25.4, nose=23.7),
}
CLOCKWISE = ['NE', 'SE', 'SW', 'NW']
CORNER_R = {'E': 9.0, 'S': 9.0, 'W': 9.0, 'N': 9.0}  # bán kính bó vỉa góc, góc nằm giữa nhánh i và nhánh kế tiếp theo chiều kim đồng hồ
CORNER_OF = {('NE', 'SE'): 'E', ('SE', 'SW'): 'S', ('SW', 'NW'): 'W', ('NW', 'NE'): 'N'}
RIGHT_TURN_SUBLANES = 5      # bề rộng nhánh rẽ phải (sublane, tính từ bó vỉa); hành lang rẽ phải trên nhánh vào là 3 sublane
RIGHT_TURN_CLEARANCE = 0.3   # m, khe hở giữa thân xe máy ở sublane sát bó vỉa và bó vỉa góc
MAX_EXIT_SHIFT = 9.0         # m, nhánh ra được bắt đầu xa hơn mép vạch qua đường tối đa chừng này để cung rẽ phải hết cong

LAT0, LON0 = 21.005326, 105.804685
KX, KY = 111320 * math.cos(math.radians(LAT0)), 110574


def frame(a):
    b = math.radians(ARMS[a]['bearing'])
    return (math.sin(b), math.cos(b)), (-math.cos(b), math.sin(b))  # u (ra ngoài), v (bên trái u)


def P(a, s, t):
    (ux, uy), (vx, vy) = frame(a)
    return (s * ux + t * vx, s * uy + t * vy)


def edges(a):
    """t của 4 mép: ngoài chiều vào, trong chiều vào, trong chiều ra, ngoài chiều ra."""
    A = ARMS[a]
    return A['c'] + A['med'] / 2 + A['w_in'], A['c'] + A['med'] / 2, A['c'] - A['med'] / 2, A['c'] - A['med'] / 2 - A['w_out']


def n_sub(w):
    return int(math.floor(w / SUBLANE_W + 1e-9))


def right_turn_fillet(p0, p3, incoming, outgoing, target_radius, max_segment=1.25):
    """Make a tangent-continuous, near-circular right-turn connector.

    ``p0``/``p3`` are the stop-line and exit endpoints. Straight tangent legs
    join them to a cubic Bezier quarter-turn whose radius is set from the
    corner curb radius plus the connector's lateral offset from the curb.
    The radius is reduced only when the available stop-line/crosswalk lengths
    cannot contain the full fillet.
    """
    norm = lambda v: math.hypot(v[0], v[1])
    unit = lambda v: (v[0] / norm(v), v[1] / norm(v))
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    add = lambda a, b: (a[0] + b[0], a[1] + b[1])
    sub = lambda a, b: (a[0] - b[0], a[1] - b[1])
    mul = lambda v, s: (v[0] * s, v[1] * s)

    incoming, outgoing = unit(incoming), unit(outgoing)
    delta = sub(p3, p0)
    reverse_out = mul(outgoing, -1.0)
    denominator = cross(incoming, reverse_out)
    if abs(denominator) < 1e-6:
        raise ValueError("right-turn tangents are parallel")
    to_corner = cross(delta, reverse_out) / denominator
    from_corner = cross(delta, incoming) / denominator
    if to_corner <= 0 or from_corner <= 0:
        raise ValueError("right-turn tangent intersection is outside the road segments")
    corner = add(p0, mul(incoming, to_corner))
    theta = abs(math.atan2(cross(incoming, outgoing),
                           incoming[0] * outgoing[0] + incoming[1] * outgoing[1]))
    if theta < math.radians(10) or theta > math.radians(150):
        raise ValueError("right-turn deflection is outside the supported range")

    tangent_factor = math.tan(theta / 2.0)
    radius = min(target_radius,
                 (min(to_corner, from_corner) - 0.25) / tangent_factor)
    if radius < target_radius - 1e-6:
        print(f"  cảnh báo: bán kính rẽ phải giảm {target_radius:.2f} -> {radius:.2f} m vì thiếu chỗ")
    if radius <= 0:
        raise ValueError("insufficient room for a right-turn curve")
    tangent_length = radius * tangent_factor
    arc_start = sub(corner, mul(incoming, tangent_length))
    arc_end = add(corner, mul(outgoing, tangent_length))
    handle = (4.0 / 3.0) * radius * math.tan(theta / 4.0)
    control1 = add(arc_start, mul(incoming, handle))
    control2 = sub(arc_end, mul(outgoing, handle))

    def line_points(start, end):
        distance = norm(sub(end, start))
        count = max(1, math.ceil(distance / max_segment))
        return [add(start, mul(sub(end, start), i / count))
                for i in range(1, count + 1)]

    points = [p0]
    points.extend(line_points(p0, arc_start))
    arc_length = radius * theta
    count = max(8, math.ceil(arc_length / max_segment))
    for i in range(1, count):
        t = i / count
        u = 1.0 - t
        points.append((u**3 * arc_start[0] + 3*u*u*t * control1[0] +
                       3*u*t*t * control2[0] + t**3 * arc_end[0],
                       u**3 * arc_start[1] + 3*u*u*t * control1[1] +
                       3*u*t*t * control2[1] + t**3 * arc_end[1]))
    points.append(arc_end)
    points.extend(line_points(arc_end, p3))
    return points


def right_turn_curb_path(s0, e0, incoming, outgoing, radius, curb, clearance, centre, max_segment=0.5):
    """Centre line of the curb sublane of a right turn.

    ``s0``: curb-sublane point on the stop line, ``e0``: on the exit start.
    Preferred shape: a circular arc of ``radius`` concentric with the curb
    corner, tangent to both curb-sublane lines. Where the exit (or approach)
    leg is too short for that arc - the curb rounding then already starts
    before the stop line - the path follows the rounding in polar coordinates
    around its centre ``centre`` (radius a cubic Hermite of the angle), tangent
    to the approach at ``s0`` and to the exit at ``e0``, keeping ``clearance``
    from the curb polygon ``curb``.
    """
    norm = lambda v: math.hypot(v[0], v[1])
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    # the straight curb sublane itself may sit a little closer than ``clearance``
    clearance = min(clearance, curb.distance(Point(s0)) - 0.02, curb.distance(Point(e0)) - 0.02)
    inc = (incoming[0] / norm(incoming), incoming[1] / norm(incoming))
    out = (outgoing[0] / norm(outgoing), outgoing[1] / norm(outgoing))
    d = (e0[0] - s0[0], e0[1] - s0[1])
    ro = (-out[0], -out[1])
    den = cross(inc, ro)
    l0 = cross(d, ro) / den          # stop line -> corner of the two lines
    l1 = cross(d, inc) / den         # corner -> exit start
    corner = (s0[0] + inc[0] * l0, s0[1] + inc[1] * l0)
    theta = abs(math.atan2(cross(inc, out), inc[0] * out[0] + inc[1] * out[1]))

    def bezier(a, h1, h2, n=160):
        A = (corner[0] - inc[0] * a, corner[1] - inc[1] * a)
        E = (corner[0] + out[0] * l1, corner[1] + out[1] * l1)
        C1 = (A[0] + inc[0] * h1, A[1] + inc[1] * h1)
        C2 = (E[0] - out[0] * h2, E[1] - out[1] * h2)
        pts = []
        for i in range(n + 1):
            q = i / n; u = 1 - q
            pts.append((u**3 * A[0] + 3*u*u*q * C1[0] + 3*u*q*q * C2[0] + q**3 * E[0],
                        u**3 * A[1] + 3*u*u*q * C1[1] + 3*u*q*q * C2[1] + q**3 * E[1]))
        return A, pts

    def min_radius(pts):
        best = math.inf
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
            area2 = abs(cross((b[0]-a[0], b[1]-a[1]), (c[0]-a[0], c[1]-a[1])))
            if area2 > 1e-12:
                best = min(best, ab * bc * ca / (2 * area2))
        return best

    tangent = radius * math.tan(theta / 2.0)
    if tangent <= l0 - 0.25 and tangent <= l1:
        # circular arc as a cubic (4/3 tan(theta/4) handles)
        h = (4.0 / 3.0) * radius * math.tan(theta / 4.0)
        A = (corner[0] - inc[0] * tangent, corner[1] - inc[1] * tangent)
        E = (corner[0] + out[0] * tangent, corner[1] + out[1] * tangent)
        C1 = (A[0] + inc[0] * h, A[1] + inc[1] * h)
        C2 = (E[0] - out[0] * h, E[1] - out[1] * h)
        arc = []
        for i in range(161):
            q = i / 160; u = 1 - q
            arc.append((u**3 * A[0] + 3*u*u*q * C1[0] + 3*u*q*q * C2[0] + q**3 * E[0],
                        u**3 * A[1] + 3*u*u*q * C1[1] + 3*u*q*q * C2[1] + q**3 * E[1]))
        tail = [(E[0] + out[0] * s, E[1] + out[1] * s) for s in
                [k * max_segment for k in range(1, int((l1 - tangent) / max_segment) + 1)]]
        start, path = A, arc + tail + ([e0] if l1 - tangent > 1e-6 else [])
    else:
        # The curb rounding already starts before the stop line (acute corner):
        # follow it in polar coordinates around the rounding centre ``centre``.
        # rho(psi) is a cubic Hermite whose end slopes make the path tangent to the
        # approach at s0 and to the exit at e0, so the curve drifts smoothly away
        # from the curb where the curb bends away, then eases back onto the exit lane.
        C = centre
        ang = lambda p: math.atan2(p[1] - C[1], p[0] - C[0])
        psi0, psi1 = ang(s0), ang(e0)
        dpsi = (psi1 - psi0) % (2 * math.pi) - 2 * math.pi  # right turn = clockwise
        r_a, r_b = math.dist(C, s0), math.dist(C, e0)

        def slope(psi, rho, h):  # d rho / d psi so that the path is parallel to h
            er = (math.cos(psi), math.sin(psi)); ep = (-er[1], er[0])
            return -rho * cross(ep, h) / cross(er, h)

        m0, m1 = slope(psi0, r_a, inc) * dpsi, slope(psi1, r_b, out) * dpsi

        def polar(c1, c2, n=160):
            pts = []
            for i in range(n + 1):
                q = i / n
                h00, h10, h01, h11 = 2*q**3 - 3*q*q + 1, q**3 - 2*q*q + q, -2*q**3 + 3*q*q, q**3 - q*q
                bump = q * q * (1 - q) ** 2   # zero value and slope at both ends
                rho = h00 * r_a + h10 * m0 + h01 * r_b + h11 * m1 + bump * (c1 + c2 * (q - 0.5))
                psi = psi0 + dpsi * q
                pts.append((C[0] + rho * math.cos(psi), C[1] + rho * math.sin(psi)))
            return pts

        # the plain Hermite turns hardest right after the stop line; two shape
        # terms (that keep the end tangents) spread the turn over the whole path
        best = None
        for c1 in [k * 1.0 for k in range(-10, 41)]:
            for c2 in [k * 4.0 for k in range(-20, 21)]:
                pts = polar(c1, c2, 60)
                if curb.distance(LineString(pts)) < clearance - 0.05:
                    continue
                # a right turn only: no reverse (left) curvature anywhere
                if any(cross((b[0]-a[0], b[1]-a[1]), (c[0]-b[0], c[1]-b[1])) > 1e-6
                       for a, b, c in zip(pts, pts[1:], pts[2:])):
                    continue
                r = min_radius(pts)
                if best is None or r > best[0]:
                    best = (r, c1, c2)
        if best is None:
            raise ValueError("right-turn curve cuts the curb rounding")
        path = polar(best[1], best[2])
        print(f"  rẽ phải: bó vỉa cong trước vạch dừng, bám bó vỉa theo tọa độ cực, "
              f"bán kính nhỏ nhất {min_radius(path):.1f} m, khe hở bó vỉa {curb.distance(LineString(path)):.2f} m")
        start = s0
    lead = norm((start[0] - s0[0], start[1] - s0[1]))
    pts = [s0] + [(s0[0] + inc[0] * s, s0[1] + inc[1] * s) for s in
                  [k * max_segment for k in range(1, int(lead / max_segment))]]
    pts.extend(path)
    # thin to ~max_segment spacing, keep both ends exactly
    out_pts = [pts[0]]
    for q in pts[1:-1]:
        if math.dist(q, out_pts[-1]) >= max_segment and math.dist(q, pts[0]) > 0.4 and math.dist(q, pts[-1]) > 0.4:
            out_pts.append(q)
    if len(out_pts) > 1 and math.dist(out_pts[-1], pts[-1]) < max_segment / 2:
        out_pts.pop()  # a stub last segment would spoil the end normal of the offset curves
    out_pts.append(pts[-1])
    # short end stubs along the approach / exit direction: the first and last
    # segments then carry the exact tangent, so every sublane meets its
    # approach and exit sublane without a heading kink or longitudinal jump
    stub = 0.25
    out_pts.insert(1, (s0[0] + inc[0] * stub, s0[1] + inc[1] * stub))
    out_pts.insert(-1, (e0[0] - out[0] * stub, e0[1] - out[1] * stub))
    return out_pts


def corner_centre(a, b):
    """Centre of the curb rounding between adjacent arms a, b (either order)."""
    if (a, b) not in CORNER_OF:
        a, b = b, a
    R = CORNER_R[CORNER_OF[(a, b)]]
    (_, va), (_, vb) = frame(a), frame(b)
    ta, tb = edges(a)[3] - R, edges(b)[0] + R
    det = va[0] * vb[1] - va[1] * vb[0]
    return ((ta * vb[1] - tb * va[1]) / det, (va[0] * tb - vb[0] * ta) / det)


def offset_left(points, dist, start_dir=None, end_dir=None):
    """Parallel curve ``dist`` metres to the LEFT of the travel direction.

    ``start_dir`` / ``end_dir`` fix the normal on the two first / last points
    (the straight end stubs), so the offset meets the approach and exit exactly.
    """
    res = []
    for i, p in enumerate(points):
        a = points[max(0, i - 1)]; b = points[min(len(points) - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if start_dir is not None and i <= 1:
            dx, dy = start_dir
        elif end_dir is not None and i >= len(points) - 2:
            dx, dy = end_dir
        m = math.hypot(dx, dy)
        res.append((p[0] - dy / m * dist, p[1] + dx / m * dist))
    return res


def right_turn_lane0(a, b):
    """Curb-sublane lines of the right turn a->b and its curb-concentric arc.

    Returns (s0, e_dir, lane0_radius, s_arc_end_on_b): the stop-line point of
    the curb sublane, the exit direction, the desired curb-sublane radius and
    the exit station (in b's frame) where that arc becomes tangent to the exit.
    """
    A, B = ARMS[a], ARMS[b]
    n_in, n_out = n_sub(A['w_in']), n_sub(B['w_out'])
    t0_in = edges(a)[1] + (n_in - 0.5) * SUBLANE_W
    t0_out = edges(b)[2] - (n_out - 0.5) * SUBLANE_W
    (ua, va), (ub, vb) = frame(a), frame(b)
    # s1*ua + t0_in*va == s2*ub + t0_out*vb
    det = ua[0] * (-ub[1]) - (-ub[0]) * ua[1]
    rx, ry = t0_out * vb[0] - t0_in * va[0], t0_out * vb[1] - t0_in * va[1]
    s2 = (ua[0] * ry - ua[1] * rx) / det
    inc = (-ua[0], -ua[1])
    theta = abs(math.atan2(inc[0] * ub[1] - inc[1] * ub[0], inc[0] * ub[0] + inc[1] * ub[1]))
    residual = max(A['w_in'] - n_in * SUBLANE_W, B['w_out'] - n_out * SUBLANE_W)
    corner = CORNER_OF.get((a, b), CORNER_OF.get((b, a)))
    r0 = CORNER_R[corner] + RIGHT_TURN_CLEARANCE + residual + SUBLANE_W / 2
    return t0_in, t0_out, r0, s2 + r0 * math.tan(theta / 2.0)


_TO_UTM = Transformer.from_crs('EPSG:4326', 'EPSG:32648', always_xy=True)


def ll(pt):
    # GAMA không tự chiếu GeoJSON lon/lat (thế giới sẽ tính bằng độ) → ghi thẳng UTM 48N, đơn vị mét
    x, y = _TO_UTM.transform(LON0 + pt[0] / KX, LAT0 + pt[1] / KY)
    return [round(x, 3), round(y, 3)]


def feature(geom, **props):
    g = mapping(geom)
    conv = lambda ring: [ll(p) for p in ring]
    if g['type'] == 'LineString':
        g = {'type': 'LineString', 'coordinates': conv(g['coordinates'])}
    elif g['type'] == 'Polygon':
        g = {'type': 'Polygon', 'coordinates': [conv(r) for r in g['coordinates']]}
    else:
        raise ValueError(g['type'])
    return {'type': 'Feature', 'geometry': g, 'properties': props}


def build():
    BIG = ARM_LEN * 3
    # --- khối vỉa hè 4 góc: phần nằm ngoài cả hai mép ngoài của 2 nhánh kề nhau, bo góc bán kính R
    blocks = []
    for i, a in enumerate(CLOCKWISE):
        b = CLOCKWISE[(i + 1) % 4]
        t_out_a = edges(a)[3]  # mép ngoài chiều ra của a (phía phải khi nhìn ra ngoài)
        t_in_b = edges(b)[0]   # mép ngoài chiều vào của b (phía trái khi nhìn ra ngoài)
        # nửa mặt phẳng t < t_out_a (trong khung a) ∩ nửa mặt phẳng t > t_in_b (trong khung b)
        ha = Polygon([P(a, -BIG, t_out_a), P(a, BIG, t_out_a), P(a, BIG, t_out_a - BIG), P(a, -BIG, t_out_a - BIG)])
        hb = Polygon([P(b, -BIG, t_in_b), P(b, BIG, t_in_b), P(b, BIG, t_in_b + BIG), P(b, -BIG, t_in_b + BIG)])
        R = CORNER_R[CORNER_OF[(a, b)]]
        blk = ha.intersection(hb).intersection(Point(0, 0).buffer(ARM_LEN + 30)).buffer(-R, join_style=1).buffer(R, join_style=1)
        blocks.append(blk)
    disk = Point(0, 0).buffer(ARM_LEN, resolution=64)
    paved = disk.difference(unary_union(blocks))
    curb_blocks = blocks

    medians = []
    for a, A in ARMS.items():
        _, ti, to, _ = edges(a)
        m = Polygon([P(a, A['nose'], ti), P(a, ARM_LEN + 5, ti), P(a, ARM_LEN + 5, to), P(a, A['nose'], to)])
        # mũi dải phân cách bo tròn theo nửa bề rộng (đúng kiểu mũi tròn trên ảnh vệ tinh)
        r = A['med'] / 2
        m = m.union(Point(P(a, A['nose'] + r, A['c'])).buffer(r)).difference(
            Polygon([P(a, A['nose'] - 50, ti + 1), P(a, A['nose'], ti + 1), P(a, A['nose'], to - 1), P(a, A['nose'] - 50, to - 1)]))
        medians.append(m.intersection(disk))
    road_surface = paved.difference(unary_union(medians))

    # --- vùng xung đột: mặt đường nằm phía trong vạch dừng (chiều vào) và mép ngoài vạch qua đường (chiều ra)
    cuts = []
    for a, A in ARMS.items():
        to_in, ti, to, to_out = edges(a)
        cuts.append(Polygon([P(a, A['stop'], ti - 0.01), P(a, BIG, ti - 0.01), P(a, BIG, to_in + BIG), P(a, A['stop'], to_in + BIG)]))
        cuts.append(Polygon([P(a, A['cw'][1], to_out - BIG), P(a, BIG, to_out - BIG), P(a, BIG, ti - 0.01), P(a, A['cw'][1], ti - 0.01)]))
    conflict = road_surface.difference(unary_union(cuts))
    if conflict.geom_type != 'Polygon':
        conflict = max(conflict.geoms, key=lambda g: g.area)

    areas, roads = [], []
    areas.append(feature(road_surface if road_surface.geom_type == 'Polygon' else max(road_surface.geoms, key=lambda g: g.area), kind='asphalt'))
    for m in medians:
        areas.append(feature(m if m.geom_type == 'Polygon' else max(m.geoms, key=lambda g: g.area), kind='median'))
    areas.append(feature(conflict, kind='conflict', area_m2=round(conflict.area, 1)))

    # Điểm bắt đầu nhánh ra: mép ngoài vạch qua đường, lùi ra xa thêm nếu cung
    # rẽ phải bám bó vỉa góc còn cong quá vạch qua đường (góc nhọn, ví dụ góc E).
    exit_s = {b_: ARMS[b_]['cw'][1] for b_ in ARMS}
    for i, a in enumerate(CLOCKWISE):
        b_ = CLOCKWISE[(i - 1) % 4]  # rẽ phải từ a sang nhánh kế bên ngược chiều kim đồng hồ
        _, _, _, s_end = right_turn_lane0(a, b_)
        exit_s[b_] = max(exit_s[b_], min(s_end + 0.3, ARMS[b_]['cw'][1] + MAX_EXIT_SHIFT))
    for a, A in ARMS.items():
        t_in_out, ti, to, t_out_out = edges(a)
        # vạch qua đường (cắt theo mặt đường để không chờm lên vỉa hè / dải phân cách)
        cw = Polygon([P(a, A['cw'][0], t_out_out - 1), P(a, A['cw'][1], t_out_out - 1), P(a, A['cw'][1], t_in_out + 1), P(a, A['cw'][0], t_in_out + 1)])
        cw = cw.intersection(paved)
        for g in (cw.geoms if hasattr(cw, 'geoms') else [cw]):
            if g.area > 1: areas.append(feature(g, kind='crosswalk', arm=a))
        areas.append(feature(LineString([P(a, A['stop'], ti), P(a, A['stop'], t_in_out)]), kind='stopline', arm=a))
        # đường chia sublane (chỉ để hiển thị)
        for side, t0, w, s0 in (('in', ti, A['w_in'], A['stop']), ('out', to, -A['w_out'], A['cw'][1])):
            n = n_sub(abs(w))
            for k in range(1, n):
                t = t0 + math.copysign(k * SUBLANE_W, w)
                areas.append(feature(LineString([P(a, s0, t), P(a, ARM_LEN, t)]), kind='sublane', arm=a, side=side))

        # --- road (đồ thị advanced_driving). Polyline nằm ở mép phía dải phân cách; GAMA vẽ xe lệch sang PHẢI, sublane 0 sát bó vỉa
        n_in, n_out = n_sub(A['w_in']), n_sub(A['w_out'])
        roads.append(feature(LineString([P(a, ARM_LEN, ti), P(a, A['stop'], ti)]), id=f'{a}_in', kind='approach', arm=a, to_arm='',
                             turn='', num_lanes=n_in, width=A['w_in'], name=A['name']))
        roads.append(feature(LineString([P(a, exit_s[a], to), P(a, ARM_LEN, to)]), id=f'{a}_out', kind='exit', arm=a, to_arm='',
                             turn='', num_lanes=n_out, width=A['w_out'], name=A['name']))

    # --- connector: rẽ trái/thẳng dùng Bezier bậc hai; rẽ phải dùng cung Bezier cubic tiếp tuyến với hai nhánh
    turn_of = {1: 'left', 2: 'through', 3: 'right'}  # số bước theo chiều kim đồng hồ từ nhánh vào tới nhánh ra
    for i, a in enumerate(CLOCKWISE):
        for j, b in enumerate(CLOCKWISE):
            k = (j - i) % 4
            if k == 0: continue
            turn = turn_of[k]  # xe đi bên phải: rẽ phải = nhánh kế bên ngược chiều kim đồng hồ (k = 3)
            A, B = ARMS[a], ARMS[b]
            p0, p1 = P(a, A['stop'], edges(a)[1]), P(b, exit_s[b], edges(b)[2])
            (ua, _), (ub, _) = frame(a), frame(b)
            d0 = (-ua[0], -ua[1])  # hướng đi vào tâm
            # điểm điều khiển: giao của 2 tia (vào theo d0, ra ngược ub); nếu gần song song (đi thẳng) lấy trung điểm
            den = d0[0] * (-ub[1]) - d0[1] * (-ub[0])
            if abs(den) < 0.2:
                c = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            else:
                lam = ((p1[0] - p0[0]) * (-ub[1]) - (p1[1] - p0[1]) * (-ub[0])) / den
                c = (p0[0] + lam * d0[0], p0[1] + lam * d0[1])
            n = min(n_sub(A['w_in']), n_sub(B['w_out']))
            if turn == 'right':
                # Rẽ phải bám theo bó vỉa góc, không theo mép dải phân cách:
                # - sublane k của nhánh rẽ phải trùng sublane k (tính từ bó vỉa) của
                #   nhánh vào và nhánh ra, nên xe không bị dịch ngang ở vạch dừng;
                # - mọi sublane là cung tròn đồng tâm, tiếp tuyến với làn của nó,
                #   sublane 0 cách bó vỉa góc (bán kính CORNER_R) một khe hở cố định.
                n_in, n_out = n_sub(A['w_in']), n_sub(B['w_out'])
                n = min(n, RIGHT_TURN_SUBLANES)
                # tâm sublane 0 (sát bó vỉa) trên nhánh vào / nhánh ra (đúng lưới sublane của hai nhánh)
                t0_in, t0_out, lane0_radius, _ = right_turn_lane0(a, b)
                s0, e0 = P(a, A['stop'], t0_in), P(b, exit_s[b], t0_out)
                curb = unary_union(curb_blocks).boundary
                lane0 = right_turn_curb_path(s0, e0, d0, ub, lane0_radius, curb,
                                             SUBLANE_W / 2 + RIGHT_TURN_CLEARANCE, corner_centre(a, b))
                # polyline của road = mép trái của n sublane: song song với sublane 0
                pts = offset_left(lane0, (n - 0.5) * SUBLANE_W, d0, ub)
                pts = [q for k, q in enumerate(pts) if k in (0, len(pts) - 1) or math.dist(q, pts[k - 1]) > 0.2]
            else:
                pts = [((1 - q) ** 2 * p0[0] + 2 * (1 - q) * q * c[0] + q * q * p1[0],
                        (1 - q) ** 2 * p0[1] + 2 * (1 - q) * q * c[1] + q * q * p1[1])
                       for q in (x / 12 for x in range(13))]
                # Match the approach tangent at the first polyline segment.
                # A sampled Bezier otherwise cuts across adjacent sublanes
                # immediately after the stop line, even when its analytic
                # tangent is correct at p0.
                first_length = math.dist(pts[0], pts[1])
                pts[1] = (p0[0] + first_length * d0[0],
                          p0[1] + first_length * d0[1])
                # The final polyline segment must share the exit tangent.
                # Otherwise an offset sublane projects behind the exit start
                # and the simulator has to jump the body sideways at the seam.
                final_length = math.dist(pts[-2], pts[-1])
                pts[-2] = (p1[0] - final_length * ub[0],
                           p1[1] - final_length * ub[1])
            roads.append(feature(LineString(pts), id=f'{a}_{b}', kind='connector', arm=a, to_arm=b, turn=turn, num_lanes=n,
                                 width=round(n * SUBLANE_W, 2), name=''))
    return roads, areas, conflict, paved


def angles():
    out = []
    for i, a in enumerate(CLOCKWISE):
        b = CLOCKWISE[(i + 1) % 4]
        out.append((a, b, (ARMS[b]['bearing'] - ARMS[a]['bearing']) % 360))
    return out


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # console Windows mặc định cp1252
    here =os.path.dirname(os.path.abspath(__file__))
    inc = os.path.join(here, '..', 'simulation', 'python_simulation', 'includes')
    gama_inc = os.path.join(here, '..', '..', 'gama-simulation', 'simulation', 'includes')
    for target in (inc, gama_inc):
        os.makedirs(target, exist_ok=True)
    roads, areas, conflict, _ = build()
    for fn, feats in (('roads.geojson', roads), ('areas.geojson', areas)):
        # GAMA lấy schema từ feature đầu tiên → mọi feature phải có cùng bộ thuộc tính, cùng kiểu
        keys = {}
        for f in feats:
            for k, v in f['properties'].items():
                keys.setdefault(k, 0.0 if isinstance(v, (int, float)) else '')
        for f in feats:
            f['properties'] = {k: f['properties'].get(k, d) for k, d in keys.items()}
        collection = {'type': 'FeatureCollection', 'crs': {'type': 'name', 'properties': {'name': 'EPSG:32648'}},
                      'features': feats}
        for target in (inc, gama_inc):
            with open(os.path.join(target, fn), 'w', encoding='utf-8') as f:
                json.dump(collection, f, ensure_ascii=False)
    # self-check: tổng góc = 360, sublane đúng công thức, vùng xung đột là 1 đa giác hợp lệ
    assert abs(sum(x[2] for x in angles()) - 360) < 1e-6
    assert all(r['properties']['num_lanes'] == n_sub(r['properties']['width']) for r in roads)
    assert conflict.is_valid and conflict.area > 500
    print(f'{"nhánh":4} {"chiều":5} {"rộng(m)":>8} {"sublane":>8} {"dư(m)":>6}')
    for a, A in ARMS.items():
        for side in ('in', 'out'):
            w = A['w_' + side]
            print(f'{a:4} {side:5} {w:8.1f} {n_sub(w):8d} {w - n_sub(w) * SUBLANE_W:6.2f}')
    print('góc giữa các nhánh kề nhau:', ', '.join(f'{a}-{b} {g:.1f}°' for a, b, g in angles()))
    print(f'vùng xung đột: {conflict.area:.0f} m²; roads: {len(roads)}; areas: {len(areas)}')
