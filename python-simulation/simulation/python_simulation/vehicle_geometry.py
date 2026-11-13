"""Small, dependency-free road and vehicle geometry helpers (metres, EPSG:32648)."""
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

CAR_WIDTH, CAR_LENGTH = 1.7, 4.5
MOTORBIKE_WIDTH, MOTORBIKE_LENGTH = 0.7, 1.9
SUBLANE_WIDTH = 1.05


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    tx: float
    ty: float
    heading: float
    progress: float


@dataclass(frozen=True)
class Road:
    points: tuple
    num_lanes: int = 1
    lengths: tuple = ()
    cumulative: tuple = ()
    vertex_tangents: tuple = ()

    def __init__(self, points: Iterable[Sequence[float]], num_lanes: int = 1):
        pts = tuple((float(p[0]), float(p[1])) for p in points)
        if int(num_lanes) <= 0:
            raise ValueError("num_lanes must be positive")
        if len(pts) < 2:
            raise ValueError("a road needs at least two points")
        lens = tuple(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(pts,pts[1:]))
        if any(v <= 0 for v in lens):
            raise ValueError("road segments must have positive length")
        cum=[0.0]
        for v in lens: cum.append(cum[-1]+v)
        seg_t = tuple(((b[0]-a[0])/l, (b[1]-a[1])/l) for (a,b), l in zip(zip(pts,pts[1:]), lens))
        v_tan = []
        for idx in range(len(pts)):
            if idx == 0:
                v_tan.append((seg_t[0], seg_t[0]))
            elif idx == len(pts) - 1:
                v_tan.append((seg_t[-1], seg_t[-1]))
            else:
                t_prev, t_next = seg_t[idx - 1], seg_t[idx]
                dot = t_prev[0] * t_next[0] + t_prev[1] * t_next[1]
                if dot > 0.90:
                    sx, sy = t_prev[0] + t_next[0], t_prev[1] + t_next[1]
                    m = math.hypot(sx, sy)
                    avg = (sx / m, sy / m) if m > 1e-6 else t_next
                    v_tan.append((avg, avg))
                else:
                    v_tan.append((t_prev, t_next))
        object.__setattr__(self,"points",pts); object.__setattr__(self,"num_lanes",int(num_lanes))
        object.__setattr__(self,"lengths",lens); object.__setattr__(self,"cumulative",tuple(cum))
        object.__setattr__(self,"vertex_tangents",tuple(v_tan))

    @classmethod
    def from_geojson(cls, feature):
        if feature["geometry"]["type"] != "LineString":
            raise ValueError("expected a GeoJSON LineString")
        return cls(feature["geometry"]["coordinates"], feature.get("properties",{}).get("num_lanes",1))

    @property
    def length(self): return self.cumulative[-1]

    def pose(self, progress: float, lateral_shift: float = 0.0) -> Pose:
        """Return lane-center pose. lateral_shift follows observer.html's SUBLANE_WIDTH convention."""
        s=float(progress)
        if s < 0: i=0; t=s/self.lengths[0]
        elif s >= self.length: i=len(self.lengths)-1; t=(s-self.cumulative[i])/self.lengths[i]
        else:
            i=next(j for j in range(len(self.lengths)) if s < self.cumulative[j+1])
            t=(s-self.cumulative[i])/self.lengths[i]
        a,b=self.points[i],self.points[i+1]; dx,dy=b[0]-a[0],b[1]-a[1]; seg=self.lengths[i]
        if self.vertex_tangents:
            t0 = self.vertex_tangents[i][1]
            t1 = self.vertex_tangents[i+1][0]
            f = min(1.0, max(0.0, t))
            lx = t0[0] + (t1[0] - t0[0]) * f
            ly = t0[1] + (t1[1] - t0[1]) * f
            lm = math.hypot(lx, ly)
            tx, ty = (lx / lm, ly / lm) if lm > 1e-9 else (dx / seg, dy / seg)
        else:
            tx,ty=dx/seg,dy/seg
        # Same right-normal convention as observer.html.
        dist=self.num_lanes*SUBLANE_WIDTH-float(lateral_shift)-SUBLANE_WIDTH/2.0
        return Pose(a[0]+dx*t+ty*dist, a[1]+dy*t-tx*dist, tx,ty, math.atan2(ty,tx),s)


class RouteGeometry:
    """Ordered roads joined by cumulative progress; seams retain source geometry exactly."""
    def __init__(self, roads):
        # Accept (road_id, Road) pairs or GeoJSON features.
        items=[]
        for item in roads:
            if isinstance(item, tuple) and len(item)==2 and isinstance(item[1],Road):
                items.append((str(item[0]),item[1]))
            else:
                props=item.get("properties",{})
                items.append((str(props.get("id",len(items))),Road.from_geojson(item)))
        if not items: raise ValueError("route needs at least one road")
        self.roads=tuple(items)
        cum=[0.0]
        for _,road in items: cum.append(cum[-1]+road.length)
        self.seams=tuple(cum[1:-1]); self.cumulative=tuple(cum)

    @property
    def length(self): return self.cumulative[-1]

    def pose(self, progress, lateral_shift=0.0, shifts=None):
        """Return (road_id, Pose); shifts may map road IDs to per-road lane offsets."""
        s=float(progress)
        if s < 0: i=0
        elif s >= self.length: i=len(self.roads)-1
        else: i=next(j for j in range(len(self.roads)) if s < self.cumulative[j+1])
        road_id,road=self.roads[i]
        shift=(shifts.get(road_id,lateral_shift) if shifts is not None else lateral_shift)
        return road_id,road.pose(s-self.cumulative[i],shift)

    def seam_discontinuities(self, lateral_shift=0.0, shifts=None):
        """Measured left/right center-position delta at each seam (next pose minus prior endpoint)."""
        out=[]
        for i,s in enumerate(self.seams):
            aid,a=self.roads[i]; bid,b=self.roads[i+1]
            sa=shifts.get(aid,lateral_shift) if shifts is not None else lateral_shift
            sb=shifts.get(bid,lateral_shift) if shifts is not None else lateral_shift
            pa=a.pose(a.length,sa); pb=b.pose(0,sb)
            out.append((pb.x-pa.x,pb.y-pa.y))
        return tuple(out)


def vehicle_size(kind="car"):
    if kind == "car": return CAR_LENGTH,CAR_WIDTH
    if kind in ("motorbike","motorcycle","bike"): return MOTORBIKE_LENGTH,MOTORBIKE_WIDTH
    raise ValueError(f"unknown vehicle kind: {kind!r}")


def footprint(pose: Pose, kind="car", length=None, width=None):
    """Four rectangle corners in counter-clockwise order."""
    L,W=vehicle_size(kind); L=L if length is None else float(length); W=W if width is None else float(width)
    fx,fy=pose.tx,pose.ty; rx,ry=fy,-fx
    return tuple((pose.x+sx*L/2*fx+sy*W/2*rx, pose.y+sx*L/2*fy+sy*W/2*ry)
                 for sx,sy in ((1,1),(-1,1),(-1,-1),(1,-1)))


def polygons_overlap(a, b, clearance=0.0):
    """SAT overlap; touching counts as overlap. clearance expands both polygons."""
    if clearance < 0: raise ValueError("clearance must be nonnegative")
    # Cheap broad phase before the separating-axis projections.
    if (max(p[0] for p in a)+clearance < min(p[0] for p in b)-clearance or
        max(p[0] for p in b)+clearance < min(p[0] for p in a)-clearance or
        max(p[1] for p in a)+clearance < min(p[1] for p in b)-clearance or
        max(p[1] for p in b)+clearance < min(p[1] for p in a)-clearance): return False
    for poly in (a,b):
        for p,q in zip(poly,poly[1:]+poly[:1]):
            ex,ey=q[0]-p[0],q[1]-p[1]; n=math.hypot(ex,ey)
            if n == 0: continue
            ax,ay=-ey/n,ex/n
            pa=[x*ax+y*ay for x,y in a]; pb=[x*ax+y*ay for x,y in b]
            if max(pa)+clearance < min(pb)-clearance or max(pb)+clearance < min(pa)-clearance: return False
    return True


def vehicles_overlap(a: Pose, b: Pose, kind_a="car", kind_b="car", clearance=0.0):
    return polygons_overlap(footprint(a,kind_a),footprint(b,kind_b),clearance)


def swept_overlap(road_a: Road, start_a: float, end_a: float, road_b: Road, start_b: float, end_b: float,
                  kind_a="car", kind_b="car", shift_a=0.0, shift_b=0.0, clearance=0.0, max_step=0.1,
                  end_shift_a=None, end_shift_b=None):
    """Conservative sampled sweep over a tick. Inflation covers translation between samples;
    heading changes are sampled at <=5 degrees. Roads supply the curved path.
    """
    if max_step <= 0: raise ValueError("max_step must be positive")
    end_shift_a=shift_a if end_shift_a is None else end_shift_a
    end_shift_b=shift_b if end_shift_b is None else end_shift_b
    da=max(abs(end_a-start_a),abs(end_b-start_b),abs(end_shift_a-shift_a),abs(end_shift_b-shift_b))
    n=max(1, math.ceil(da/max_step))
    # Also limit angular spacing on polyline turns by finer spatial samples.
    n=max(n, math.ceil(da/0.05))
    La,Wa=vehicle_size(kind_a); Lb,Wb=vehicle_size(kind_b)
    # Half-step expansion bounds positional error at each sample; turn approximation is covered by 5cm steps.
    pad=da/(2*n)
    for i in range(n+1):
        t=i/n
        pa=road_a.pose(start_a+(end_a-start_a)*t,shift_a+(end_shift_a-shift_a)*t)
        pb=road_b.pose(start_b+(end_b-start_b)*t,shift_b+(end_shift_b-shift_b)*t)
        if isinstance(pa,tuple): pa=pa[-1]
        if isinstance(pb,tuple): pb=pb[-1]
        if polygons_overlap(footprint(pa,length=La+2*pad,width=Wa+2*pad),
                            footprint(pb,length=Lb+2*pad,width=Wb+2*pad),clearance): return True
    return False
