"""Sampled full-body motion guard for polyline road routes.

The guard checks synchronized oriented vehicle rectangles over one proposed tick.
It uses a swept-AABB broad phase, route vertex/seam samples, and a conservative
translation pad between samples. It is a practical sampled approximation, not
a continuous-time collision proof. Keep tick displacement and fleet size modest;
the narrow phase is quadratic in candidate pairs and linear in route samples.
"""
from dataclasses import dataclass
import math
from typing import Iterable, Optional

try:
    from .vehicle_geometry import RouteGeometry, footprint, polygons_overlap
except ImportError:  # direct module import when this folder is on sys.path
    from vehicle_geometry import RouteGeometry, footprint, polygons_overlap


@dataclass(frozen=True)
class Motion:
    """One vehicle's motion over a tick, expressed in global route metres.

    ``progress_start/end`` and ``lateral_start/end`` are linear in tick fraction.
    Stationary traffic uses equal start/end values. ``vehicle_id`` must be stable
    and unique within a scene; IDs provide deterministic result tie breaking.
    """
    vehicle_id: str
    route: RouteGeometry
    progress_start: float
    progress_end: float
    lateral_start: float = 0.0
    lateral_end: float = 0.0
    length: float = 4.5
    width: float = 1.7

    def __post_init__(self):
        if not self.vehicle_id:
            raise ValueError("vehicle_id must be nonempty")
        if self.progress_end < self.progress_start:
            raise ValueError("backwards route progress is not supported")
        if self.length <= 0 or self.width <= 0:
            raise ValueError("vehicle length and width must be positive")
        vals=(self.progress_start,self.progress_end,self.lateral_start,self.lateral_end,
              self.length,self.width)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("motion values must be finite")


@dataclass(frozen=True)
class Contact:
    fraction: float
    vehicle_id: str
    start_overlap: bool = False


def _at(m: Motion, t: float):
    s=m.progress_start+(m.progress_end-m.progress_start)*t
    shift=m.lateral_start+(m.lateral_end-m.lateral_start)*t
    return m.route.pose(s,shift)[1]


def _events(m: Motion):
    """Fractions at route vertices, plus fine spatial samples on each span."""
    ds=m.progress_end-m.progress_start
    # A stationary vehicle may still change lanes.
    travel=max(ds,abs(m.lateral_end-m.lateral_start))
    n=max(1,math.ceil(travel/0.10))
    ts={i/n for i in range(n+1)}
    if ds > 0:
        for seam in m.route.seams:
            t=(seam-m.progress_start)/ds
            if 0 < t < 1:
                ts.update((max(0.0,t-1e-9),t,min(1.0,t+1e-9)))
        for i,(_,road) in enumerate(m.route.roads):
            base=m.route.cumulative[i]
            for vertex in road.cumulative[1:-1]:
                t=(base+vertex-m.progress_start)/ds
                if 0 < t < 1:
                    ts.update((max(0.0,t-1e-9),t,min(1.0,t+1e-9)))
    return sorted(ts)


def _bounds(m: Motion):
    # Circumradius bound around every center on the route sweep.
    r=math.hypot(m.length,m.width)/2
    poses=[_at(m,t) for t in _events(m)]
    return (min(p.x for p in poses)-r, min(p.y for p in poses)-r,
            max(p.x for p in poses)+r, max(p.y for p in poses)+r)


def _broad_overlap(a, b, clearance):
    aa=_bounds(a); bb=_bounds(b)
    c=clearance
    return not (aa[2]+c < bb[0] or bb[2]+c < aa[0] or
                aa[3]+c < bb[1] or bb[3]+c < aa[1])


def _sweep_pair(a: Motion, b: Motion, clearance: float, max_step: float):
    """Return first sampled/conservatively padded pair contact, or None."""
    if not _broad_overlap(a,b,clearance):
        return None
    ts=sorted(set(_events(a)) | set(_events(b)))
    # Between events, each center follows a straight segment with fixed heading.
    # Add subdivision by center travel so the translation pad stays tight.
    dense=set(ts)
    for lo,hi in zip(ts,ts[1:]):
        p0,p1=_at(a,lo),_at(a,hi); q0,q1=_at(b,lo),_at(b,hi)
        d=max(math.hypot(p1.x-p0.x,p1.y-p0.y),math.hypot(q1.x-q0.x,q1.y-q0.y))
        n=max(1,math.ceil(d/max_step))
        dense.update(lo+(hi-lo)*i/n for i in range(1,n))
    ts=sorted(dense)
    # Report only actual OBB intersection as start_overlap. The configured
    # clearance and translation padding can make an impending contact appear at
    # fraction zero, but those are not an overlap at the initial snapshot.
    pa0,pb0=_at(a,0.0),_at(b,0.0)
    if polygons_overlap(footprint(pa0,length=a.length,width=a.width),
                        footprint(pb0,length=b.length,width=b.width),0.0):
        return (0.0, True)
    # Clearance is Minkowski-expanded by polygons_overlap on both bodies, so
    # split the requested gap and the maximum relative travel pad.
    for ix,t in enumerate(ts):
        pa,pb=_at(a,t),_at(b,t)
        pad=0.0
        if ix+1 < len(ts):
            dt=ts[ix+1]-t
            pa1,pb1=_at(a,ts[ix+1]),_at(b,ts[ix+1])
            rel=math.hypot((pa1.x-pa.x)-(pb1.x-pb.x),(pa1.y-pa.y)-(pb1.y-pb.y))
            # A collision may lie halfway to the next synchronized sample.
            pad=rel*0.5
        if polygons_overlap(footprint(pa,length=a.length,width=a.width),
                            footprint(pb,length=b.length,width=b.width),
                            clearance*0.5+pad*0.5):
            return (t, False)
    return None


def first_contact(motion: Motion, others: Iterable[Motion], *, clearance: float=0.05,
                  max_step: float=0.10) -> Optional[Contact]:
    """Find earliest conservative contact fraction with any other motion.

    Returns ``None`` when the complete tick is clear. A fraction of zero with
    ``start_overlap=True`` means the bodies overlap at tick start. Clearance is
    metres of additional separation required. Results break equal-time ties by
    vehicle ID. No reverse progress is accepted by :class:`Motion`.
    """
    if clearance < 0 or max_step <= 0:
        raise ValueError("clearance must be nonnegative and max_step positive")
    contacts=[]
    for other in others:
        if other.vehicle_id == motion.vehicle_id:
            continue
        hit=_sweep_pair(motion,other,clearance,max_step)
        if hit is not None:
            contacts.append(Contact(hit[0],other.vehicle_id,hit[1]))
    return min(contacts,key=lambda c:(c.fraction,c.vehicle_id)) if contacts else None


def first_contact_fraction(motion: Motion, others: Iterable[Motion], *, clearance: float=0.05,
                           max_step: float=0.10) -> Optional[float]:
    """Convenience form returning just the earliest conservative contact fraction."""
    hit=first_contact(motion,others,clearance=clearance,max_step=max_step)
    return None if hit is None else hit.fraction


def max_safe_fraction(motion: Motion, others: Iterable[Motion], *, clearance: float=0.05,
                      max_step: float=0.10, iterations: int=24) -> float:
    """Return a conservative prefix fraction safe for the proposed tick.

    Start-overlap returns zero. On contact, binary-searches the longest sampled
    collision-free prefix, then backs off a tiny deterministic fraction to avoid
    stopping exactly on a touching SAT boundary. The prefix test is monotone.
    """
    others=tuple(others)
    hit=first_contact(motion,others,clearance=clearance,max_step=max_step)
    if hit is None:
        return 1.0
    if hit.start_overlap or hit.fraction <= 0:
        return 0.0
    lo,hi=0.0,hit.fraction
    for _ in range(iterations):
        mid=(lo+hi)/2
        prefix=Motion(motion.vehicle_id,motion.route,motion.progress_start,
                      motion.progress_start+(motion.progress_end-motion.progress_start)*mid,
                      motion.lateral_start,
                      motion.lateral_start+(motion.lateral_end-motion.lateral_start)*mid,
                      motion.length,motion.width)
        # Evaluate all bodies at the same prefix of the original tick. Re-scaling
        # only the candidate would make it collide against where traffic is at
        # the end of the full tick instead of where it is when we stop it.
        clipped=[]
        for other in others:
            clipped.append(Motion(other.vehicle_id,other.route,other.progress_start,
                                  other.progress_start+(other.progress_end-other.progress_start)*mid,
                                  other.lateral_start,
                                  other.lateral_start+(other.lateral_end-other.lateral_start)*mid,
                                  other.length,other.width))
        if first_contact(prefix,clipped,clearance=clearance,max_step=max_step) is None:
            lo=mid
        else:
            hi=mid
    return max(0.0,lo-1e-6)
