"""Measure in-junction body overlaps, throughput and delays for LeVanLuongSimulator."""
import argparse, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path

def run(simdir, seed, ticks, flow, extra):
    sys.path.insert(0, str(simdir))
    import python_sim as ps
    from vehicle_geometry import footprint, polygons_overlap
    sim = ps.LeVanLuongSimulator(seed=seed, flow_per_approach=flow, **extra)
    pairs = Counter(); pair_ticks = Counter(); first = {}
    created = {}; maneuver = {}; enter_conn = {}; exit_conn = {}; near_stop = {}
    stop_green = 0.0
    prev_speed = {}; hard_brakes = 0; traj = defaultdict(list); movement = {}
    t0 = time.time()
    for _ in range(ticks):
        sim.step(1)
        now = sim.cycle * ps.DT
        polys = []
        live = set()
        for v in sim.vehicles:
            vid = v.creation_id; live.add(vid)
            created.setdefault(vid, sim.cycle); maneuver[vid] = v.maneuver
            road, prog = sim._road_state(v)
            pv = prev_speed.get(vid)
            if pv is not None and (pv - v.speed) / ps.DT > 6.0:
                hard_brakes += 1
            prev_speed[vid] = v.speed
            if road == f"{v.arm}_{v.to_arm}" or (road == f"{v.arm}_in" and v.approach_end - v.position < 6):
                movement.setdefault(vid, sim._lane_movement_id(v, on_connector=road == f"{v.arm}_{v.to_arm}"))
                traj[vid].append((now, v.position + v.length / 2 - v.approach_end, v.length))
            if road == f"{v.arm}_{v.to_arm}":
                enter_conn.setdefault(vid, now)
            elif road == f"{v.to_arm}_out":
                enter_conn.setdefault(vid, now); exit_conn.setdefault(vid, now)
            elif road == f"{v.arm}_in":
                if v.approach_end - v.position < 12.0:
                    near_stop.setdefault(vid, now)
                green = sim._arm_is_green(v.arm)
                if (green and sim.green_ticks * ps.DT > 3.0 and v.speed < 0.1 and
                        0 <= sim._stop_position(v) - v.position < 1.0):
                    stop_green += ps.DT
            if road.endswith("_in") and v.approach_end - v.position > 8.0:
                continue
            if road.endswith("_out") and prog > 12.0:
                continue
            pose = sim._geometry_roads[road].pose(prog, v.current_lateral_shift)
            kind = "car" if v.occupied == 2 else "motorbike"
            polys.append((vid, road, v, footprint(pose, kind)))
        for i in range(len(polys)):
            ia, ra, va, pa = polys[i]
            for j in range(i + 1, len(polys)):
                ib, rb, vb, pb = polys[j]
                if ra == rb:
                    continue  # same-road following is audited elsewhere
                conn = (ra == f"{va.arm}_{va.to_arm}") or (rb == f"{vb.arm}_{vb.to_arm}")
                if not conn:
                    continue
                if polygons_overlap(pa, pb):
                    key = (min(ia, ib), max(ia, ib))
                    cls = "-".join(sorted((va.maneuver, vb.maneuver))) + ("|same_arm" if va.arm == vb.arm else "")
                    if key not in pairs:
                        first[key] = (sim.cycle, ra, rb)
                    pairs[key] = cls; pair_ticks[cls] += 1
    by_cls = Counter(pairs.values())
    tt = defaultdict(list); jt = defaultdict(list)
    for vid, t in exit_conn.items():
        if vid in near_stop:
            jt[maneuver[vid]].append(t - near_stop[vid])
    # Post-encroachment time on sublane conflict zones (cross-arm pairs).
    import junction_negotiation as jn
    zm = jn.ZoneMap(lambda m: jn.SimulatorAdapter._sample(type("S", (), {"sim": sim, "sample_step": 0.5})(), m))
    def cross_times(tr, z_in, z_out):
        t_in = t_out = None
        for (t, f, L) in tr:
            if t_in is None and f > z_in: t_in = t
            if t_in is not None and f - L >= z_out: t_out = t; break
        return t_in, t_out
    vids = [v for v in traj if v in movement and movement[v] in sim._lane_movement_defs]
    pets = []; pet_by = defaultdict(list)
    for i in range(len(vids)):
        a = vids[i]
        for b in vids[i+1:]:
            if movement[a].split("@")[0].split("_")[0] == movement[b].split("@")[0].split("_")[0]:
                continue
            if abs(traj[a][0][0] - traj[b][0][0]) > 60: continue
            for z in zm.zones(movement[a], movement[b]):
                ia, oa = cross_times(traj[a], z.a_in, z.a_out)
                ib, ob = cross_times(traj[b], z.b_in, z.b_out)
                if None in (ia, oa, ib, ob): continue
                pet = (ib - oa) if ia <= ib else (ia - ob)
                pets.append(pet)
                pet_by["-".join(sorted((maneuver[a], maneuver[b])))].append(pet)
    pets.sort()
    def q(xs, f): return round(sorted(xs)[int(f*(len(xs)-1))], 2) if xs else None
    neg = getattr(sim, "_negotiation", None)
    res_extra = dict(pet_n=len(pets), pet_p05=q(pets, .05), pet_p50=q(pets, .5),
                     pet_lt1=sum(p < 1.0 for p in pets), pet_lt0=sum(p < 0 for p in pets),
                     pet_by={k: dict(n=len(v), p05=q(v, .05), lt1=sum(x < 1 for x in v)) for k, v in pet_by.items()},
                     hard_brakes=hard_brakes,
                     neg_stats=(neg.negotiator.stats if neg else None),
                     neg_hard_caps=(neg.hard_caps if neg else None), filter_moves=(getattr(neg, "filter_moves", None) if neg else None))
    res = dict(seed=seed, ticks=ticks, flow=flow, extra={k: str(v) for k, v in extra.items()},
               created=sim.created, passed=sim.passed, active=len(sim.vehicles),
               overlap_pairs=len(pairs), overlap_pairs_by_class=dict(by_cls),
               overlap_pair_ticks=dict(pair_ticks), stop_on_green_s=round(stop_green, 1),
               junction_time_mean={k: round(sum(v)/len(v), 2) for k, v in jt.items() if v},
               junction_time_p90={k: round(sorted(v)[int(0.9*len(v))], 2) for k, v in jt.items() if v},
               n_exited={k: len(v) for k, v in jt.items()},
               weighted_delay=round(sim.weighted_delay_integral, 1),
               runtime_s=round(time.time()-t0, 1),
               first_overlaps=[(k, first[k]) for k in list(first)[:5]], **res_extra)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", default=str(Path(__file__).resolve().parents[1] / "python-simulation" / "simulation" / "python_simulation")); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=1000); ap.add_argument("--flow", type=float, default=2400)
    ap.add_argument("--extra", default="{}")
    a = ap.parse_args()
    print(json.dumps(run(Path(a.simdir), a.seed, a.ticks, a.flow, json.loads(a.extra))))
