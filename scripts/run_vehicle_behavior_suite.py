#!/usr/bin/env python3
"""
==============================================================================
AUTOMATED VEHICLE BEHAVIOR & EDGE CASE TEST SUITE (PYTHON SIMULATOR)
==============================================================================
Thực thi kiểm thử toàn diện các hành vi xe vi mô (kinematics, IDM, MOBIL),
chu kỳ đèn tín hiệu, quan sát không gian (spatial state), các edge case
cực hạn (zero-traffic, mega-jam, spawn overlap, boundary sublane, UTF-8)
và bảo toàn vật lý trên mô phỏng Python Lê Văn Lương và Legacy.
==============================================================================
"""

import math
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "python-simulation" / "simulation"))
sys.path.insert(0, str(BASE_DIR / "train" / "logic"))

from python_simulation.python_sim import (
    LeVanLuongSimulator, ARMS, DT, road_lengths
)
from python_simulation.demand import generate_demand
from train.logic.lvl_env import _observation
from train.logic.python_env import PythonTrafficEnv


class VehicleBehaviorTestSuite:
    def __init__(self):
        self.results = []
        self.start_time = None

    def assert_test(self, suite, test_id, condition, message):
        status = "PASS" if condition else "FAIL"
        icon = "✅" if condition else "❌"
        print(f"  {icon} [{status}] {suite:12s} | {test_id:32s} | {message}")
        self.results.append({
            "suite": suite,
            "test_id": test_id,
            "status": status,
            "message": message
        })
        return condition

    def run_all(self):
        self.start_time = time.time()
        print("=" * 95)
        print("🧪 BẮT ĐẦU CHẠY BỘ KIỂM THỬ: VEHICLE BEHAVIOR & EDGE CASES (PYTHON ENGINE)")
        print("=" * 95 + "\n")

        self.test_kinematics_and_idm()
        self.test_mobil_lane_changing()
        self.test_traffic_signal_dynamics()
        self.test_spatial_state_and_observations()
        self.test_extreme_edge_cases()
        self.test_legacy_controller_guards()

        return self.print_summary()

    # -------------------------------------------------------------------------
    # 1. KINEMATICS & IDM
    # -------------------------------------------------------------------------
    def test_kinematics_and_idm(self):
        print("--- [1] KIỂM THỬ ĐỘNG HỌC & MÔ HÌNH BÁM ĐUÔI IDM ---")

        # KINE-01: Free road acceleration towards speed limit
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 5, 40.0)  # motorbike, desired 40 km/h = 11.11 m/s
        v = sim.vehicles[0]
        initial_speed = v.speed
        sim.step(10)
        self.assert_test(
            "Kinematics", "KINE-01-FreeRoadAccel",
            v.speed > initial_speed and v.speed <= 40.0 / 3.6 + 1e-4,
            f"Xe tăng tốc mượt mà trên đường vắng ({initial_speed:.2f} -> {v.speed:.2f} m/s <= 11.11 m/s)"
        )

        # KINE-02: Leader follower deceleration (IDM car-following)
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 5, 20.0)  # leader at 20 km/h
        sim.vehicles[0].position = 30.0
        sim.vehicles[0].speed = 20.0 / 3.6
        sim._create_vehicle("NE", "through", False, 5, 40.0)  # follower at 35 km/h
        sim.vehicles[1].position = 10.0
        sim.vehicles[1].speed = 35.0 / 3.6
        initial_follower_spd = sim.vehicles[1].speed
        sim.step(15)
        leader, follower = sim.vehicles[0], sim.vehicles[1]
        gap = leader.position - follower.position - (leader.length + follower.length) / 2
        decelerated = follower.speed < initial_follower_spd
        self.assert_test(
            "Kinematics", "KINE-02-LeaderFollowerDecel",
            gap > 0 and decelerated and follower.lane == leader.lane,
            f"Xe sau giảm tốc duy trì khoảng cách an toàn IDM (gap={gap:.2f}m > 0, v={follower.speed*3.6:.1f} < {initial_follower_spd*3.6:.1f} km/h)"
        )

        # KINE-03: Spawn queue keeps clearance even before the road origin.
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 5, 40.0)
        sim._create_vehicle("NE", "through", False, 5, 40.0)
        sim.step(1)
        speeds_finite = all(math.isfinite(v.speed) and v.speed >= 0 for v in sim.vehicles)
        positions_finite = all(math.isfinite(v.position) for v in sim.vehicles)
        front, rear = sorted(sim.vehicles, key=lambda v: v.position, reverse=True)
        gap = front.position - rear.position - (front.length + rear.length) / 2
        self.assert_test(
            "Kinematics", "KINE-03-SpawnClearance",
            speeds_finite and positions_finite and gap >= max(1.0, front.gap, rear.gap) - 1e-8,
            "Xe mới có thể xếp hàng trước đầu đường; thân xe vẫn cách nhau an toàn"
        )

        # KINE-04: Speed cap adherence
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", True, 5, 30.0)  # Car requested 30 km/h
        sim.step(50)
        car = sim.vehicles[0]
        self.assert_test(
            "Kinematics", "KINE-04-SpeedCapAdherence",
            car.speed <= 30.0 / 3.6 + 1e-5,
            f"Xe ô tô tuân thủ trần tốc độ mong muốn (v={car.speed*3.6:.2f} km/h <= 30 km/h)"
        )

        # KINE-05: Conservation of vehicles
        sim = LeVanLuongSimulator(flow_per_approach=4000, seed=99)
        sim.step(200)
        conservation_ok = (sim.created == len(sim.vehicles) + sim.passed)
        self.assert_test(
            "Kinematics", "KINE-05-VehicleConservation",
            conservation_ok and sim.created > 0,
            f"Bảo toàn số lượng xe tuyệt đối (created={sim.created} = active={len(sim.vehicles)} + passed={sim.passed})"
        )
        print()

    # -------------------------------------------------------------------------
    # 2. MOBIL LANE CHANGING
    # -------------------------------------------------------------------------
    def test_mobil_lane_changing(self):
        print("--- [2] KIỂM THỬ ĐỔI LÀN MOBIL & RÀNG BUỘC SUBLANE ---")

        # MOBIL-01: Cooldown and no unnecessary free-road lane change.
        sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True)
        sim._create_vehicle("NE", "through", False, 10, 40)
        sim.step(19)
        lane_at_19 = sim.vehicles[0].lane
        sim.step(1)
        lane_at_20 = sim.vehicles[0].lane
        sim.step(1)
        lane_at_21 = sim.vehicles[0].lane
        cooldown_elapsed = sim.vehicles[0].time_since_lane_change >= 4.0
        self.assert_test(
            "MOBIL", "MOBIL-01-CooldownEnforcement",
            lane_at_19 == 10 and lane_at_20 == 10 and lane_at_21 == 10 and cooldown_elapsed,
            f"Không đổi làn khi chưa đủ 4s hoặc khi không có lợi ích: tick 19={lane_at_19}, tick 20={lane_at_20}, tick 21={lane_at_21}"
        )

        # MOBIL-02: Boundary clamping (Lane 0 cannot go < 0)
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 0, 40)  # motorbike on sublane 0
        sim.step(50)
        v = sim.vehicles[0]
        self.assert_test(
            "MOBIL", "MOBIL-02-BoundaryClampingZero",
            v.lane >= 0,
            f"Xe tại làn sát mép (lane 0) không bao giờ nhảy ra ngoài biên trái (lane={v.lane})"
        )

        # MOBIL-03: Boundary clamping right edge
        sim = LeVanLuongSimulator(flow_per_approach=0)
        ne_lanes = sim.roads["NE_in"]["num_lanes"]
        sim._create_vehicle("NE", "through", True, ne_lanes - 2, 40)
        sim.step(50)
        car = sim.vehicles[0]
        self.assert_test(
            "MOBIL", "MOBIL-03-BoundaryClampingMax",
            car.lane + car.occupied <= ne_lanes,
            f"Ô tô (2 sublane) không bao giờ vượt biên phải (lane={car.lane}, lane+occupied={car.lane+car.occupied} <= {ne_lanes})"
        )

        # MOBIL-04: NW road sublanes geometry constraint
        sim = LeVanLuongSimulator(flow_per_approach=0)
        nw_lanes = sim.roads["NW_in"]["num_lanes"]
        sim._create_vehicle("NW", "through", True, nw_lanes - 2, 40)
        sim._create_vehicle("NW", "through", False, nw_lanes - 1, 40)
        sim.step(30)
        nw_car = sim.vehicles[0]
        nw_moto = sim.vehicles[1]
        self.assert_test(
            "MOBIL", "MOBIL-04-NWSublaneGeometry",
            nw_car.lane + nw_car.occupied <= nw_lanes and nw_moto.lane + nw_moto.occupied <= nw_lanes,
            f"Nhánh NW ({nw_lanes} sublane) giới hạn làn chính xác: ô tô ({nw_car.lane}..{nw_car.lane+nw_car.occupied}), xe máy ({nw_moto.lane}..{nw_moto.lane+nw_moto.occupied})"
        )

        # MOBIL-05: Same-lane vehicles may queue safely without lane changes.
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, ne_lanes - 2, 40)
        sim._create_vehicle("NE", "through", False, ne_lanes - 2, 40)
        sim.step(1)
        front, rear = sorted(sim.vehicles, key=lambda v: v.position, reverse=True)
        gap = front.position - rear.position - (front.length + rear.length) / 2
        self.assert_test(
            "MOBIL", "MOBIL-05-OverlapResolution",
            gap >= max(1.0, front.gap, rear.gap) - 1e-8,
            f"Xe cùng làn giữ khoảng hở thay vì bắt buộc lách ngang (gap={gap:.2f}m)"
        )

        # MOBIL-06: A lane change in progress follows its smooth lateral trajectory.
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 10, 40)
        v = sim.vehicles[0]
        v.speed = 8.0
        v.lane = 9
        v.lane_change_from = 10
        v.lane_change_progress = 0.0
        v.start_lateral_shift = v.current_lateral_shift = 8.0
        v.target_lateral_shift = 7.2
        v.lateral_duration = 1.0
        sim.step(1)
        shift_1 = v.current_lateral_shift
        prog_1 = v.lane_change_progress
        smooth_active = (0.0 < prog_1 < 1.0 and shift_1 != v.start_lateral_shift and shift_1 != v.target_lateral_shift)
        sim.step(6)
        smooth_complete = (v.lane_change_progress == 1.0 and abs(v.current_lateral_shift - v.target_lateral_shift) < 1e-5)
        self.assert_test(
            "MOBIL", "MOBIL-06-BezierLateralShift",
            smooth_active and smooth_complete,
            f"Lách ngang nội suy mượt mà qua đường cong Bézier (prog={prog_1:.2f}, shift={shift_1:.3f}m -> {v.current_lateral_shift:.3f}m, không teleport)"
        )
        print()

    # -------------------------------------------------------------------------
    # 3. TRAFFIC SIGNAL DYNAMICS
    # -------------------------------------------------------------------------
    def test_traffic_signal_dynamics(self):
        print("--- [3] KIỂM THỬ CHU KỲ ĐÈN & TÍN HIỆU ĐIỀU KHIỂN ---")

        # SIG-01: Fixed time cycle schedule
        sim = LeVanLuongSimulator(flow_per_approach=0)
        self.assert_test(
            "Signal", "SIG-01-FixedTimeInitial",
            sim.phase == "LVL" and sim.step(250)["phase"] == "LVL",
            "Pha LVL xanh chuẩn 250 tick (50s)"
        )
        self.assert_test(
            "Signal", "SIG-01-FixedTimeAllRed1",
            sim.step(1)["phase"] == "none" and sim.step(14)["phase"] == "none",
            "Đỏ toàn phần (all-red) 15 tick (3s) sau pha LVL"
        )
        self.assert_test(
            "Signal", "SIG-01-FixedTimeHDT",
            sim.step(1)["phase"] == "HDT" and sim.step(199)["phase"] == "HDT",
            "Pha HDT xanh chuẩn 200 tick (40s)"
        )
        self.assert_test(
            "Signal", "SIG-01-FixedTimeAllRed2",
            sim.step(1)["phase"] == "none" and sim.step(14)["phase"] == "none",
            "Đỏ toàn phần 15 tick sau pha HDT"
        )
        self.assert_test(
            "Signal", "SIG-01-FixedTimeCycleEnd",
            sim.step(1)["phase"] == "LVL",
            "Lặp lại pha LVL chính xác ở chu kỳ kế tiếp"
        )

        # SIG-02: Train mode decision intervals & all-red switching
        sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True)
        sim.step(50)
        self.assert_test(
            "Signal", "SIG-02-DecisionInterval50",
            sim.dqn_waiting_action and sim.cycle == 50,
            "Tick quyết định RL kích hoạt đúng mỗi 50 cycle (10.0s)"
        )
        sim.set_action(2)  # Request HDT (action 2)
        sim.step(14)
        in_all_red = (sim.phase == "none")
        sim.step(1)
        in_hdt = (sim.phase == "HDT")
        self.assert_test(
            "Signal", "SIG-02-TrainAllRedClearance",
            in_all_red and in_hdt,
            "Đổi pha trong train mode có 15 tick dọn nút (all-red) rồi mới mở pha mới"
        )

        # SIG-03: Red light stopping compliance
        sim = LeVanLuongSimulator(flow_per_approach=3000, seed=77)
        for _ in range(240):  # LVL green, SE & NW are RED
            sim.step()
        se_nw_breaches = [
            v for v in sim.vehicles
            if v.arm in ("SE", "NW") and v.position > v.approach_end
        ]
        self.assert_test(
            "Signal", "SIG-03-RedLightCompliance",
            len(se_nw_breaches) == 0,
            f"100% xe tuân thủ đèn đỏ dừng trước vạch (số xe vi phạm={len(se_nw_breaches)})"
        )

        # SIG-04: Green light releases queue
        sim = LeVanLuongSimulator(flow_per_approach=3000, seed=77)
        sim.step(250)  # SE/NW queued up during red
        queue_before = sum(v.speed * 3.6 < 1.0 and v.arm in ("SE", "NW") for v in sim.vehicles)
        sim.step(70)   # 15 ticks red clearance, then enough green to clear the stop line
        passed_into_conn = sum(v.arm in ("SE", "NW") and v.position > v.approach_end for v in sim.vehicles)
        self.assert_test(
            "Signal", "SIG-04-GreenReleasesQueue",
            queue_before > 0 and passed_into_conn > 0,
            f"Đèn xanh kích hoạt giải tỏa hàng chờ (trước={queue_before} xe dừng, vào nút={passed_into_conn} xe)"
        )

        # SIG-05: Green ticks tracking
        sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True)
        sim.step(50)  # 50 ticks of LVL green
        green_at_50 = sim.green_ticks
        sim.set_action(2)  # switch to HDT -> triggers 15 ticks all-red
        sim.step(10)
        green_at_red = sim.green_ticks
        self.assert_test(
            "Signal", "SIG-05-GreenTicksAccumulation",
            green_at_50 == 50 and green_at_red == 0,
            f"Bộ đếm green_ticks đếm đủ thời gian xanh và reset về 0 khi đỏ (green={green_at_50}, red={green_at_red})"
        )
        print()

    # -------------------------------------------------------------------------
    # 4. SPATIAL STATE & OBSERVATIONS
    # -------------------------------------------------------------------------
    def test_spatial_state_and_observations(self):
        print("--- [4] KIỂM THỬ KHÔNG GIAN TRẠNG THÁI & OBSERVATION ---")

        # OBS-01: Spatial grid dimensions and DTSE dimensions
        obs_spatial = _observation(100, "LVL", True, 1, 10, 5, 20.0, 50.0, 15.0, [], "spatial")
        obs_dtse = _observation(100, "LVL", True, 1, 10, 5, 20.0, 50.0, 15.0, [], "dtse")
        self.assert_test(
            "Observation", "OBS-01-TensorShapes",
            obs_spatial["spatial"].shape == (4, 6, 20, 16) and obs_dtse["dtse"].shape == (3, 8, 20),
            f"Kích thước tensor chuẩn xác: Spatial={obs_spatial['spatial'].shape}, DTSE={obs_dtse['dtse'].shape}"
        )

        # OBS-02: Sublane validity mask in Channel 5
        grid = obs_spatial["spatial"]
        roads = road_lengths()
        masks_ok = True
        for idx, arm in enumerate(ARMS):
            n = roads[f"{arm}_in"]["num_lanes"]
            if not ((grid[idx, 5, :, :n] == 1.0).all() and (grid[idx, 5, :, n:] == 0.0).all()):
                masks_ok = False
        self.assert_test(
            "Observation", "OBS-02-Channel5SublaneMask",
            masks_ok,
            f"Kênh C5 phản ánh đúng mặt nạ hình học (NE={roads['NE_in']['num_lanes']}, SE={roads['SE_in']['num_lanes']}, SW={roads['SW_in']['num_lanes']}, NW={roads['NW_in']['num_lanes']} sublane)"
        )

        # OBS-03: Channel 1 speed ratio averaging across multi-vehicles
        v1 = (10.0 / 3.6, 40.0 / 3.6, 4, 1, "NE_in", 6.0)  # ratio = 0.25, bin_id = 1
        v2 = (30.0 / 3.6, 40.0 / 3.6, 4, 1, "NE_in", 6.0)  # ratio = 0.75, bin_id = 1
        obs_multi = _observation(1, "LVL", False, 49, 2, 0, 0, 0, 0, [v1, v2], "spatial")
        avg_speed_ratio = obs_multi["spatial"][0, 1, 1, 4]
        self.assert_test(
            "Observation", "OBS-03-SpeedRatioAveraging",
            abs(avg_speed_ratio - 0.5) < 1e-5,
            f"Kênh C1 tính đúng vận tốc trung bình ô có nhiều xe (tính được={avg_speed_ratio}, kỳ vọng=0.5)"
        )

        # OBS-04: Finite metric values (never NaN)
        self.assert_test(
            "Observation", "OBS-04-MetricFiniteInvariants",
            math.isfinite(obs_spatial["average_waiting_time"]) and
            math.isfinite(obs_spatial["average_travel_time"]) and
            math.isfinite(obs_spatial["average_speed"]),
            f"Các chỉ số thống kê hữu hạn, không bị NaN khi khởi tạo (wait={obs_spatial['average_waiting_time']:.2f}s, travel={obs_spatial['average_travel_time']:.2f}s)"
        )
        print()

    # -------------------------------------------------------------------------
    # 5. EXTREME EDGE CASES & STABILITY
    # -------------------------------------------------------------------------
    def test_extreme_edge_cases(self):
        print("--- [5] KIỂM THỬ CÁC EDGE CASES CỰC HẠN & ĐỘ ỔN ĐỊNH ---")

        # EDGE-01: Zero traffic stability
        sim = LeVanLuongSimulator(flow_per_approach=0, seed=1)
        sim.step(1000)
        snap = sim.snapshot()
        self.assert_test(
            "EdgeCases", "EDGE-01-ZeroTrafficStability",
            snap["active"] == 0 and snap["created"] == 0 and snap["queue_time_integral"] == 0.0,
            "1.000 tick không có xe chạy mượt mà, không gặp ZeroDivisionError hay crash"
        )

        # EDGE-02: Mega traffic jam stress test (10.000 veh/hr/approach)
        sim_jam = LeVanLuongSimulator(flow_per_approach=10000, seed=42)
        all_speeds_finite = True
        all_positions_finite = True
        for _ in range(500):
            sim_jam.step()
            for v in sim_jam.vehicles:
                if not (math.isfinite(v.speed) and v.speed >= 0):
                    all_speeds_finite = False
                if not math.isfinite(v.position):
                    all_positions_finite = False
        self.assert_test(
            "EdgeCases", "EDGE-02-MegaJamLiveness",
            all_speeds_finite and all_positions_finite and sim_jam.passed > 0,
            f"Tải siêu nghẽn 10.000 xe/h: duy trì liveness (passed={sim_jam.passed} xe), tọa độ và tốc độ hữu hạn"
        )

        # EDGE-03: UTF-8 encoding in demand generation and replay
        with tempfile.TemporaryDirectory(prefix="lvl_edge_") as tmp_dir:
            utf8_path = Path(tmp_dir) / "lịch_xe_tiếng_việt.csv"
            digest = generate_demand(utf8_path, seed=1, horizon=100)
            sim_replay = LeVanLuongSimulator(demand_file=utf8_path)
            sim_replay.step(100)
            self.assert_test(
                "EdgeCases", "EDGE-03-UTF8FileHandling",
                utf8_path.exists() and sim_replay.created > 0,
                f"Đọc/ghi file CSV và GeoJSON có dấu tiếng Việt trơn tru (sha256={digest[:8]}...)"
            )

        # EDGE-04: Front row measurement accuracy (<= 1.5m before stop line only)
        sim_front = LeVanLuongSimulator(flow_per_approach=0)
        # NE approach length is incoming['length'] = ~130.87m
        ne_len = sim_front.roads["NE_in"]["length"]
        # v1: 1.0m before stop line -> should be in front row
        sim_front._create_vehicle("NE", "through", False, 0, 40)
        sim_front.vehicles[0].position = ne_len - 1.0
        # v2: 2.0m before stop line -> should NOT be in front row
        sim_front._create_vehicle("NE", "through", False, 1, 40)
        sim_front.vehicles[1].position = ne_len - 2.0
        # v3: already on connector (crossed stop line) -> should NOT be in approach front row
        sim_front._create_vehicle("NE", "through", False, 2, 40)
        sim_front.vehicles[2].position = ne_len + 0.5
        sim_front._measure()
        self.assert_test(
            "EdgeCases", "EDGE-04-FrontRowDefinition",
            sim_front.front_row_max["NE"] == 1,
            f"Đo hàng đầu chính xác: chỉ nhận xe cách vạch <= 1.5m trên chiều vào (front_row_max={sim_front.front_row_max['NE']})"
        )
        print()

    # -------------------------------------------------------------------------
    # 6. LEGACY CONTROLLER & STARVATION GUARDS
    # -------------------------------------------------------------------------
    def test_legacy_controller_guards(self):
        print("--- [6] KIỂM THỬ BỘ ĐIỀU KHIỂN LEGACY & CHỐNG ĐÓI PHA ---")

        # LEGACY-01: Phase wait cycle accumulation (with demand tracking per GAML)
        import asyncio
        from train.logic.python_env import Vehicle as LegacyVehicle, STOP_DISTANCE as LEGACY_STOP
        async def run_legacy():
            env = PythonTrafficEnv(verbose=False, state_mode="dtse")
            # Without vehicles, all phase wait times must stay 0 (GAML logic: phase_demand = 0 resets wait)
            await env.step_cycles(50)
            c_empty = all(w == 0 for w in env._phase_wait)

            # Place vehicle on approach 5 waiting to turn left (phase 1)
            v = LegacyVehicle(origin=0, kind=1, desired=11.11, maneuver="left",
                              lane=8, approach=5, destination=6, length=4.5,
                              width=1.8, spawn_cycle=0, position=LEGACY_STOP - 50.0)
            env.vehicles.append(v)
            await env.step_cycles(50)
            # Phase 0 is active/green -> wait is 0; Phase 1 has vehicle -> wait accumulated to 50
            # Phase 2 and 3 have no vehicles -> wait remains 0
            c_accum = (env._phase_wait[0] == 0 and env._phase_wait[1] == 50 and
                       env._phase_wait[2] == 0 and env._phase_wait[3] == 0)

            # LEGACY-02: Starvation forced service
            await env.reset(parameters=[{"name": "dqn_service_guard", "type": "string", "value": "true"}])
            env._phase_wait[2] = 1250
            state = await env.read_state(force_dtse=True)
            mask = state["action_mask"]
            c2 = mask[2] and not any(mask[i] for i in (0, 1, 3))

            # LEGACY-03: Empty phase reset (no starvation lock on empty phase)
            await env.reset(parameters=[{"name": "dqn_service_guard", "type": "string", "value": "true"}])
            env._phase_wait[3] = 1250
            await env.step_cycles(1)  # demand[3] == 0 resets _phase_wait[3] to 0
            c3 = env._phase_wait[3] == 0

            return c_empty and c_accum, c2, c3

        c1, c2, c3 = asyncio.run(run_legacy())
        self.assert_test(
            "Legacy", "LEGACY-01-PhaseWaitAccumulation",
            c1,
            "Bộ đếm chờ pha tích lũy chính xác khi có xe và reset về 0 khi có đèn xanh"
        )
        self.assert_test(
            "Legacy", "LEGACY-02-StarvationGuardMask",
            c2,
            "Chống đói pha (wait >= 1200) khóa action_mask chỉ cho phép pha bị bỏ đói"
        )
        self.assert_test(
            "Legacy", "LEGACY-03-EmptyPhaseNoStarvation",
            c3,
            "Pha không có xe (demand=0) tự động reset bộ đếm chờ về 0, không ép phục vụ pha trống"
        )
        print()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    def print_summary(self):
        duration = time.time() - self.start_time
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")

        print("=" * 95)
        print(f"📊 BẢNG TỔNG KẾT KẾT QUẢ KIỂM THỬ VI MÔ & EDGE CASES ({duration:.2f}s)")
        print("=" * 95)
        print(f"{'Module / Suite':<16} | {'Mã Test Case':<32} | {'Trạng thái':<10} | {'Chi tiết'}")
        print("-" * 95)
        for r in self.results:
            print(f"{r['suite']:<16} | {r['test_id']:<32} | {r['status']:<10} | {r['message']}")
        print("=" * 95)
        print(f"📈 TỔNG KẾT: Đã hoàn thành {total} Test Cases.")
        print(f"   - ĐẠT (PASS) : {passed} / {total} ({passed/total*100.0:.1f}%)")
        print(f"   - HỎNG (FAIL): {failed} / {total}")

        if failed == 0 and total > 0:
            print("\n🎉 XÁC NHẬN: 100% TẤT CẢ EDGE CASES & HÀNH VI XE ĐỀU ĐẠT CHUẨN XUẤT SẮC!")
            print("   Mô phỏng Python đã vượt qua toàn bộ kiểm thử hành vi vi mô và các trường hợp biên cực hạn.")
        else:
            print("\n⚠️ CẢNH BÁO: CÓ TEST CASE BỊ LỖI, VUI LÒNG KIỂM TRA LẠI!")
        print("=" * 95 + "\n")
        return 0 if failed == 0 and total > 0 else 1


if __name__ == "__main__":
    suite = VehicleBehaviorTestSuite()
    sys.exit(suite.run_all())
