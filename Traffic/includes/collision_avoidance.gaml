/**
* Name: collision_avoidance
* Description: Upgraded realistic micro-simulation model for mixed traffic (cars, motorcycles, buses).
*              Features:
*                - Realistic road geometry with sublane disciplines (Lane 0: Right, Lane 1: Thru, Lane 2: Left).
*                - Detailed pavement visuals: Asphalt, stone curbs, double yellow center lines, white sublane dashes.
*                - Advanced maneuvers:
*                    + Smooth IDM straight cruising with motorcycle staggered formations.
*                    + Right-turn hugging curb with blinking amber indicator.
*                    + Separate protected left-turn signals and safe exit-capacity gating.
*                    + Lane-aware motorcycle following without cross-lane phantom blockers.
*                    + Microscopic Yielding & Anti-ghosting (no phantom vehicles passing through each other).
*                - Rich vehicle rendering: Headlights, windshields, helmets, glowing red brake lights, blinking indicators.
*/

model collision_avoidance

import "doan_scenery.gaml"

global {
	// DoAn network topology and route metadata.
	bool enable_bypass_traffic <- true;
	float right_turn_share <- 0.15;
	float uturn_share <- 0.05;
	int bypass_spawn_interval <- 120;
	map<string, intersection> network_nodes <- [];
	map<string, road> network_roads <- [];
	map<int, intersection> ground_exits <- [];
	map<int, intersection> ground_exit_entries <- [];
	map<int, intersection> ground_diverges <- [];
	map<int, intersection> ground_merges <- [];
	map<int, intersection> ground_uturn_entries <- [];
	map<int, intersection> ground_uturn_exits <- [];
	map<string, list<intersection>> bypass_routes <- [];
	list<string> bypass_route_names <- ["flyover_NS", "flyover_SN", "on_N", "on_S",
		"off_N", "off_S", "tunnel_WE", "tunnel_EW"];
	map<string, int> route_created <- [];
	map<string, int> route_completed <- [];
	int bypass_vehicles_created <- 0;
	int bypass_vehicles_passed <- 0;
	int right_turn_completed_count <- 0;
	int uturn_completed_count <- 0;
	int failed_route_count <- 0;
	int doan_horizon_cycles <- 8000;

	geometry doan_turn(point start, point finish, point incoming, point outgoing, float bend) {
		point c1 <- start + incoming * bend;
		point c2 <- finish - outgoing * bend;
		list<point> pts <- [];
		loop j from: 0 to: 120 {
			float t <- j / 120.0;
			pts <- pts + (start * ((1-t)^3) + c1 * (3*(1-t)^2*t) + c2 * (3*(1-t)*t*t) + finish * (t^3));
		}
		return line(pts);
	}

	action add_doan_node(string key, point pos) {
		point planar <- {pos.x,pos.y};
		create intersection with: (location: planar, shape: planar, node_key: key, surface_z: pos.z) {
			network_nodes[key] <- self;
		}
	}

	action add_doan_road(string key, string from_key, string to_key, string kind,
		int approach, bool incoming, int lanes, float width, list<point> via) {
		list<point> pts <- [network_nodes[from_key].location + {0,0,network_nodes[from_key].surface_z}] + via +
			[network_nodes[to_key].location + {0,0,network_nodes[to_key].surface_z}];
		create road {
			road_key <- key;
			// DrivingSkill 2025.6.4 flattens agent movement. Keep topology planar
			// and interpolate the independent physical/display elevation profile.
			elevation_points <- pts;
			shape <- line(pts collect {each.x,each.y});
			num_lanes <- lanes;
			sublane_width <- width;
			road_kind <- kind;
			approach_id <- approach;
			is_incoming <- incoming;
			is_outer_road <- kind = "ground";
			is_signal_connector <- kind = "through" or kind = "left";
			level_type <- (kind = "flyover" or kind = "ramp" or kind = "tunnel") ? kind : "ground";
			maxspeed <- kind = "flyover" ? 60 #km/#h : (kind = "uturn" ? 10 #km/#h :
				(kind = "left" or kind = "right" or kind = "ramp" ? 20 #km/#h : 50 #km/#h));
			network_roads[key] <- self;
		}
	}

	action init_doan_network {
		// Retain the four entry and four source indices used by existing controllers.
		list<string> arms <- ["E", "N", "W", "S"];
		list<point> vectors <- [{1,0}, {0,-1}, {-1,0}, {0,1}];
		list<point> sides <- [{0,-2}, {-2,0}, {0,2}, {2,0}];
		loop i from: 0 to: 3 {
			do add_doan_node(key: arms[i] + "_IN", pos: intersection_center + vectors[i] * distance_to_intersection_center + sides[i]);
		}
		loop i from: 0 to: 3 {
			do add_doan_node(key: arms[i] + "_SOURCE", pos: intersection_center + vectors[i] * 990.0 + sides[i]);
		}
		loop i from: 0 to: 3 {
			string arm <- arms[i];
			point outward <- vectors[i];
			point side <- sides[i];
			int approach <- i + 4;
			float portal_distance <- (i = 1 or i = 3) ? 520.0 : 160.0;
			do add_doan_node(key: arm + "_OUT", pos: intersection_center + outward * distance_to_intersection_center - side);
			do add_doan_node(key: arm + "_TARGET", pos: intersection_center + outward * 990.0 - side);
			do add_doan_node(key: arm + "_PORTAL_IN", pos: intersection_center + outward * portal_distance + side);
			do add_doan_node(key: arm + "_PORTAL_OUT", pos: intersection_center + outward * portal_distance - side);
			do add_doan_node(key: arm + "_DIV", pos: intersection_center + outward * 80.0 + side);
			do add_doan_node(key: arm + "_MERGE", pos: intersection_center + outward * 80.0 - side);
			do add_doan_node(key: arm + "_U_IN", pos: intersection_center + outward * 70.0 + side);
			do add_doan_node(key: arm + "_U_OUT", pos: intersection_center + outward * 70.0 - side);
			ground_exits[approach] <- network_nodes[arm + "_TARGET"];
			ground_exit_entries[approach] <- network_nodes[arm + "_OUT"];
			ground_diverges[approach] <- network_nodes[arm + "_DIV"];
			ground_merges[approach] <- network_nodes[arm + "_MERGE"];
			ground_uturn_entries[approach] <- network_nodes[arm + "_U_IN"];
			ground_uturn_exits[approach] <- network_nodes[arm + "_U_OUT"];
			list<string> ins <- ["_SOURCE", "_PORTAL_IN", "_DIV", "_U_IN", "_IN"];
			list<string> outs <- ["_OUT", "_U_OUT", "_MERGE", "_PORTAL_OUT", "_TARGET"];
			loop j from: 0 to: 3 {
				do add_doan_road(key: arm + "_in_" + j, from_key: arm + ins[j], to_key: arm + ins[j + 1],
					kind: "ground", approach: approach, incoming: true, lanes: nb_lanes, width: lane_width, via: []);
				do add_doan_road(key: arm + "_out_" + j, from_key: arm + outs[j], to_key: arm + outs[j + 1],
					kind: "ground", approach: approach, incoming: false, lanes: nb_lanes, width: lane_width, via: []);
			}
			// The U-turn is upstream of the stop bar, on the innermost lane.
			list<point> upts <- [intersection_center + outward * 63.0 + side,
				intersection_center + outward * 61.0,
				intersection_center + outward * 63.0 - side];
			do add_doan_road(key: arm + "_uturn", from_key: arm + "_U_IN", to_key: arm + "_U_OUT",
				kind: "uturn", approach: approach, incoming: false, lanes: nb_lanes, width: lane_width, via: upts);
			geometry ucurve <- doan_turn(network_nodes[arm + "_U_IN"].location,
				network_nodes[arm + "_U_OUT"].location, outward * -1.0, outward, 13.0);
			ask network_roads[arm + "_uturn"] { shape <- ucurve; }
		}
		list<int> straight_ids <- [2,3,0,1];
		list<int> left_ids <- [3,0,1,2];
		list<int> right_ids <- [1,2,3,0];
		loop i from: 0 to: 3 {
			string arm <- arms[i];
			do add_doan_road(key: arm + "_through", from_key: arm + "_IN", to_key: arms[straight_ids[i]] + "_OUT",
				kind: "through", approach: i + 4, incoming: false, lanes: nb_lanes, width: lane_width, via: []);
			geometry left_curve <- doan_turn(network_nodes[arm + "_IN"].location,
				network_nodes[arms[left_ids[i]] + "_OUT"].location, vectors[i] * -1.0, vectors[left_ids[i]], 25.0);
			do add_doan_road(key: arm + "_left", from_key: arm + "_IN", to_key: arms[left_ids[i]] + "_OUT",
				kind: "left", approach: i + 4, incoming: false, lanes: nb_lanes, width: lane_width, via: []);
			ask network_roads[arm + "_left"] { shape <- left_curve; }
			geometry right_curve <- doan_turn(network_nodes[arm + "_DIV"].location,
				network_nodes[arms[right_ids[i]] + "_MERGE"].location, vectors[i] * -1.0, vectors[right_ids[i]], 45.0);
			do add_doan_road(key: arm + "_right", from_key: arm + "_DIV", to_key: arms[right_ids[i]] + "_MERGE",
				kind: "right", approach: i + 4, incoming: false, lanes: nb_lanes, width: lane_width, via: []);
			ask network_roads[arm + "_right"] { shape <- right_curve; }
		}

		// Separate directed flyover nodes. Grade separation never creates a junction.
		loop direction over: ["NS", "SN"] {
			float x <- direction = "NS" ? 998.5 : 1001.5;
			list<float> ys <- direction = "NS" ? [10.0,640.0,1360.0,1990.0] : [1990.0,1360.0,640.0,10.0];
			loop j from: 0 to: 3 { do add_doan_node(key: "F_" + direction + j, pos: {x,ys[j],50.0}); }
			loop j from: 0 to: 2 {
				do add_doan_road(key: "F_" + direction + j, from_key: "F_" + direction + j, to_key: "F_" + direction + (j+1),
					kind: "flyover", approach: -1, incoming: false, lanes: nb_lanes, width: 1.5, via: []);
			}
			bypass_routes["flyover_" + direction] <- [network_nodes["F_" + direction + "0"], network_nodes["F_" + direction + "3"]];
		}
		do add_doan_road(key: "on_N", from_key: "N_PORTAL_IN", to_key: "F_NS1", kind: "ramp", approach: -1, incoming: false, lanes: nb_lanes, width: 1.5, via: []);
		do add_doan_road(key: "on_S", from_key: "S_PORTAL_IN", to_key: "F_SN1", kind: "ramp", approach: -1, incoming: false, lanes: nb_lanes, width: 1.5, via: []);
		do add_doan_road(key: "off_N", from_key: "F_SN2", to_key: "N_PORTAL_OUT", kind: "ramp", approach: -1, incoming: false, lanes: nb_lanes, width: 1.5, via: []);
		do add_doan_road(key: "off_S", from_key: "F_NS2", to_key: "S_PORTAL_OUT", kind: "ramp", approach: -1, incoming: false, lanes: nb_lanes, width: 1.5, via: []);
		// Keep lane slot IDs across merges: DrivingSkill's look-ahead uses those
		// IDs BEFORE choose_lane runs. Vehicles may only use their route's slots;
		// these 16 logical slots do not mean 16 physical lanes on the flyover.
		ask road where (each.road_kind = "flyover") { shift_start <- -4.5; shift_end <- -4.5; }
		ask road where (each.road_kind = "ramp") {
			// Map the ramp's carriageway continuously onto its ground/flyover lanes.
			shift_start <- road_key = "on_N" or road_key = "on_S" ? 1.5 : -4.5;
			shift_end <- road_key = "on_N" or road_key = "on_S" ? -4.5 : 1.5;
		}
		bypass_routes["on_N"] <- [network_nodes["N_SOURCE"], network_nodes["N_PORTAL_IN"], network_nodes["F_NS1"], network_nodes["F_NS3"]];
		bypass_routes["on_S"] <- [network_nodes["S_SOURCE"], network_nodes["S_PORTAL_IN"], network_nodes["F_SN1"], network_nodes["F_SN3"]];
		bypass_routes["off_N"] <- [network_nodes["F_SN0"], network_nodes["F_SN2"], network_nodes["N_PORTAL_OUT"], network_nodes["N_TARGET"]];
		bypass_routes["off_S"] <- [network_nodes["F_NS0"], network_nodes["F_NS2"], network_nodes["S_PORTAL_OUT"], network_nodes["S_TARGET"]];

		// Dedicated tunnel approaches sit outside the surface carriageway.
		// Separate boundary nodes keep them clear of ground traffic at both ends.
		do add_doan_node(key: "W_TUNNEL_SOURCE", pos: {10,1045,0});
		do add_doan_node(key: "E_TUNNEL_TARGET", pos: {1990,1045,0});
		do add_doan_node(key: "E_TUNNEL_SOURCE", pos: {1990,955,0});
		do add_doan_node(key: "W_TUNNEL_TARGET", pos: {10,955,0});
		do add_doan_node(key: "W_TUNNEL_IN", pos: {760,1045,0});
		do add_doan_node(key: "E_TUNNEL_OUT", pos: {1240,1045,0});
		do add_doan_node(key: "E_TUNNEL_IN", pos: {1240,955,0});
		do add_doan_node(key: "W_TUNNEL_OUT", pos: {760,955,0});
		do add_doan_road(key: "tunnel_WE_feed", from_key: "W_TUNNEL_SOURCE", to_key: "W_TUNNEL_IN",
			kind: "tunnel_feeder", approach: -1, incoming: false, lanes: nb_lanes, width: 1.75, via: []);
		do add_doan_road(key: "tunnel_WE", from_key: "W_TUNNEL_IN", to_key: "E_TUNNEL_OUT",
			kind: "tunnel", approach: -1, incoming: false, lanes: nb_lanes, width: 1.75,
			via: [{920.0,1045.0,-60.0}, {1080.0,1045.0,-60.0}]);
		do add_doan_road(key: "tunnel_WE_exit", from_key: "E_TUNNEL_OUT", to_key: "E_TUNNEL_TARGET",
			kind: "tunnel_feeder", approach: -1, incoming: false, lanes: nb_lanes, width: 1.75, via: []);
		do add_doan_road(key: "tunnel_EW_feed", from_key: "E_TUNNEL_SOURCE", to_key: "E_TUNNEL_IN",
			kind: "tunnel_feeder", approach: -1, incoming: false, lanes: nb_lanes, width: 1.75, via: []);
		do add_doan_road(key: "tunnel_EW", from_key: "E_TUNNEL_IN", to_key: "W_TUNNEL_OUT",
			kind: "tunnel", approach: -1, incoming: false, lanes: nb_lanes, width: 1.75,
			via: [{1080.0,955.0,-60.0}, {920.0,955.0,-60.0}]);
		do add_doan_road(key: "tunnel_EW_exit", from_key: "W_TUNNEL_OUT", to_key: "W_TUNNEL_TARGET",
			kind: "tunnel_feeder", approach: -1, incoming: false, lanes: nb_lanes, width: 1.75, via: []);
		bypass_routes["tunnel_WE"] <- [network_nodes["W_TUNNEL_SOURCE"], network_nodes["W_TUNNEL_IN"],
			network_nodes["E_TUNNEL_OUT"], network_nodes["E_TUNNEL_TARGET"]];
		bypass_routes["tunnel_EW"] <- [network_nodes["E_TUNNEL_SOURCE"], network_nodes["E_TUNNEL_IN"],
			network_nodes["W_TUNNEL_OUT"], network_nodes["W_TUNNEL_TARGET"]];
	}

	// Batch experiments replace the built-in seed after init. Apply an ordinary
	// parameter before the first spawn so audit/benchmark seeds are reproducible.
	float traffic_seed <- -1.0;
	reflex apply_traffic_seed when: cycle = 0 and traffic_seed >= 0 { seed <- traffic_seed; }

	// Optional immutable demand replay for paired controller benchmarks.
	// CSV columns: requested cycle, origin (0..3 ground, 4..11 bypass),
	// vehicle type (0 motorcycle, 1 car, 2 bus), speed km/h, maneuver, lane.
	string demand_file_path <- "";
	bool replay_demand <- false;
	matrix<float> replay_rows;
	int replay_cursor <- 0;
	map<int, list<int>> replay_queues <- [];
	int replay_type <- 1;
	float replay_speed <- 35.0;
	int replay_maneuver <- 0;
	int replay_lane <- 4;
	int replay_request_id <- -1; // stable CSV row for paired microscopic diagnostics
	int ground_requests <- 0;
	int bypass_requests <- 0;
	int ground_pending <- 0;
	int bypass_pending <- 0;
	list<int> pending_by_origin <- [0,0,0,0,0,0,0,0,0,0,0,0]; // paired diagnostics
	float queue_time_integral <- 0.0;
	float system_time_integral <- 0.0; // R1: vehicle-seconds on roads and waiting to enter
	float stopped_time_integral <- 0.0; // R2: stopped vehicle-seconds on all routes
	float outgoing_block_integral <- 0.0; // R3: slow vehicles in first 80 m of ground exits
	float weighted_delay_integral <- 0.0; // R4: PCU-weighted speed shortfall, including pending
	float external_weighted_pending <- 0.0;
	int total_lane_change_events <- 0; // diagnostic: actual lowest_lane changes on the same road
	int near_lane_change_events <- 0; // lane changes inside the 160 m spatial observation window
	float external_wait_integral <- 0.0;
	float external_all_time_integral <- 0.0; // pending ground and bypass vehicle-seconds
	float cumulative_max_wait <- 0.0;

	float step <- 0.1 #s;
	bool debug_vehicle_log <- false;
	bool debug_merge_arbitration <- false;
	bool enable_surrogate_safety <- false;
	// Headless Python training can omit presentation and metrics that its reward
	// does not use. Vehicle movement, signal timing and R1 remain unchanged.
	bool fast_training <- false;
	string training_reward_mode <- "all";
	float lane_width <- 1.75 #m;  // Sublane width (motorcycle = 1 sublane = 1.5m; car = 2 sublanes = 3.0m)
	int nb_lanes <- 16;           // 16 sublanes per road direction

	graph road_network;

	point intersection_center <- {1000,1000};
	float distance_to_intersection_center <- 40.5 #m;
	// Painted stop bars are 45 m from the centre; signal nodes are 40.5 m out.
	float signal_node_to_stop_bar <- 4.5 #m;
	float stop_bar_clearance <- 2.0 #m;
	float road_length <- 949.5 #m;

	list<intersection> intersection_target_list;
	geometry shape <- envelope(2000);

	// Global Traffic Light state variables shared by all controllers.
	// Display text is set by each controller after its phase decision.
	// Adaptive greens show an upper bound; all yellow phases last 3 model seconds.
	string signal_countdown <- "";

	// Each axis has a protected through phase and a separate protected left phase.
	list<string> phase_names <- ["NS-THROUGH-GREEN", "NS-THROUGH-YELLOW",
		"NS-LEFT-GREEN", "NS-LEFT-YELLOW", "EW-THROUGH-GREEN",
		"EW-THROUGH-YELLOW", "EW-LEFT-GREEN", "EW-LEFT-YELLOW"];

	// 6 PERFORMANCE METRICS
	int total_vehicles_created <- 0;
	int total_vehicles_passed <- 0;
	list<base_vehicle_species> active_vehicles <- [];
	float average_waiting_time <- 0.0;
	float max_waiting_time <- 0.0;
	float average_speed <- 0.0;
	int current_queue_length <- 0;
	float total_travel_time_sum <- 0.0;
	float average_travel_time <- 0.0;

	// Spawning rates (Lưu lượng xe)
	int spawn_rate_NS <- 15;
	int spawn_rate_EW <- 45;
	bool stochastic_spawn <- false; // B5: Tùy chọn phân phối ngẫu nhiên tự nhiên
	bool enable_ground_traffic <- true;
	float left_turn_share <- 0.20; // Cars and motorcycles; buses remain straight.
	int left_turn_selected_count <- 0;
	int left_turn_completed_count <- 0;

	// A5: Recent PET reset theo pha đèn (bổ sung min_pet lịch sử)
	float recent_phase_min_pet <- 99.0;  // PET nhỏ nhất trong pha đèn hiện tại

	// =========================================================================
	// CÁC BIẾN CHỈ SỐ AN TOÀN THAY THẾ (SSM) TOÀN CỤC
	// =========================================================================
	float total_tit_sum <- 0.0;             // Tổng tích phân rủi ro TIT (s) tích lũy toàn bộ simulation
	float cycle_tit <- 0.0;                 // Tích phân rủi ro TIT (s) trong chu kỳ/pha đèn hiện tại
	int severe_conflicts_count <- 0;        // Số lần TTC < 1.0s (xung đột nghiêm trọng suýt va chạm)
	int hard_braking_count <- 0;            // Số lần phanh khẩn cấp (a < -3.4 m/s² theo chuẩn AASHTO)
	int panic_braking_count <- 0;           // Số lần phanh hoảng loạn (a < -5.0 m/s²)
	int pet_conflicts_count <- 0;           // Số lần vi phạm PET < 1.5s tại 4 ô xung đột ngã tư
	int critical_pet_count <- 0;            // Số lần vi phạm PET < 1.0s (cận kề va chạm cắt góc)
	float min_pet_observed <- 99.0;         // Giá trị PET nhỏ nhất từng quan sát được (s)
	float recent_pet <- 99.0;               // Giá trị PET mới nhất vừa xảy ra (s)
	float drac_max_observed <- 0.0;         // Giá trị DRAC lớn nhất quan sát được (m/s²)
	int dilemma_zone_vehicles_count <- 0;   // Số xe hiện đang kẹt trong Vùng phân vân (Dilemma Zone)
	int prev_global_phase <- -1;
	float detector_distance <- 190.5 #m;

	list<road> controller_roads(bool north_south, bool incoming) {
		return road where (each.is_outer_road and each.is_incoming = incoming and
			((each.approach_id = 5 or each.approach_id = 7) = north_south));
	}

	// =========================================================================
	// KHỞI TẠO CÁC Ô XUNG ĐỘT TRUNG TÂM (4 QUADRANT CONFLICT ZONES)
	// =========================================================================
	action init_conflict_zones {
		if (empty(conflict_zone)) {
			float size <- 30.0;
			loop dx over: [-1.0,1.0] {
				loop dy over: [-1.0,1.0] {
					create conflict_zone with: (location: intersection_center + {dx * size / 2, dy * size / 2}, shape: rectangle(size,size));
				}
			}
		}
	}

	// =========================================================================
	// REFLEX TOÀN CỤC: THEO DÕI VÙNG PHÂN VÂN & RESET CHU KỲ TIT
	// =========================================================================
	reflex ssm_global_monitor {
		// Keep completed agents inert until the experiment is reset. GAMA can
		// still have a driving reflex in flight when the next world step starts;
		// disposing of that agent here can invalidate the driving skill's scope.
		active_vehicles <- active_vehicles where (each.final_target != nil);
		if (global_phase != prev_global_phase) {
			cycle_tit <- 0.0;
			recent_phase_min_pet <- 99.0;
			prev_global_phase <- global_phase;
		}

		dilemma_zone_vehicles_count <- 0;
	}

		// Shared clearance rule for every controller on this geometry.
		bool signal_junction_clear {
			return empty(active_vehicles where (each.current_road != nil and each.uses_signal and
				(road(each.current_road).is_signal_connector or
				 (road(each.current_road).is_incoming and each.yellow_commit and
				  each.distance_to_current_target <= signal_node_to_stop_bar + each.vehicle_length / 2.0 + stop_bar_clearance) or
				 (!road(each.current_road).is_incoming and distance_to(each.location, intersection_center) <=
				  distance_to_intersection_center + each.vehicle_length / 2.0 + 5.0 #m))));
		}

		// Kiểm tra xe còn trên đoạn đường trong nút giao (dùng bởi kiểm thử tích hợp).
	bool junction_clear {
		return empty(active_vehicles where (
			each.current_road != nil and road(each.current_road).is_signal_connector and
			(each.distance_to_current_target >= 8.0 #m or
			 (abs(each.location.x - intersection_center.x) < 30.0 #m + each.vehicle_length / 2.0 + 1.0 #m and
			  abs(each.location.y - intersection_center.y) < 30.0 #m + each.vehicle_length / 2.0 + 1.0 #m))));
	}

	bool left_exit_clear(base_vehicle_species turning) {
		list<road> outlets <- road where (each.is_outer_road and !each.is_incoming and
			distance_to(first(each.shape.points), ground_exit_entries[turning.destination_approach].location) < 0.1 #m);
		if (empty(outlets)) { return false; }
		road outlet <- first(outlets);
		point exit_entry <- first(outlet.shape.points);
		return empty(active_vehicles where (each != turning and each.current_road = outlet and
			distance_to(each.location, exit_entry) < turning.vehicle_length + 8.0 #m));
	}

	// Admit at most one left turn per approach during its protected green.
	reflex arbitrate_left_turns {
		list<int> busy_approaches <- [];
		loop turning over: active_vehicles {
			if (turning.maneuver = "left" and turning.left_committed and
				turning.current_road != nil) {
				bool green_for_turn <- (turning.approach_ns and global_phase = 2) or
					(!turning.approach_ns and global_phase = 6);
				bool yellow_for_turn <- (turning.approach_ns and global_phase = 3) or
					(!turning.approach_ns and global_phase = 7);
				bool crossed_on_yellow <- turning.yellow_commit and
					turning.distance_to_current_target <= signal_node_to_stop_bar +
					turning.vehicle_length / 2.0 + stop_bar_clearance;
				if (!turning.left_entered and !green_for_turn and !yellow_for_turn and
					!crossed_on_yellow) {
					ask turning { left_committed <- false; }
				} else if (!turning.left_entered or
					(road(turning.current_road).is_signal_connector and
						distance_to(turning.location, first(road(turning.current_road).shape.points)) <
							turning.vehicle_length + 12.0 #m)) {
					busy_approaches <- busy_approaches + turning.approach_id;
				}
			}
		}
		loop approach over: [5, 7, 4, 6] {
			bool axis_green <- ((approach = 5 or approach = 7) and global_phase = 2) or
				((approach = 4 or approach = 6) and global_phase = 6);
			if (axis_green and !(busy_approaches contains approach)) {
				list<base_vehicle_species> candidates <- active_vehicles where (
					each.final_target != nil and each.maneuver = "left" and !each.left_entered and
					each.approach_id = approach and each.current_road != nil and
					road(each.current_road).is_outer_road and
					distance_to(each.location, intersection_center) < (distance_to_intersection_center + 15.0 #m) and
					distance_to(each.location, intersection_center) > distance_to_intersection_center and
					left_exit_clear(each));
				if (!empty(candidates)) {
					base_vehicle_species turning <- candidates with_min_of distance_to(each.location, intersection_center);
					ask turning { left_committed <- true; }
					busy_approaches <- busy_approaches + approach;
				}
			}
		}
	}

	// =========================================================================
	// A1+A4: ACTION SINH XE TẬP TRUNG (Centralized Vehicle Spawning)
	// Gọi từ 4 model: do spawn_vehicle_at(start_inter, "NS" or "EW");
	// =========================================================================
	action spawn_vehicle_at(intersection start_inter) {
		if (empty(active_vehicles where (distance_to(each.location, start_inter.location) < 15.0 #m and abs(each.surface_height()) < 3))) {
			create base_vehicle_species number: 1 {
				total_vehicles_created <- total_vehicles_created + 1;
				demand_row_id <- replay_request_id;

				do configure_vehicle(car_only: false);

				location <- start_inter.location;
				if (start_inter = intersection(5)) { heading <- 90.0; }
				else if (start_inter = intersection(7)) { heading <- 270.0; }
				else if (start_inter = intersection(4)) { heading <- 180.0; }
				else { heading <- 0.0; }
				approach_id <- start_inter = intersection(5) ? 5 :
					(start_inter = intersection(7) ? 7 : (start_inter = intersection(4) ? 4 : 6));
				approach_ns <- approach_id = 5 or approach_id = 7;
				int straight_id <- approach_id = 5 ? 7 : (approach_id = 7 ? 5 : (approach_id = 4 ? 6 : 4));
				int left_id <- approach_id = 5 ? 4 : (approach_id = 7 ? 6 : (approach_id = 4 ? 7 : 5));
				int right_id <- approach_id = 5 ? 6 : (approach_id = 7 ? 4 : (approach_id = 4 ? 5 : 7));
				float choice <- replay_demand or vehicle_type = "bus" ? 2.0 : rnd(1.0);
				maneuver <- choice < left_turn_share ? "left" : "straight";
				if (maneuver != "left" and vehicle_type != "bus") {
					if (choice < left_turn_share + right_turn_share) { maneuver <- "right"; }
					else if (choice < left_turn_share + right_turn_share + uturn_share) { maneuver <- "uturn"; }
				}
				if (replay_demand) { maneuver <- ["straight", "left", "right", "uturn"][replay_maneuver]; }
				destination_approach <- maneuver = "left" ? left_id : (maneuver = "right" ? right_id :
					(maneuver = "uturn" ? approach_id : straight_id));
				target <- ground_exits[destination_approach];
				uses_signal <- maneuver = "straight" or maneuver = "left";
				intersection entry_node <- intersection(approach_id - 4);
				intersection exit_node <- ground_exit_entries[destination_approach];
				list<intersection> waypoints <- [start_inter, entry_node, exit_node, target];
				if (maneuver = "right") {
					waypoints <- [start_inter, ground_diverges[approach_id], ground_merges[destination_approach], target];
				} else if (maneuver = "uturn") {
					waypoints <- [start_inter, ground_uturn_entries[approach_id], ground_uturn_exits[approach_id], target];
				}
				do compute_path graph: road_network nodes: waypoints;
				if (final_target = nil) { failed_route_count <- failed_route_count + 1; }
				if (maneuver = "left") { left_turn_selected_count <- left_turn_selected_count + 1; }
					if (maneuver = "left") {
						lowest_lane <- 12;
						allowed_lanes <- [12];
					} else if (maneuver = "uturn") {
						// The U-turn pocket stays beside the median; left turns use
						// the next lane outward so the two paths never cross.
						lowest_lane <- 14;
						allowed_lanes <- [14];
					} else if (maneuver = "right") {
						lowest_lane <- 0; allowed_lanes <- vehicle_type = "motorcycle" ? [0,1,2,3] : [0,1,2];
					} else {
						lowest_lane <- replay_demand ? replay_lane : (vehicle_type = "motorcycle" ? rnd(4,11) : (vehicle_type = "bus" ? 8 : one_of([4,8])));
						allowed_lanes <- vehicle_type = "motorcycle" ? [4,5,6,7,8,9,10,11] : (vehicle_type = "bus" ? [8] : [4,5,6,7,8,9,10]);
					}
				base_allowed_lanes <- allowed_lanes;
				// Draw the vehicle in its actual starting lane on its first step.
				if (current_road != nil) {
					current_lateral_shift <- (road(current_road).num_lanes - lowest_lane -
						mean(range(num_lanes_occupied - 1)) - 0.5) * road(current_road).sublane_width + road(current_road).render_offset(location);
					target_lateral_shift <- current_lateral_shift;
					lateral_shift_initialized <- true;
				}
				active_vehicles <- active_vehicles + self;
			}
		}
	}

	// =========================================================================
	// A4: REFLEX CẬP NHẬT METRICS TẬP TRUNG (Centralized Metrics Update)
	// =========================================================================
	reflex update_metrics {
		bool r1_only <- fast_training and training_reward_mode = "r1";
		int queue <- 0;
		int stopped_all <- 0;
		int blocked_outgoing <- 0;
		float weighted_delay <- external_weighted_pending;
		float sum_wait <- 0.0;
		float max_wait <- 0.0;
		float sum_speed <- 0.0;
		int valid_speed_count <- 0;
		int count <- 0;

		loop v over: active_vehicles {
			if (!r1_only and v.measured_speed < 1.0 #km/#h) { stopped_all <- stopped_all + 1; }
			if (!r1_only and v.current_road != nil and v.location != nil and v.route_kind = "ground" and
				road(v.current_road).is_outer_road and !road(v.current_road).is_incoming and
				sqrt((v.location.x - intersection_center.x)^2 +
					(v.location.y - intersection_center.y)^2) < distance_to_intersection_center + 80.0 #m and
				v.measured_speed < 5.0 #km/#h) {
				blocked_outgoing <- blocked_outgoing + 1;
			}
			if (!r1_only) {
				float vehicle_weight <- v.vehicle_type = "motorcycle" ? 0.3 :
					(v.vehicle_type = "bus" ? 2.0 : 1.0);
				float desired <- max([0.1 #km/#h, v.original_max_speed]);
				weighted_delay <- weighted_delay + vehicle_weight *
					(1.0 - min([1.0, max([0.0, v.measured_speed / desired])]));
			}
			if (v.route_kind = "ground") {
				count <- count + 1;
				if (v.speed < 1.0 #km/#h) {
					queue <- queue + 1;
				}
				sum_wait <- sum_wait + v.waiting_time;
				if (v.waiting_time > max_wait) {
					max_wait <- v.waiting_time;
				}
				if (v.speed = v.speed and v.speed >= 0.0) {
					sum_speed <- sum_speed + v.speed;
					valid_speed_count <- valid_speed_count + 1;
				}
			}
		}

		if (count > 0) {
			average_waiting_time <- sum_wait / count;
			max_waiting_time <- max_wait;
			average_speed <- valid_speed_count > 0 ?
				(sum_speed / valid_speed_count) * 3.6 : 0.0;
			current_queue_length <- queue;
		} else {
			average_waiting_time <- 0.0;
			max_waiting_time <- 0.0;
			average_speed <- 0.0;
			current_queue_length <- 0;
		}

		if (!fast_training and cycle mod 10 = 0 and cycle > 0) {
			write "METRIC_CSV," + cycle + "," + total_vehicles_passed + "," + (round(average_waiting_time*100.0)/100.0) + "," + current_queue_length + "," + (round(average_speed*100.0)/100.0) + "," + (round(average_travel_time*100.0)/100.0) + "," + (round(max_waiting_time*100.0)/100.0);
			if (enable_surrogate_safety) {
				write "SAFETY_CSV," + cycle + "," + (round(total_tit_sum*100.0)/100.0) + "," + (round(drac_max_observed*100.0)/100.0) + "," + severe_conflicts_count + "," + hard_braking_count + "," + pet_conflicts_count + "," + (round(min_pet_observed*100.0)/100.0) + "," + dilemma_zone_vehicles_count;
			}
		}
		queue_time_integral <- queue_time_integral + current_queue_length * step;
		external_wait_integral <- external_wait_integral + ground_pending * step;
		external_all_time_integral <- external_all_time_integral +
			(ground_pending + bypass_pending) * step;
		system_time_integral <- system_time_integral +
			(length(active_vehicles) + ground_pending + bypass_pending) * step;
		if (!r1_only) {
			stopped_time_integral <- stopped_time_integral + stopped_all * step;
			outgoing_block_integral <- outgoing_block_integral + blocked_outgoing * step;
			weighted_delay_integral <- weighted_delay_integral + weighted_delay * step;
		}
		cumulative_max_wait <- max([cumulative_max_wait, max_waiting_time]);
	}

	// =========================================================================
	// REFLEX SINH XE TẬP TRUNG (thay thế reflex riêng trong từng model)
	// =========================================================================
	reflex load_replay_demand when: cycle = 0 and demand_file_path != "" {
		replay_demand <- true;
		replay_rows <- matrix<float>(csv_file(demand_file_path, ",", float, false));
		external_weighted_pending <- 0.0;
		loop origin from: 0 to: 11 { replay_queues[origin] <- []; }
	}

	reflex admit_replay_demand when: replay_demand {
		loop while: replay_cursor < replay_rows.rows {
			if (replay_rows[0, replay_cursor] > cycle) { break; }
			int origin <- int(replay_rows[1, replay_cursor]);
			replay_queues[origin] <- replay_queues[origin] + replay_cursor;
			int requested_class <- int(replay_rows[2, replay_cursor]);
			external_weighted_pending <- external_weighted_pending +
				(requested_class = 0 ? 0.3 : (requested_class = 2 ? 2.0 : 1.0));
			if (origin < 4) { ground_requests <- ground_requests + 1; }
			else { bypass_requests <- bypass_requests + 1; }
			replay_cursor <- replay_cursor + 1;
		}
		// Rotate admission order for routes sharing a source; retain FIFO per route.
		loop k from: 0 to: 11 {
			int origin <- (k + int(cycle / bypass_spawn_interval)) mod 12;
			list<int> pending <- replay_queues[origin];
			if (!empty(pending)) {
				int row <- first(pending);
				replay_type <- int(replay_rows[2, row]);
				replay_speed <- replay_rows[3, row];
				replay_maneuver <- int(replay_rows[4, row]);
				replay_lane <- int(replay_rows[5, row]);
				replay_request_id <- row;
				int before <- total_vehicles_created + bypass_vehicles_created;
				if (origin < 4) {
					intersection start_inter <- network_nodes[["N_SOURCE", "S_SOURCE", "E_SOURCE", "W_SOURCE"][origin]];
					do spawn_vehicle_at(start_inter);
				} else {
					do spawn_bypass_vehicle(bypass_route_names[origin - 4]);
				}
				if (total_vehicles_created + bypass_vehicles_created > before) {
					replay_queues[origin] <- pending - row;
					external_weighted_pending <- max([0.0, external_weighted_pending -
						(replay_type = 0 ? 0.3 : (replay_type = 2 ? 2.0 : 1.0))]);
				}
			}
		}
		ground_pending <- 0;
		bypass_pending <- 0;
		loop origin from: 0 to: 11 {
			pending_by_origin[origin] <- length(replay_queues[origin]);
			if (origin < 4) { ground_pending <- ground_pending + length(replay_queues[origin]); }
			else { bypass_pending <- bypass_pending + length(replay_queues[origin]); }
		}
	}

	reflex spawn_bypass_traffic when: !replay_demand and enable_bypass_traffic and every(bypass_spawn_interval #cycle) {
		// Rotate demand so ground and flyover origins are not monopolised by one route.
		int offset <- int(cycle / bypass_spawn_interval) mod length(bypass_route_names);
		loop k from: 0 to: length(bypass_route_names) - 1 {
			do spawn_bypass_vehicle(route_name: bypass_route_names[(k + offset) mod length(bypass_route_names)]);
		}
	}

	reflex spawn_traffic_ns when: !replay_demand and enable_ground_traffic and every(spawn_rate_NS #cycle) {
		loop start_inter over: [network_nodes["N_SOURCE"], network_nodes["S_SOURCE"]] {
			do spawn_vehicle_at(start_inter);
		}
	}

	reflex spawn_traffic_ew when: !replay_demand and enable_ground_traffic and every(spawn_rate_EW #cycle) {
		loop start_inter over: [network_nodes["E_SOURCE"], network_nodes["W_SOURCE"]] {
			do spawn_vehicle_at(start_inter);
		}
	}

	action spawn_bypass_vehicle(string route_name) {
		list<intersection> waypoints <- bypass_routes[route_name];
		intersection origin <- first(waypoints);
		if (empty(active_vehicles where (distance_to(each.location, origin.location) < 15 #m and abs(each.surface_height() - origin.surface_z) < 3))) {
			create base_vehicle_species {
				demand_row_id <- replay_request_id;
				do configure_vehicle(car_only: true);
				route_kind <- route_name;
				spawn_surface_z <- origin.surface_z;
				uses_signal <- false;
				target <- last(waypoints);
				location <- origin.location;
				heading <- origin.location direction_to waypoints[1].location;
				lowest_lane <- route_name = "tunnel_WE" or route_name = "tunnel_EW" ? 14 : 9;
				allowed_lanes <- [lowest_lane]; base_allowed_lanes <- allowed_lanes;
				do compute_path graph: road_network nodes: waypoints;
				if (final_target = nil) { failed_route_count <- failed_route_count + 1; }
				bypass_vehicles_created <- bypass_vehicles_created + 1;
				route_created[route_name] <- (route_created contains_key route_name ? route_created[route_name] : 0) + 1;
				active_vehicles <- active_vehicles + self;
			}
		}
	}


	init {
		do init_doan_network;
		road_network <- as_driving_graph(road, intersection);
		ask intersection {
			priority_roads <- roads_in where (road(each).road_kind = "ground" or road(each).road_kind = "flyover");
			managed_merge <- (length(roads_in) > 1 or length(roads_out) > 1) and
				!empty((roads_in + roads_out) where (road(each).road_kind = "ramp" or road(each).road_kind = "tunnel" or
				road(each).road_kind = "right" or road(each).road_kind = "uturn"));
			merge_clearance <- !empty(roads_in where (road(each).road_kind = "ramp" or road(each).road_kind = "tunnel")) ?
				((node_key = "F_SN1" or node_key = "F_NS1") ? 70.0 : 35.0) :
				(!empty(roads_out where (road(each).road_kind = "ramp" or road(each).road_kind = "tunnel")) ? 25.0 :
				(!empty((roads_in + roads_out) where
					(road(each).road_kind = "uturn" or road(each).road_kind = "right")) ? 18.0 : 8.0));
		}
		// Divergences have no crossing traffic to reserve. Holding a slot here
		// made vehicles stop far upstream of the signal, including U-turns.
		loop arm over: ["E", "N", "W", "S"] {
			ask network_nodes[arm + "_U_IN"] { managed_merge <- false; }
			ask network_nodes[arm + "_DIV"] { managed_merge <- false; }
		}
		ask network_nodes["N_PORTAL_IN"] { managed_merge <- false; }
		ask network_nodes["S_PORTAL_IN"] { managed_merge <- false; }
		intersection_target_list <- ground_exits.values;

		// 4. Cau hinh den tin hieu tai 4 nut giao vao (p1..p4)
		ask intersection(0) { is_traffic_signal <- true; direction_type <- 1; do init_signal; } // East (p1)
		ask intersection(1) { is_traffic_signal <- true; direction_type <- 0; do init_signal; } // North (p2)
		ask intersection(2) { is_traffic_signal <- true; direction_type <- 1; do init_signal; } // West (p3)
		ask intersection(3) { is_traffic_signal <- true; direction_type <- 0; do init_signal; } // South (p4)

		// 5. Khoi tao 4 o xung dot trung tam cho PET sensor
		if (!fast_training or enable_surrogate_safety) { do init_conflict_zones; }

		// 6. Khoi tao vach ke duong, vach di bo, cot den tin hieu truc quan
		if (!fast_training) { do init_doan_scenery; }
	}
}

// =========================================================================
// LOÀI TÁC TỬ: INTERSECTION
// =========================================================================
species intersection skills: [intersection_skill] {
	string node_key <- "";
	float surface_z <- 0.0;
	bool managed_merge <- false;
	float merge_clearance <- 8.0;
	list<base_vehicle_species> merge_owners <- [];
	map<string, float> merge_owner_remaining <- [];
	map<string, int> merge_owner_progress_cycle <- [];
	map<string, int> merge_retry_cycle <- [];
	map<string, int> merge_wait_since <- [];
	map<string, int> ramp_stop_ticks <- [];

	// Reserve each occupied lane across the physical convergence/divergence
	// zone. Independent lanes can proceed together; followers wait for clearance.
	reflex manage_merge when: managed_merge {
		bool ramp_portal <- !empty(roads_in where (road(each).road_kind = "ramp"));
		// The route target can lag behind the road transition. Keep a slot
		// while the vehicle approaches this node or its rear still occupies
		// the conflict area, then release it on the outgoing edge.
		merge_owners <- merge_owners where (each.final_target != nil and
			((each.current_road != nil and road(each.current_road).target_node = self) or
			 (each.next_road != nil and road(each.next_road).target_node = self and
			  each.current_road != nil and road(each.current_road).target_node != self) or
			 (each.current_road != nil and road(each.current_road).source_node = self and
			  distance_to(each.location,location) <= each.vehicle_length / 2.0 + 3.0 #m)));
		list<base_vehicle_species> candidates <- [];
		loop r over: roads_in {
			road incoming <- road(r);
			candidates <- candidates + ((incoming.all_agents collect base_vehicle_species(each)) where
				(each.final_target != nil and each.current_target = self and each.distance_to_current_target < 120));
			// A short incoming edge can be shorter than the stopping margin.
			// Reserve the merge for a vehicle on the edge immediately before it,
			// otherwise that vehicle cannot enter and never becomes a candidate.
			if (incoming.shape.perimeter < 120) {
				loop prior_edge over: intersection(incoming.source_node).roads_in {
					candidates <- candidates + ((road(prior_edge).all_agents collect base_vehicle_species(each)) where
						(each.final_target != nil and each.next_road = incoming and
						 each.distance_to_current_target + incoming.shape.perimeter < 120));
				}
			}
		}
		loop candidate over: candidates {
			if (!(merge_wait_since contains_key candidate.name) or merge_owners contains candidate) {
				merge_wait_since[candidate.name] <- cycle;
			}
		}
		// Once the head of an off-ramp has stopped for five seconds, reserve
		// the next opening for it. Existing owners still clear normally; only
		// new conflicting reservations from the through road are held back.
		base_vehicle_species ramp_priority <- nil;
		if (ramp_portal) {
			loop candidate over: candidates {
				bool stopped_ramp <- candidate.current_road != nil and
					road(candidate.current_road).road_kind = "ramp" and
					!(merge_owners contains candidate) and
					candidate.measured_speed < 0.5 #km/#h;
				ramp_stop_ticks[candidate.name] <- stopped_ramp ?
					(ramp_stop_ticks contains_key candidate.name ? ramp_stop_ticks[candidate.name] + 1 : 1) : 0;
				if (ramp_stop_ticks[candidate.name] >= 50) {
					bool head_of_lane <- empty((road(candidate.current_road).all_agents collect base_vehicle_species(each)) where
						(each != candidate and each.final_target != nil and
						 each.distance_to_current_target < candidate.distance_to_current_target and
						 each.lowest_lane < candidate.lowest_lane + candidate.num_lanes_occupied and
						 each.lowest_lane + each.num_lanes_occupied > candidate.lowest_lane));
					if (head_of_lane and (ramp_priority = nil or
						ramp_stop_ticks[candidate.name] > ramp_stop_ticks[ramp_priority.name])) {
						ramp_priority <- candidate;
					}
				}
			}
		}
		// A reservation is for a vehicle approaching the conflict zone, not a
		// permanent lock on its lane. If an owner makes no progress for 10 s well
		// outside that zone, let a waiting vehicle from another edge use it.
		// The old owner is held by yield_at_merge/test_next_road until readmitted.
		list<base_vehicle_species> owners_to_release <- [];
		loop owner over: merge_owners {
			string key <- owner.name;
			float remaining <- owner.current_target = self ? owner.distance_to_current_target :
				(owner.next_road != nil and road(owner.next_road).target_node = self ?
				 owner.distance_to_current_target + road(owner.next_road).shape.perimeter : 0.0 #m);
			if (!(merge_owner_remaining contains_key key) or
				remaining < merge_owner_remaining[key] - 1.0 #m) {
				merge_owner_remaining[key] <- remaining;
				merge_owner_progress_cycle[key] <- cycle;
			}
			bool outside_zone <- distance_to(owner.location, location) >
				merge_clearance + owner.vehicle_length / 2.0 + 2.0 #m;
			agent owner_entry <- owner.current_target = self ? owner.current_road : owner.next_road;
			bool competing_approach <- !empty(candidates where
				(each != owner and (each.current_target = self ? each.current_road : each.next_road) != owner_entry and
				 ((each.lowest_lane < owner.lowest_lane + owner.num_lanes_occupied and
				   each.lowest_lane + each.num_lanes_occupied > owner.lowest_lane) or
				  (ramp_portal and each.current_road != nil and owner.current_road != nil and
				   (road(each.current_road).road_kind = "ramp" or road(owner.current_road).road_kind = "ramp") and
				   abs(each.lowest_lane - owner.lowest_lane) <= 2))));
			// A lane change can put an unreserved vehicle ahead of an earlier
			// owner on the same edge. That owner cannot reach the node first and
			// must not keep the slot while the vehicle ahead waits for it.
			bool overtaken_on_edge <- !empty(candidates where
				(each != owner and !(merge_owners contains each) and
				 each.current_road = owner.current_road and
				 each.distance_to_current_target + 1.0 #m < owner.distance_to_current_target and
				 each.lowest_lane < owner.lowest_lane + owner.num_lanes_occupied and
				 each.lowest_lane + each.num_lanes_occupied > owner.lowest_lane));
			// A reservation made 120 m upstream must not block a whole queue
			// while its owner approaches slowly from another edge. Transfer it
			// only when the owner can stop before the conflict area and the
			// head of the waiting queue could reach the node much sooner.
			bool slower_owner_blocks_queue <- false;
			base_vehicle_species waiting_challenger <- nil;
			float owner_stop_margin <- merge_clearance + owner.vehicle_length / 2.0 +
				owner.min_safety_distance + 7.0 #m +
				owner.measured_speed * owner.measured_speed / (2.0 * owner.max_deceleration);
			if (remaining > owner_stop_margin and outside_zone) {
				loop waiting over: candidates {
					bool conflicts <- (waiting.lowest_lane < owner.lowest_lane + owner.num_lanes_occupied and
						waiting.lowest_lane + waiting.num_lanes_occupied > owner.lowest_lane) or
						(ramp_portal and waiting.current_road != nil and owner.current_road != nil and
						 (road(waiting.current_road).road_kind = "ramp" or road(owner.current_road).road_kind = "ramp") and
						 abs(waiting.lowest_lane - owner.lowest_lane) <= 2);
					if (waiting != owner and !(merge_owners contains waiting) and conflicts and
						cycle - merge_wait_since[waiting.name] >= 50 and waiting.current_road != nil) {
						bool head_of_lane <- empty((road(waiting.current_road).all_agents collect base_vehicle_species(each)) where
							(each != waiting and each.final_target != nil and
							 each.distance_to_current_target < waiting.distance_to_current_target and
							 each.lowest_lane < waiting.lowest_lane + waiting.num_lanes_occupied and
							 each.lowest_lane + each.num_lanes_occupied > waiting.lowest_lane));
						float waiting_distance <- waiting.current_target = self ? waiting.distance_to_current_target :
							waiting.distance_to_current_target + road(waiting.next_road).shape.perimeter;
						float owner_eta <- remaining / max([owner.measured_speed,0.5 #m/#s]);
						float waiting_eta <- sqrt(2.0 * waiting_distance /
							max([waiting.max_acceleration,0.5])) + 2.0 #s;
						if (head_of_lane and owner_eta > waiting_eta + 5.0 #s) {
							slower_owner_blocks_queue <- true;
							waiting_challenger <- waiting;
							break;
						}
					}
				}
			}
			if (overtaken_on_edge) {
				owners_to_release <- owners_to_release + owner;
			} else if (slower_owner_blocks_queue or
				(outside_zone and competing_approach and owner.measured_speed < 0.5 #km/#h and
				 cycle - merge_owner_progress_cycle[key] >= 100)) {
				owners_to_release <- owners_to_release + owner;
				merge_retry_cycle[key] <- cycle + 100;
				if (slower_owner_blocks_queue and debug_merge_arbitration) {
					write "DOAN_MERGE_PREEMPT," + cycle + "," + node_key + "," +
						owner.name + "," + owner.distance_to_current_target + "," +
						owner.measured_speed + "," + waiting_challenger.name + "," +
						waiting_challenger.distance_to_current_target;
				}
			}
		}
		loop owner over: owners_to_release {
			merge_owners <- merge_owners where (each != owner);
		}
		loop candidate over: candidates sort_by
			(each.current_target = self ? each.distance_to_current_target :
			 each.distance_to_current_target + road(each.next_road).shape.perimeter) {
			// At an off-ramp, slot 9 is rendered beside ground slot 11.
			// Reserve the neighbouring physical lanes as one merge path.
			bool yields_to_ramp <- ramp_priority != nil and candidate != ramp_priority and
				((candidate.lowest_lane < ramp_priority.lowest_lane + ramp_priority.num_lanes_occupied and
				  candidate.lowest_lane + candidate.num_lanes_occupied > ramp_priority.lowest_lane) or
				 (candidate.current_road != nil and abs(candidate.lowest_lane - ramp_priority.lowest_lane) <= 2));
			int retry_after <- merge_retry_cycle contains_key candidate.name ?
				merge_retry_cycle[candidate.name] : 0;
			if (!(merge_owners contains candidate) and !yields_to_ramp and
				cycle >= retry_after and
				empty(merge_owners where
				((each.lowest_lane < candidate.lowest_lane + candidate.num_lanes_occupied and
				  each.lowest_lane + each.num_lanes_occupied > candidate.lowest_lane) or
				 (ramp_portal and each.current_road != nil and candidate.current_road != nil and
				  (road(each.current_road).road_kind = "ramp" or road(candidate.current_road).road_kind = "ramp") and
				  abs(each.lowest_lane - candidate.lowest_lane) <= 2)))) {
				merge_owners <- merge_owners + candidate;
				merge_owner_remaining[candidate.name] <- candidate.current_target = self ?
					candidate.distance_to_current_target :
					candidate.distance_to_current_target + road(candidate.next_road).shape.perimeter;
				merge_owner_progress_cycle[candidate.name] <- cycle;
			}
		}
	}

	bool is_traffic_signal <- false;
	int direction_type <- 0; // 0: NS, 1: EW
	rgb light_color <- #red;
	list<road> incoming_outer_roads;

	action init_signal {
		stop <- [[], []];
		incoming_outer_roads <- road where (each.is_outer_road and distance_to(last(each.shape.points), location) < 0.1#m);
	}

	reflex update_lights when: is_traffic_signal {
		bool through_green <- (direction_type = 0 and global_phase = 0) or
			(direction_type = 1 and global_phase = 4);
		bool through_yellow <- phase_timer < signal_yellow_seconds and
			((direction_type = 0 and global_phase = 1) or
			 (direction_type = 1 and global_phase = 5));
		light_color <- through_green ? #lime : (through_yellow ? #yellow : #red);
		stop[0] <- through_green ? [] : incoming_outer_roads;
		// The left signal has an additional per-vehicle exit-capacity gate.
		stop[1] <- incoming_outer_roads;
	}

	aspect base {
		// Trụ đèn tại nút giao được vẽ qua traffic_signal_post.
	}
}

// =========================================================================
// LOÀI TÁC TỬ: ROAD
// Ve mat duong Asphalt sang trong, via he lat da, vach vang doi o tim duong
// =========================================================================
species road skills: [road_skill] {
	string road_key <- "";
	string road_kind <- "ground";
	string level_type <- "ground";
	int approach_id <- -1;
	bool is_incoming <- false;
	bool is_signal_connector <- false;
	float sublane_width <- lane_width;
	float shift_start <- 0.0;
	float shift_end <- 0.0;
	list<point> elevation_points <- [];
	float elevation_at(point pos) {
		float best <- 1e20;
		float height <- 0.0;
		if (length(elevation_points) > 1) {
			loop j from: 0 to: length(elevation_points) - 2 {
				point a <- elevation_points[j]; point b <- elevation_points[j+1];
				float dx <- b.x-a.x; float dy <- b.y-a.y;
				float squared <- dx*dx + dy*dy;
				float t <- squared > 0 ? max([0.0,min([1.0,((pos.x-a.x)*dx+(pos.y-a.y)*dy)/squared])]) : 0.0;
				float d <- (pos.x-a.x-t*dx)^2 + (pos.y-a.y-t*dy)^2;
				if (d < best) { best <- d; height <- a.z + t * (b.z-a.z); }
			}
		}
		return height;
	}
	float render_offset(point pos) {
		float span <- distance_to(first(shape.points), last(shape.points));
		float progress <- span > 0.0 ? min([1.0, distance_to(first(shape.points), pos) / span]) : 0.0;
		return shift_start + (shift_end - shift_start) * progress;
	}

	bool is_outer_road <- true;
	rgb asphalt_color <- rgb(34, 39, 46);      // Mat duong nhua Asphalt sam hien dai
	rgb curb_color <- rgb(80, 85, 95);         // Vien via he bo vieng

	aspect base {
		// Scenery is drawn via doan_scenery.gaml
	}
}

// =========================================================================
// LOÀI TÁC TỬ: BASE VEHICLE SPECIES
// Mo hinh vi mo da tac tu dong xe hon hop (O to, Xe may, Xe bus)
// =========================================================================
species base_vehicle_species skills: [driving] parallel: false {
	int demand_row_id <- -1;
	string vehicle_type <- "car"; // "motorcycle", "car", "bus"
	rgb color;
	float vehicle_width <- 1.9 #m;
	intersection target;
	string route_kind <- "ground";
	bool uses_signal <- true;
	int destination_approach <- -1;
	string maneuver <- "straight"; // "straight", "left", "right"
	float waiting_time <- 0.0;
	float spawn_time <- 0.0;
	float original_max_speed;
	list<int> base_allowed_lanes <- [];
	bool yellow_commit <- false;
	int approach_id <- -1;
	bool approach_ns <- true;
	bool left_committed <- false;
	bool left_entered <- false;

	// BIẾN ĐỘNG HỌC & ĐO LƯỜNG AN TOÀN VI MÔ
	float prev_speed <- 0.0;
	float measured_speed <- 0.0;
	float current_acc <- 0.0;
	bool is_in_dz <- false;
	float min_ttc_experienced <- 99.0;
	bool was_hard_braking <- false;
	bool was_panic_braking <- false;
	list<base_vehicle_species> previous_severe_partners <- [];

	// Rear-end TTC is directional: the follower observes the leader. Filtering
	// by agent names first discards real conflicts. Crossing conflicts use PET.
	float following_ttc(base_vehicle_species leader) {
		float direction_diff <- abs(heading - leader.heading);
		if (direction_diff > 180.0) { direction_diff <- 360.0 - direction_diff; }
		point delta <- leader.location - location;
		float along <- delta.x * cos(heading) + delta.y * sin(heading);
		float lateral <- abs(delta.x * sin(heading) - delta.y * cos(heading));
		float gap <- along - (vehicle_length + leader.vehicle_length) / 2.0;
		float closing <- speed - leader.speed * cos(direction_diff);
		if (direction_diff > 30.0 or lateral >= (vehicle_width + leader.vehicle_width) / 2.0 or
			gap <= 0.0 or closing <= 0.5) { return 99.0; }
		return gap / closing;
	}

	// HIỂN THỊ CHUYỂN LÀN MƯỢT MÀ (SMOOTHERSTEP)
	bool lateral_shift_initialized <- false;
	float current_lateral_shift <- 0.0;
	float lateral_velocity <- 0.0 #m/#s; // signed visual lane-change velocity; positive to vehicle's left
	float recent_lateral_peak <- 0.0 #m/#s; // largest signed speed since the previous decision snapshot
	road observed_lane_road <- nil;
	int observed_lane_index <- -1;
	float target_lateral_shift <- 0.0;
	float start_lateral_shift <- 0.0;
	float lane_change_progress <- 1.0;
	float current_yaw_deviation <- 0.0;
	float spawn_surface_z <- 0.0;
	float surface_height {
		return current_road = nil ? spawn_surface_z : road(current_road).elevation_at(location);
	}
	list<base_vehicle_species> nearby_vehicles(float radius) {
		return (base_vehicle_species at_distance radius) where (each.current_road != nil and
			each.final_target != nil and abs(each.surface_height() - surface_height()) < 3);
	}

	action configure_vehicle(bool car_only) {
				// Phan tach 3 loai xe theo thuc te SinD/VN: 55% Xe may, 38% O to, 7% Xe bus
				float r_type <- replay_demand ? [0.2, 0.8, 0.99][replay_type] : (car_only ? 0.8 : rnd(1.0));
				if (r_type < 0.55) {
					vehicle_type <- "motorcycle";
					vehicle_length <- 2.0 #m;
					vehicle_width <- 0.8 #m;
					num_lanes_occupied <- 1;
					max_acceleration <- 2.8;
					max_deceleration <- 4.5;
					min_safety_distance <- 0.8;
					max_speed <- (replay_demand ? replay_speed : (25 + rnd(12))) #km / #h;
					color <- replay_demand ? #dodgerblue : rnd_choice([#dodgerblue::0.35, #crimson::0.30, #forestgreen::0.20, #orange::0.15]);
				} else if (r_type < 0.93) {
					vehicle_type <- "car";
					vehicle_length <- 4.6 #m;
					vehicle_width <- 1.9 #m;
					num_lanes_occupied <- 2;
					max_acceleration <- 1.8;
					max_deceleration <- 3.8;
					min_safety_distance <- 2.0;
					max_speed <- (replay_demand ? replay_speed : (30 + rnd(15))) #km / #h;
					color <- replay_demand ? #white : rnd_choice([#white::0.35, #silver::0.25, #black::0.25, #navy::0.15]);
					} else {
						vehicle_type <- "bus";
					vehicle_length <- 10.0 #m;
					vehicle_width <- 2.4 #m;
					num_lanes_occupied <- 2;
					max_acceleration <- 1.2;
					max_deceleration <- 2.8;
					min_safety_distance <- 3.0;
						max_speed <- (replay_demand ? replay_speed : (22 + rnd(10))) #km / #h;
						color <- #goldenrod;
					}
					// The vehicle init runs before these per-type assignments.
					// Preserve the actual sampled speed, not the driving skill default.
					original_max_speed <- max_speed;

	}

	// Driving calls this hook before entering a road with a different lane count.
	int choose_lane(agent new_road) {
		road next <- road(new_road);
		int chosen <- min([lowest_lane, next.num_lanes - num_lanes_occupied]);
		if (route_kind != "ground") {
			chosen <- route_kind = "tunnel_WE" or route_kind = "tunnel_EW" ? 14 : 9;
			allowed_lanes <- [chosen]; base_allowed_lanes <- [chosen];
		}
		return chosen;
	}

	init {
		spawn_time <- time;
		original_max_speed <- max_speed;
		prev_speed <- measured_speed;
		right_side_driving <- true;

			// Không tự đi vào khi đoạn sau nút giao chưa đủ chỗ.
		proba_block_node <- 0.0;
		proba_respect_priorities <- 1.0;
		proba_respect_stops <- [1.0, 1.0];
		proba_use_linked_road <- 0.0;

		// KHỬ LÁCH LÀN TỰ DO & THIÊN VỊ BÁM PHẢI KHI ĐƯỜNG THÔNG THOÁNG (MOBIL MODEL)
		acc_bias <- 0.0;             // Triệt tiêu keep-right bias khi không có chướng ngại
		acc_gain_threshold <- 0.5;   // Ngưỡng gia tốc kích hoạt chuyển làn chống rung giật
		lane_change_limit <- 1;      // Mỗi bước chỉ đổi tối đa 1 làn (thay vì 2)
		lane_change_cooldown <- 3.0; // Khoảng thời gian hồi chiêu giữa 2 lần đổi làn
		max_safe_deceleration <- 3.0; // Không ép xe sau phanh quá mạnh khi nhập làn

		// Tham so IDM mac dinh theo tung loai xe se duoc thiet lap tai reflex sinh xe
		speed_coeff <- 1.0;
	}

	// ĐẾM THỜI GIAN CHỜ (cho metrics)
	reflex count_waiting when: final_target != nil {
		if (speed < 1.0 #km/#h) {
			waiting_time <- waiting_time + step;
		}
	}

	// Chỉ mở làn kề khi đang bị xe chậm cản và cả khoảng trống trước/sau đủ an toàn.
	// driving/MOBIL vẫn quyết định có đổi làn hay không trong tập làn được mở.
	reflex choose_safe_lane when: final_target != nil and current_road != nil {
		allowed_lanes <- [lowest_lane];
		list<base_vehicle_species> lane_vehicles <- road(current_road).all_agents collect base_vehicle_species(each);
		if (road(current_road).is_outer_road and
			distance_to(location, intersection_center) > (distance_to_intersection_center + 13.0 #m) and
			!empty(base_allowed_lanes) and
			!(intersection(current_target).managed_merge and distance_to_current_target < 120)) {
			bool slowed_by_leader <- false;
			loop other over: lane_vehicles {
				if (other != self and other.current_road = current_road) {
					float along <- (other.location.x - location.x) * cos(heading) +
						(other.location.y - location.y) * sin(heading);
					float gap <- along - (vehicle_length + other.vehicle_length) / 2.0;
					bool same_lane <- other.lowest_lane <= lowest_lane + num_lanes_occupied - 1 and
						other.lowest_lane + other.num_lanes_occupied - 1 >= lowest_lane;
					if (same_lane and along > 0.0 and
						gap < max([12.0 #m, speed * 2.0 #s]) and
						other.speed + 1.0 < speed) {
						slowed_by_leader <- true;
						break;
					}
				}
			}
			if (slowed_by_leader) {
				loop candidate over: base_allowed_lanes {
					if (abs(candidate - lowest_lane) = 1) {
						bool gap_safe <- true;
						loop other over: lane_vehicles {
							if (other != self and other.current_road = current_road) {
								bool shares_target <- other.lowest_lane <= candidate + num_lanes_occupied - 1 and
									other.lowest_lane + other.num_lanes_occupied - 1 >= candidate;
								if (shares_target) {
									float along <- (other.location.x - location.x) * cos(heading) +
										(other.location.y - location.y) * sin(heading);
									float bumper_gap <- abs(along) - (vehicle_length + other.vehicle_length) / 2.0;
									if ((along >= 0.0 and bumper_gap < max([4.0 #m, speed * 1.5 #s])) or
										(along < 0.0 and bumper_gap < max([5.0 #m, (other.speed - speed) * 2.5 #s + 5.0 #m]))) {
										gap_safe <- false;
										break;
									}
								}
							}
						}
						if (gap_safe) {
							allowed_lanes <- [lowest_lane, candidate];
							break;
						}
					}
				}
			}
		}
	}

	// =========================================================================
	// REFLEX 0: NỘI SUY HIỂN THỊ CHUYỂN LÀN MƯỢT MÀ (SMOOTHERSTEP)
	// =========================================================================
	reflex smooth_lateral_movement when: current_road != nil {
		if (location != nil and observed_lane_road = road(current_road) and observed_lane_index >= 0 and
			observed_lane_index != lowest_lane) {
			total_lane_change_events <- total_lane_change_events + 1;
			float stop_distance <- sqrt((location.x - intersection_center.x)^2 +
				(location.y - intersection_center.y)^2) -
				distance_to_intersection_center - signal_node_to_stop_bar;
			if (uses_signal and stop_distance >= 0.0 #m and stop_distance < 160.0 #m) {
				near_lane_change_events <- near_lane_change_events + 1;
			}
		}
		observed_lane_road <- road(current_road);
		observed_lane_index <- lowest_lane;
		lateral_velocity <- 0.0 #m/#s;
		float ideal_dist <- (road(current_road).num_lanes - lowest_lane - mean(range(num_lanes_occupied - 1)) - 0.5) * road(current_road).sublane_width + road(current_road).render_offset(location);
		if violating_oneway { ideal_dist <- -ideal_dist; }

		if (!lateral_shift_initialized) {
			current_lateral_shift <- ideal_dist;
			target_lateral_shift <- ideal_dist;
			lateral_shift_initialized <- true;
		}

		if (abs(ideal_dist - target_lateral_shift) > 0.001) {
			start_lateral_shift <- current_lateral_shift;
			target_lateral_shift <- ideal_dist;
			lane_change_progress <- 0.0;
		}

		if (lane_change_progress < 1.0) {
			if (speed > 0.5 #km/#h) {
				float lateral_dist <- abs(target_lateral_shift - start_lateral_shift);
				// Thời gian chuyển làn phụ thuộc khoảng cách ngang, giới hạn [0.8s, 2.5s]
				float base_duration <- lateral_dist / 1.5;
				float type_multiplier <- (vehicle_type = "motorcycle") ? 0.6 : ((vehicle_type = "bus") ? 1.8 : 1.0);
				float duration_cap <- max([0.8 * type_multiplier, min([2.5 * type_multiplier, base_duration * type_multiplier])]);

				lane_change_progress <- lane_change_progress + (step / duration_cap);
				if (lane_change_progress > 1.0) { lane_change_progress <- 1.0; }
			}

			float old_lateral_shift <- current_lateral_shift;

			// Smootherstep (Ken Perlin) bậc 5: 6t^5 - 15t^4 + 10t^3
			float t <- lane_change_progress;
			float smoother_t <- t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
			current_lateral_shift <- start_lateral_shift + (target_lateral_shift - start_lateral_shift) * smoother_t;

			// Tính lateral velocity bằng finite difference
			float lateral_vel <- (current_lateral_shift - old_lateral_shift) / max([0.001, step]);
			lateral_velocity <- lateral_vel;
			if (abs(lateral_vel) > abs(recent_lateral_peak)) {
				recent_lateral_peak <- lateral_vel;
			}

			if (!fast_training) {
				// Yaw is read only by the GUI drawing and visual integration test.
				float raw_yaw <- atan2(lateral_vel, max([1.0, speed]));
				float max_yaw_allowed <- (vehicle_type = "motorcycle") ? 30.0 : ((vehicle_type = "car") ? 15.0 : 8.0);
				raw_yaw <- max([-max_yaw_allowed, min([max_yaw_allowed, raw_yaw])]);
				current_yaw_deviation <- current_yaw_deviation + (raw_yaw - current_yaw_deviation) * 0.2;
			}
		} else {
			current_lateral_shift <- target_lateral_shift;
			if (!fast_training) {
				// Từ từ trả thẳng lái khi về đúng làn
				if (abs(current_yaw_deviation) > 0.1) {
					current_yaw_deviation <- current_yaw_deviation * 0.8;
				} else {
					current_yaw_deviation <- 0.0;
				}
			}
		}
	}

	// =========================================================================
	// REFLEX 1: QUAN SÁT & ĐIỀU HƯỚNG VI MÔ (GAP ACCEPTANCE & XI-NHAN)
	// =========================================================================
	reflex micro_navigation when: final_target != nil {
		if (!uses_signal) {
			max_speed <- original_max_speed;
			proba_respect_stops <- [0.0,0.0];
			if (maneuver = "right" or maneuver = "uturn") {
				float cap <- maneuver = "uturn" ? 8 #km/#h : 15 #km/#h;
				float to_turn <- max([0.0, distance_to(location, intersection_center) - (maneuver = "uturn" ? 70.0 : 80.0)]);
				float turn_decel <- maneuver = "uturn" ? 0.35 * max_deceleration : max_deceleration;
				max_speed <- min([max_speed, sqrt(cap * cap + 2.0 * turn_decel * to_turn)]);
			}
			return;
		}
		float dist_to_center <- distance_to(location, intersection_center);
		bool travels_ns <- approach_ns;
			bool direction_has_green <- maneuver = "left" ?
				((travels_ns and global_phase = 2) or (!travels_ns and global_phase = 6)) :
				((travels_ns and global_phase = 0) or (!travels_ns and global_phase = 4));
			bool direction_is_yellow_phase <- maneuver = "left" ?
				((travels_ns and global_phase = 3) or (!travels_ns and global_phase = 7)) :
				((travels_ns and global_phase = 1) or (!travels_ns and global_phase = 5));
			bool direction_has_yellow <- direction_is_yellow_phase and phase_timer < signal_yellow_seconds;

		// Khôi phục max_speed ban đầu trước khi tính toán điều hướng
			max_speed <- original_max_speed;
			float through_respect <- maneuver = "left" ? 0.0 : 1.0;
			float left_respect <- maneuver = "left" and !left_entered and
				!(direction_has_green and left_committed) ? 1.0 : 0.0;
			proba_respect_stops <- [through_respect, left_respect];
			// Ứng xử vùng phân vân: đi tiếp nếu không còn đủ đường phanh ở đầu vàng.
			// Xe còn đủ khoảng cách thì giảm tốc từ trước vạch dừng.
			bool approaching_signal <- uses_signal and current_road != nil and
				road(current_road).is_incoming and road(current_road).is_outer_road;
			float to_stop_line <- max([0.0 #m, dist_to_center - distance_to_intersection_center]);
			if (approaching_signal) {
				if (intersection(road(current_road).target_node).is_traffic_signal) {
					// Exact along-edge distance on the final approach segment.
					to_stop_line <- distance_to_current_target;
				}
			}
			// Once the front bumper passed the stop bar on yellow, the vehicle
			// must finish entering the junction even after the lamp turns red.
			bool crossed_stop_bar <- approaching_signal and
				to_stop_line <= signal_node_to_stop_bar + vehicle_length / 2.0 + stop_bar_clearance;
			if (!direction_is_yellow_phase and !(yellow_commit and crossed_stop_bar)) {
				yellow_commit <- false;
			}
			if (approaching_signal) {
					float stop_margin <- signal_node_to_stop_bar + vehicle_length / 2.0 + stop_bar_clearance;
					float stopping_distance <- measured_speed * measured_speed /
						(2.0 * max_deceleration) + stop_margin;
					// Include one short graph edge of reaction room: a vehicle that
					// cannot stop comfortably at yellow must clear before the next phase.
					if (direction_has_yellow and phase_timer < 3.0 #s and
						(maneuver != "left" or left_committed or left_entered) and
						to_stop_line < stopping_distance + 6.0 #m) { yellow_commit <- true; }
					if (yellow_commit and (direction_is_yellow_phase or crossed_stop_bar)) {
						proba_respect_stops <- [0.0, 0.0];
					} else if (!direction_has_green) {
						float usable_gap <- max([0.0 #m, to_stop_line - stop_margin]);
						// Begin braking with room for the DrivingSkill's reaction step.
						float safe_cap <- min([
							sqrt(0.2 * max_deceleration * usable_gap),
							usable_gap / (4.0 #s)]);
						// DrivingSkill responds gradually to max_speed. Ramp its command
						// through yellow, then let long buses finish braking on red.
						float commanded_cap <- max([0.1 #km/#h, safe_cap]);
						if (direction_is_yellow_phase or phase_timer < 1.0 #s) {
							float decel_factor <- direction_has_yellow ? 3.0 : 2.0;
							commanded_cap <- max([commanded_cap,
								measured_speed - decel_factor * max_deceleration * step]);
						}
						// On the final approach, brake early and taper near the bar.
						// Upstream short edges keep the gentler command; the stronger
						// one caused hard braking there when a leader slowed.
						bool final_signal_edge <- intersection(road(current_road).target_node).is_traffic_signal;
						float command_factor <- final_signal_edge ? (measured_speed > 2.0 #km/#h ? 2.5 : 1.0) : 1.0;
						max_speed <- min([max_speed,
							max([commanded_cap, measured_speed - command_factor * max_deceleration * step])]);
					}
			}
		// Hold left turns before the node until their protected phase grants entry.
		bool wait_for_turn <- maneuver = "left" and !left_committed and !left_entered;
		if (wait_for_turn and approaching_signal) {
				float to_node <- to_stop_line;
				float stop_margin <- signal_node_to_stop_bar + vehicle_length / 2.0 + stop_bar_clearance;
				float usable_gap <- max([0.0 #m, to_node - stop_margin]);
				float desired_cap <- max([0.1 #km/#h,
					min([sqrt(0.2 * max_deceleration * usable_gap), usable_gap / (4.0 #s)])]);
				max_speed <- min([max_speed, desired_cap]);
			}
			if (maneuver = "left" and dist_to_center < (distance_to_intersection_center + 55.0 #m) and
				(current_road = nil or road(current_road).is_outer_road)) {
				float distance_before_turn <- max([0.0 #m, dist_to_center - distance_to_intersection_center]);
				float turn_speed <- 18.0 #km/#h;
				max_speed <- min([max_speed, sqrt(turn_speed * turn_speed +
					2.0 * max_deceleration * distance_before_turn)]);
			}
			if (maneuver = "left" and current_road != nil and
				road(current_road).is_signal_connector) {
				max_speed <- min([max_speed, 18.0 #km/#h]);
			}

		// B2: XE RẼ PHẢI GIẢM TỐC CHỦ ĐỘNG TRƯỚC KHI VÀO CUNG
		if (maneuver = "right" and dist_to_center < distance_to_intersection_center) {
			// Giới hạn tốc độ rẽ phải: 12-15 km/h ôm sát lề
			max_speed <- min([max_speed, 15.0 #km / #h]);
		}

		// DrivingSkill already follows the leader in the occupied sublanes. Do not
		// impose a second proximity-based motorcycle creep cap here: a vehicle in
		// an adjacent lane used to be mistaken for a blocker and pinned otherwise
		// free motorcycles at 0.1 km/h several metres before the stop bar.
	}

	// =========================================================================
	// Brake before a slower connector, including ramps. DrivingSkill applies
	// maxspeed at the edge boundary; anticipating it avoids an instantaneous drop.
	reflex anticipate_speed_limit when: final_target != nil and current_road != nil and next_road != nil {
		float next_limit <- road(next_road).maxspeed;
		float usable <- road(next_road).road_kind = "uturn" ? distance_to_current_target :
			max([0.0, distance_to_current_target - vehicle_length / 2.0 - min_safety_distance - 5.0]);
		max_speed <- min([max_speed, sqrt(next_limit * next_limit + 2.0 * max_deceleration * usable)]);
	}

	reflex follow_current_ground_edge when: final_target != nil and current_road != nil and
		road(current_road).road_kind = "ground" {
		float nearest_gap <- 1e9 #m;
		float leader_speed <- 0.0 #m/#s;
		loop other over: road(current_road).all_agents collect base_vehicle_species(each) {
			if (other != self and other.final_target != nil and
				other.distance_to_current_target < distance_to_current_target and
				other.lowest_lane < lowest_lane + num_lanes_occupied and
				other.lowest_lane + other.num_lanes_occupied > lowest_lane) {
				float gap <- distance_to_current_target - other.distance_to_current_target -
					(vehicle_length + other.vehicle_length) / 2.0 - min_safety_distance - 2.0 #m;
				if (gap < nearest_gap) { nearest_gap <- gap; leader_speed <- other.measured_speed; }
			}
		}
		if (nearest_gap < 80 #m) {
			float cap <- nearest_gap <= 0.0 ? 0.01 #km/#h :
				min([sqrt(leader_speed * leader_speed + 1.2 * max_deceleration * nearest_gap),
					leader_speed + nearest_gap / (2.5 #s)]);
			if (maneuver = "uturn" and next_road != nil and
				road(next_road).shape.perimeter < 30.0 #m and
				nearest_gap < 8.0 #m) {
				cap <- min([cap, nearest_gap / (4.0 #s)]);
			}
			float decel_allowance <- 2.0 * max_deceleration * step;
			max_speed <- min([max_speed, max([cap, measured_speed - decel_allowance])]);
		}
	}

	// Look across a short ground edge before its leader becomes visible to
	// DrivingSkill on the same edge. This avoids an abrupt stop at U_IN/DIV.
	reflex follow_next_ground_edge when: final_target != nil and current_road != nil and next_road != nil and
		road(next_road).road_kind = "ground" {
		float nearest_gap <- 1e9 #m;
		float leader_speed <- 0.0 #m/#s;
		list<road> lookahead_edges <- [road(next_road)];
		if (road(next_road).shape.perimeter < 30.0 #m) {
			intersection mid_node <- intersection(road(next_road).target_node);
			road edge_after <- first(mid_node.roads_out where (road(each).road_kind = "ground" and road(each).is_incoming));
			if (edge_after != nil) {
				lookahead_edges <- lookahead_edges + edge_after;
			}
		}
		loop following_edge over: lookahead_edges {
			float dist_to_following_edge <- following_edge = road(next_road) ?
				distance_to_current_target :
				distance_to_current_target + road(next_road).shape.perimeter;
			loop other over: following_edge.all_agents collect base_vehicle_species(each) {
				if (other != self and other.final_target != nil and
					other.lowest_lane < lowest_lane + num_lanes_occupied and
					other.lowest_lane + other.num_lanes_occupied > lowest_lane) {
					float progress <- following_edge.shape.perimeter - other.distance_to_current_target;
					float gap <- dist_to_following_edge + progress -
						(vehicle_length + other.vehicle_length) / 2.0 - min_safety_distance - 2.0 #m;
					if (gap < nearest_gap) { nearest_gap <- gap; leader_speed <- other.measured_speed; }
				}
			}
		}
		if (nearest_gap < 80 #m) {
			float cap <- nearest_gap <= 0.0 ? 0.01 #km/#h :
				min([sqrt(leader_speed * leader_speed + 1.2 * max_deceleration * nearest_gap),
					leader_speed + nearest_gap / (2.5 #s)]);
			// A leader on a short next edge can stop before DrivingSkill's
			// connector transition sees it. Keep a small absolute headway speed
			// close to that edge, even while the leader is still moving.
			if (maneuver = "uturn" and road(next_road).shape.perimeter < 30.0 #m and nearest_gap < 8.0 #m) {
				cap <- min([cap, nearest_gap / (4.0 #s)]);
			}
			float decel_allowance <- 2.0 * max_deceleration * step;
			max_speed <- min([max_speed, max([cap, measured_speed - decel_allowance])]);
		}
	}

	reflex yield_at_merge when: final_target != nil and current_road != nil and current_target != nil {
		intersection merge <- intersection(current_target);
		bool through_uturn_exit <- maneuver != "uturn" and
			(merge.node_key = "E_U_OUT" or merge.node_key = "N_U_OUT" or
			 merge.node_key = "W_U_OUT" or merge.node_key = "S_U_OUT");
		if (merge.managed_merge and !through_uturn_exit) {
			// The explicit reservation replaces the skill's lane-blind right-of-way
			// check here. Stops at signal nodes still use the existing controller.
			proba_respect_priorities <- 0.0;
			if (!(merge.merge_owners contains self)) {
				float usable <- max([0.0,distance_to_current_target - merge.merge_clearance - vehicle_length / 2.0 - min_safety_distance - 4.0]);
				float cap <- max([0.01 #km/#h,sqrt(2.0 * max_deceleration * usable)]);
				// Once the stopping margin is used up, the IDM free-flow term
				// lets a long queue creep into the portal over thousands of cycles.
				// Keep GAMA's tiny nonzero speed floor to avoid a zero-speed NaN.
				max_speed <- min([max_speed,usable <= 0.0 ?
					max([0.01 #km/#h,measured_speed - max_deceleration * step]) :
					max([cap,measured_speed - max_deceleration * step,
						speed / ((1.0 + max_deceleration / max_acceleration) ^ (1.0 / delta_idm))])]);
			}
		} else {
			// Signal connectors have already been admitted by their own phase.
			// Do not let DrivingSkill stop them abruptly at the outbound node.
			proba_respect_priorities <- current_road != nil and
				road(current_road).is_signal_connector ? 0.0 : 1.0;
		}
		if (next_road != nil) {
			intersection ahead <- intersection(road(next_road).target_node);
			bool through_ahead <- maneuver != "uturn" and
				(ahead.node_key = "E_U_OUT" or ahead.node_key = "N_U_OUT" or
				 ahead.node_key = "W_U_OUT" or ahead.node_key = "S_U_OUT");
			if (ahead.managed_merge and !through_ahead and !(ahead.merge_owners contains self)) {
				float usable <- max([0.0,distance_to_current_target + road(next_road).shape.perimeter - ahead.merge_clearance - vehicle_length / 2.0 - min_safety_distance - 4.0]);
				float cap <- max([0.01 #km/#h,sqrt(2.0 * max_deceleration * usable)]);
				max_speed <- min([max_speed,usable <= 0.0 ?
					max([0.01 #km/#h,measured_speed - max_deceleration * step]) :
					max([cap,measured_speed - max_deceleration * step,
						speed / ((1.0 + max_deceleration / max_acceleration) ^ (1.0 / delta_idm))])]);
			}
		}
	}

	// DrivingSkill's straight-line look-ahead can miss a stopped leader on a
	// curved right connector. Compare progress along the shared edge instead.
	reflex follow_right_connector when: final_target != nil and current_road != nil and road(current_road).road_kind = "right" {
		float nearest_gap <- 1e9 #m;
		float leader_speed <- 0.0 #m/#s;
		loop other over: road(current_road).all_agents collect base_vehicle_species(each) {
			if (other != self and other.final_target != nil and
				other.lowest_lane < lowest_lane + num_lanes_occupied and
				other.lowest_lane + other.num_lanes_occupied > lowest_lane and
				other.distance_to_current_target < distance_to_current_target) {
				float gap <- distance_to_current_target - other.distance_to_current_target -
					(vehicle_length + other.vehicle_length) / 2.0 - min_safety_distance - 2.0 #m;
				if (gap < nearest_gap) { nearest_gap <- gap; leader_speed <- other.measured_speed; }
			}
		}
		if (nearest_gap < 1e8 #m) {
			float cap <- nearest_gap <= 0.0 ? 0.01 #km/#h :
				sqrt(leader_speed * leader_speed + 2.0 * max_deceleration * nearest_gap);
			max_speed <- min([max_speed,max([cap,measured_speed - max_deceleration * step])]);
		}
	}

	bool test_next_road(agent new_road) {
		if (current_target != nil) {
			intersection node <- intersection(current_target);
			bool through_uturn_exit <- maneuver != "uturn" and
				(node.node_key = "E_U_OUT" or node.node_key = "N_U_OUT" or
				 node.node_key = "W_U_OUT" or node.node_key = "S_U_OUT");
			if (node.managed_merge and !through_uturn_exit and !(node.merge_owners contains self)) { return false; }
		}
		if (new_road != nil) {
			road upcoming <- road(new_road);
			if (upcoming.is_signal_connector) {
				bool committed_crossing <- yellow_commit and current_road != nil and
					road(current_road).is_incoming and distance_to_current_target <=
					signal_node_to_stop_bar + vehicle_length / 2.0 + stop_bar_clearance;
				bool yellow_pass <- phase_timer < signal_yellow_seconds or yellow_commit;
				bool permitted <- maneuver = "left" ?
					(left_committed and ((approach_ns and (global_phase = 2 or (global_phase = 3 and yellow_pass))) or
					 (!approach_ns and (global_phase = 6 or (global_phase = 7 and yellow_pass))))) :
					((approach_ns and (global_phase = 0 or (global_phase = 1 and yellow_pass))) or
					 (!approach_ns and (global_phase = 4 or (global_phase = 5 and yellow_pass))));
				if (!permitted and !committed_crossing) { return false; }
			}
			intersection ahead <- intersection(upcoming.target_node);
			bool through_ahead <- maneuver != "uturn" and
				(ahead.node_key = "E_U_OUT" or ahead.node_key = "N_U_OUT" or
				 ahead.node_key = "W_U_OUT" or ahead.node_key = "S_U_OUT");
			float stop_margin <- ahead.merge_clearance + vehicle_length / 2.0 + min_safety_distance + 4.0 #m;
			if (ahead.managed_merge and !through_ahead and upcoming.shape.perimeter < stop_margin and
				!(ahead.merge_owners contains self)) { return false; }
		}
		return true;
	}

	// REFLEX 2: ĐO LƯỜNG AN TOÀN VI MÔ (SURROGATE SAFETY MEASURES - SSM)
	// =========================================================================
	reflex micro_safety_sensing when: final_target != nil and enable_surrogate_safety {
		// 1. Giam sat gia toc & phanh khanc cap (Hard-Braking)
		current_acc <- (measured_speed - prev_speed) / step;
		prev_speed <- measured_speed;
		if (current_acc < -3.4 and !was_hard_braking) {
			hard_braking_count <- hard_braking_count + 1;
		}
		if (current_acc < -5.0 and !was_panic_braking) { panic_braking_count <- panic_braking_count + 1; }
		was_hard_braking <- current_acc < -3.4;
		was_panic_braking <- current_acc < -5.0;

		// Each following pair is counted by its rear vehicle, independent of names.
		list<base_vehicle_species> severe_now <- [];
		list<base_vehicle_species> neighbors <- base_vehicle_species at_distance 15.0 #m;
		loop target_v over: neighbors {
			if (target_v != self and target_v.final_target != nil and
				abs(surface_height() - target_v.surface_height()) < 2.5) {
				float bearing <- self.location direction_to target_v.location;
				float angle_diff <- abs(heading - bearing);
				if (angle_diff > 180.0) { angle_diff <- 360.0 - angle_diff; }

				if (angle_diff < 45.0) {
					float rel_speed <- self.speed - target_v.speed;
					float gap <- distance_to(self.location, target_v.location) - (self.vehicle_length / 2.0 + target_v.vehicle_length / 2.0);

					if (rel_speed > 0.5 and gap > 0.0) {
						float ttc <- following_ttc(target_v);
						if (ttc < min_ttc_experienced) { min_ttc_experienced <- ttc; }

						// Do lech tam CS (Centerline Separation - Das & Maurya 2020)
						float cs <- abs((target_v.location.x - self.location.x) * sin(heading) - (target_v.location.y - self.location.y) * cos(heading));

						// Nguong an toan dong ttc_star
						float ttc_star <- 3.0; // O to
						if (vehicle_type = "motorcycle") {
							ttc_star <- (cs > 1.0) ? 1.20 : 1.58;
						}

						// Tich phan rui ro TIT
						if (ttc < ttc_star) {
							float d_tit <- (ttc_star - ttc) * step;
							cycle_tit <- cycle_tit + d_tit;
							total_tit_sum <- total_tit_sum + d_tit;

							if (ttc < 1.0) {
								severe_now <- severe_now + target_v;
								if (!(previous_severe_partners contains target_v)) {
									severe_conflicts_count <- severe_conflicts_count + 1;
								}
							}
						}

						// A2 FIX: DRAC clamp — floor gap 0.5m, cap 15 m/s²
						float drac <- min([15.0, (rel_speed * rel_speed) / (2.0 * max([0.5, gap]))]);
						if (drac > drac_max_observed) {
							drac_max_observed <- drac;
						}
					}
				}
			}
		}
		previous_severe_partners <- severe_now;

		// 3. Cam bien Vung phan van (Dilemma Zone)
		float dist_center <- distance_to(self.location, intersection_center);
		if (uses_signal and dist_center >= (distance_to_intersection_center - 3.0 #m) and dist_center <= (distance_to_intersection_center + 20.0 #m) and self.speed > (15.0 #km / #h)) {
			bool own_yellow <- maneuver = "left" ?
				((approach_ns and global_phase = 3) or (!approach_ns and global_phase = 7)) :
				((approach_ns and global_phase = 1) or (!approach_ns and global_phase = 5));
			if (own_yellow) {
				is_in_dz <- true;
			} else {
				is_in_dz <- false;
			}
			} else {
				is_in_dz <- false;
			}
			if (is_in_dz) {
				dilemma_zone_vehicles_count <- dilemma_zone_vehicles_count + 1;
			}
		}

	// =========================================================================
	// REFLEX 3: DI CHUYỂN & GIẢI PHÓNG TÁC TỬ
	// =========================================================================
	reflex move when: final_target != nil {
		// A stopped vehicle's front bumper must remain behind the painted bar.
		// DrivingSkill retains a small speed floor, so explicitly hold at the bar.
		if (uses_signal and current_road != nil and road(current_road).is_incoming and
			intersection(road(current_road).target_node).is_traffic_signal) {
			bool has_green <- maneuver = "left" ?
				((approach_ns and global_phase = 2) or (!approach_ns and global_phase = 6)) :
				((approach_ns and global_phase = 0) or (!approach_ns and global_phase = 4));
			bool has_yellow <- maneuver = "left" ?
				((approach_ns and global_phase = 3) or (!approach_ns and global_phase = 7)) :
				((approach_ns and global_phase = 1) or (!approach_ns and global_phase = 5));
			float remaining <- distance_to_current_target - signal_node_to_stop_bar -
				vehicle_length / 2.0 - stop_bar_clearance;
			bool must_wait <- (!has_green and !(has_yellow and yellow_commit) and
				!(yellow_commit and remaining <= 0.0 #m)) or
				(maneuver = "left" and !left_committed and !left_entered);
			if (must_wait and remaining <= max([0.5 #m, measured_speed * step])) {
				if (measured_speed < 2.0 #km/#h or remaining <= 0.0 #m) {
					measured_speed <- 0.0 #km/#h;
					speed <- 0.0 #km/#h;
					return;
				}
			}
		}
		// Prevent IDM free-flow creep from drifting into a converging merge while waiting for a reservation
		if (current_target != nil) {
			intersection merge_node <- intersection(current_target);
			bool through_uturn_exit <- maneuver != "uturn" and
				(merge_node.node_key = "E_U_OUT" or merge_node.node_key = "N_U_OUT" or
				 merge_node.node_key = "W_U_OUT" or merge_node.node_key = "S_U_OUT");
			if (merge_node.managed_merge and !through_uturn_exit and !(merge_node.merge_owners contains self)) {
				float merge_usable <- distance_to_current_target - merge_node.merge_clearance -
					vehicle_length / 2.0 - min_safety_distance - 2.0 #m;
				if (merge_usable <= 0.0 #m and measured_speed < 1.5 #km/#h) {
					measured_speed <- 0.0 #km/#h;
					speed <- 0.0 #km/#h;
					return;
				}
			}
		}
		if (next_road != nil) {
			intersection ahead_node <- intersection(road(next_road).target_node);
			bool through_ahead <- maneuver != "uturn" and
				(ahead_node.node_key = "E_U_OUT" or ahead_node.node_key = "N_U_OUT" or
				 ahead_node.node_key = "W_U_OUT" or ahead_node.node_key = "S_U_OUT");
			if (ahead_node.managed_merge and !through_ahead and !(ahead_node.merge_owners contains self)) {
				float ahead_usable <- distance_to_current_target + road(next_road).shape.perimeter -
					ahead_node.merge_clearance - vehicle_length / 2.0 - min_safety_distance - 2.0 #m;
				if (ahead_usable <= 0.0 #m and measured_speed < 1.5 #km/#h) {
					measured_speed <- 0.0 #km/#h;
					speed <- 0.0 #km/#h;
					return;
				}
			}
		}
		point before_drive <- location;
		do drive; // Luôn gọi drive để skill quản lý di chuyển mượt mà
		measured_speed <- distance_to(location, before_drive) / step;
		if (current_road != nil and !lateral_shift_initialized) {
			current_lateral_shift <- (road(current_road).num_lanes - lowest_lane -
				mean(range(num_lanes_occupied-1)) - 0.5) * road(current_road).sublane_width + road(current_road).render_offset(location);
			target_lateral_shift <- current_lateral_shift;
			lateral_shift_initialized <- true;
		}
		if (maneuver = "left" and current_road != nil and road(current_road).is_signal_connector) {
			left_entered <- true;
		}

		if (debug_vehicle_log and self.name = "base_vehicle_species0") {
			write "TEST_SPEED," + cycle + "," + speed + "," + max_speed + "," + current_acc + "," + min_ttc_experienced;
		}

		if (final_target = nil) {
			if (maneuver = "left") { left_turn_completed_count <- left_turn_completed_count + 1; }
			if (maneuver = "right") { right_turn_completed_count <- right_turn_completed_count + 1; }
			if (maneuver = "uturn") { uturn_completed_count <- uturn_completed_count + 1; }
			if (route_kind != "ground") {
				bypass_vehicles_passed <- bypass_vehicles_passed + 1;
				route_completed[route_kind] <- (route_completed contains_key route_kind ? route_completed[route_kind] : 0) + 1;
			} else {
				total_vehicles_passed <- total_vehicles_passed + 1;
				total_travel_time_sum <- total_travel_time_sum + (time - spawn_time);
				average_travel_time <- total_travel_time_sum / total_vehicles_passed;
			}
			do unregister;
			location <- {0, 0};
		}
	}

	// TÍNH TOÁN TỌA ĐỘ HIỂN THỊ CHUẨN THEO LÀN VÀ ĐỘ LỆCH TÂM
	point compute_position {
		if (current_road != nil) {
			point shift_pt <- {cos(heading + 90) * current_lateral_shift,
				sin(heading + 90) * current_lateral_shift};
			float elevation <- road(current_road).road_kind = "flyover" ? 2.2 : 1.6;
			return location + shift_pt + {0,0,elevation + surface_height()};
		} else { return location; }
	}

	// =========================================================================
	// ASPECT RENDER PHƯƠNG TIỆN CHI TIẾT SẮC NÉT (XE MÁY, Ô TÔ, XE BUS)
	// =========================================================================
	aspect base {
		if (current_road != nil) {
			point pos <- compute_position();
			float render_heading <- heading + current_yaw_deviation;

			// -------------------------------------------------------------
			// TRƯỜNG HỢP 1: XE MÁY (MOTORCYCLE)
			// -------------------------------------------------------------
			if (vehicle_type = "motorcycle") {
				// 1. Banh xe truoc va sau (mau den)
				point f_tire <- pos + {cos(render_heading) * 0.8, sin(render_heading) * 0.8};
				point r_tire <- pos - {cos(render_heading) * 0.8, sin(render_heading) * 0.8};
				draw rectangle(0.45, 0.15) at: f_tire color: #black rotate: render_heading;
				draw rectangle(0.45, 0.15) at: r_tire color: #black rotate: render_heading;

				// 2. Than xe may thon dai
				draw rectangle(1.6, 0.45) at: pos color: color rotate: render_heading border: #black;

				// 3. Tay lai (Handlebar)
				point h_bar <- pos + {cos(render_heading) * 0.4, sin(render_heading) * 0.4};
				draw rectangle(0.15, 0.7) at: h_bar color: #silver rotate: render_heading;

				// 4. Nguoi lai voi mu bao hiem tron (Helmet)
				draw circle(0.35) at: pos color: rgb(240, 240, 240) border: #black;
				draw circle(0.2) at: pos + {cos(render_heading)*0.15, sin(render_heading)*0.15} color: #dimgray;
			}
			// -------------------------------------------------------------
			// TRƯỜNG HỢP 2: Ô TÔ CON (CAR)
			// -------------------------------------------------------------
			else if (vehicle_type = "car") {
				float draw_width <- vehicle_width;

				// 1. Khung xe ngoai (Metallic Chassis)
				draw rectangle(vehicle_length, draw_width) at: pos color: color rotate: render_heading border: #black;

				// 2. Kinh chan gio truoc (Front Windshield)
				point f_windshield <- pos + {cos(render_heading) * (vehicle_length * 0.22), sin(render_heading) * (vehicle_length * 0.22)};
				draw rectangle(vehicle_length * 0.22, draw_width * 0.75) at: f_windshield color: rgb(45, 55, 72) rotate: render_heading border: false;

				// 3. Noc xe (Car Roof)
				draw rectangle(vehicle_length * 0.35, draw_width * 0.78) at: pos color: (color - 20) rotate: render_heading border: false;

				// 4. Kinh sau (Rear Windshield)
				point r_windshield <- pos - {cos(render_heading) * (vehicle_length * 0.22), sin(render_heading) * (vehicle_length * 0.22)};
				draw rectangle(vehicle_length * 0.18, draw_width * 0.75) at: r_windshield color: rgb(45, 55, 72) rotate: render_heading border: false;
			}
			// -------------------------------------------------------------
			// TRƯỜNG HỢP 3: XE BUS / XE TẢI LỚN (C4: XE BUS ĐÔ THỊ)
			// -------------------------------------------------------------
			else {
				float draw_width <- vehicle_width;
				// 1. Thân xe bus vàng đô thị
				draw rectangle(vehicle_length, draw_width) at: pos color: rgb(245, 166, 35) rotate: render_heading border: #black;

				// 2. Kính chắn gió trước bản rộng
				point bus_front <- pos + {cos(render_heading) * (vehicle_length * 0.42), sin(render_heading) * (vehicle_length * 0.42)};
				draw rectangle(vehicle_length * 0.12, draw_width * 0.88) at: bus_front color: rgb(40, 50, 70) rotate: render_heading;

				// 3. Kính cửa sổ hai bên sườn
				draw rectangle(vehicle_length * 0.65, draw_width * 0.42) at: pos color: rgb(220, 230, 242) rotate: render_heading border: rgb(100, 110, 130);

				// 4. Khối điều hòa trên nóc (Roof AC Unit)
				point roof_ac <- pos - {cos(render_heading) * 0.5, sin(render_heading) * 0.5};
				draw rectangle(vehicle_length * 0.22, draw_width * 0.5) at: roof_ac color: rgb(200, 205, 215) rotate: render_heading border: rgb(130, 135, 145);
			}
		}
	}

	// Solid vehicles for the infrastructure camera. The overview keeps its
	// existing detailed 2D symbols.
	aspect doan_3d {
		if (current_road != nil) {
			point pos <- compute_position();
			float render_heading <- heading + current_yaw_deviation;
			if (vehicle_type = "motorcycle") {
				draw box(1.8, 0.7, 0.6) at: pos - {0,0,0.9}
					color: color rotate: render_heading border: #black;
				draw sphere(0.42) at: pos + {0,0,0.35}
					color: rgb(235,238,242) border: #black;
			} else if (vehicle_type = "car") {
				point body <- pos - {0,0,0.6};
				draw box(vehicle_length, vehicle_width, 1.5) at: body
					color: color rotate: render_heading border: #black;
				draw box(vehicle_length * 0.52, vehicle_width * 0.82, 0.45)
					at: body + {0,0,0.9} color: rgb(54,70,88)
					rotate: render_heading border: #black;
			} else {
				draw box(vehicle_length, vehicle_width, 2.8) at: pos
					color: rgb(245,166,35) rotate: render_heading border: #black;
				draw box(vehicle_length * 0.75, vehicle_width * 0.84, 0.25)
					at: pos + {0,0,1.52} color: rgb(52,67,82)
					rotate: render_heading border: #black;
			}
		}
	}
}

// =========================================================================
// LOÀI TÁC TỬ: CONFLICT ZONE (Ô XUNG ĐỘT TẠI TÂM NGÃ TƯ)
// =========================================================================
species conflict_zone {
	int zone_id;
	string last_occupant_name <- "";
	float last_occupant_heading <- 0.0;
	float last_exit_time <- -999.0;
	list<base_vehicle_species> current_occupants <- [];

	reflex monitor_encroachment when: enable_surrogate_safety {
		list<base_vehicle_species> present_vehicles <- (base_vehicle_species inside self.shape) where (each.final_target != nil and each.uses_signal and abs(each.surface_height()) < 2.5);

		loop past_v over: current_occupants {
			if (!(present_vehicles contains past_v)) {
				if (!dead(past_v)) {
					last_occupant_name <- past_v.name;
					last_occupant_heading <- past_v.heading;
				}
				last_exit_time <- time;
			}
		}

		loop new_v over: present_vehicles {
			if (!(current_occupants contains new_v)) {
				if (last_occupant_name != "" and last_occupant_name != new_v.name and last_exit_time > 0.0) {
					float diff_h <- abs(new_v.heading - last_occupant_heading);
					if (diff_h > 180.0) { diff_h <- 360.0 - diff_h; }

					if (diff_h > 45.0) {
						float raw_pet <- time - last_exit_time;
						float pet <- max([0.1, round(raw_pet * 10.0) / 10.0]);

						if (raw_pet >= 0.0 and raw_pet < 4.0) {
							recent_pet <- pet;
							if (pet < min_pet_observed) {
								min_pet_observed <- pet;
							}
							if (pet < recent_phase_min_pet) {
								recent_phase_min_pet <- pet;
							}
							if (pet < 1.5) {
								pet_conflicts_count <- pet_conflicts_count + 1;
							}
							if (pet < 1.0) {
								critical_pet_count <- critical_pet_count + 1;
							}
						}
					}
				}
			}
		}

		current_occupants <- present_vehicles;
	}

	aspect base {
		draw shape color: rgb(255, 60, 60, 35) border: #red;
	}
}
