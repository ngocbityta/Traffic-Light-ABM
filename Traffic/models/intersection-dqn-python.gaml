/**
* Name: intersection_dqn_python
* Description: Four-phase 3DQN traffic signal controller driven by Python.
*   DTSE is captured at decision ticks; action 0/1/2/3 selects NS through,
*   NS left, EW through or EW left. Repeating a phase extends it for 10 s.
*   A change uses 3 s yellow, at least 2 s all red until clear, then 10 s green.
*
*   Handshake protocol (Python <- -> GAMA), moi diem quyet dinh:
*     1. Python GET state_pack and capture_dtse (at a decision tick)
*     2. Python SET  dqn_action
*     3. Python step(1)                (controller ap dung action)
*     4. Python GET  next_decision_in_cycles
*     5. Python step(next_decision_in_cycles)  -> ve lai dung tick ke tiep
*
*   Integer cycle counters keep the 0.1 s schedule deterministic.
*/

model intersection_dqn_python

import "../includes/collision_avoidance.gaml"

global {
	float step <- 0.1 #s;

	// =====================================================
	// TIMING & SAFETY CONSTRAINTS
	// =====================================================
	// 0/4: through green, 1/5: through yellow,
	// 2/6: protected left green, 3/7: left yellow.
	int green_cycles <- 0;            // so cycle da troi trong pha xanh hien tai
	int yellow_cycles <- 0;           // so cycle da troi trong pha vang hien tai
	int min_green_cycles <- 100;      // 10 s / 0.1 s
	int yellow_len_cycles <- 30;      // 3 s / 0.1 s

	// =====================================================
	// PYTHON DQN AGENT INTERFACE
	// =====================================================
	int dqn_action <- -1;             // 0=NS through, 1=NS left, 2=EW through, 3=EW left
	bool dqn_waiting_action <- false; // true = controller dang cho action tu Python
	int next_decision_in_cycles <- 100; // Python step den tick ke tiep
	int current_action <- 0;          // action cuoi cung da ap dung
	int decision_count <- 0;          // tong so quyet dinh AI da thuc hien
	bool python_controlled <- true;   // false = chay heuristic fallback (demo khong can Python)
	int pending_green_phase <- 0;
	int all_red_cycles <- 0;
	bool dqn_all_red <- false;
	string dtse_pack <- "";
	string spatial_pack <- "";
	list<float> spatial_pressure <- [0.0,0.0,0.0,0.0];
	int spatial_capture_cycle <- -1;
	float dtse_delay_sq <- 0.0;
	float dtse_delay_sq_near <- 0.0;
	bool dqn_service_guard <- false; // Set to false to match paper's pure 4-phase 3DQN without heuristic phase forcing
	list<int> dqn_action_mask <- [1,1,1,1];
		list<int> phase_wait_cycles <- [0,0,0,0];
		list<int> phase_demand <- [0,0,0,0];
		int max_phase_wait_cycles <- 1200;
		int max_green_cycles <- 600;

	// Called only at decision ticks through gama-server. Shape: [3,8,20].
		// Rows E,N,W,S x (through,left); cells aggregate movement groups.
		// Mean speed makes the grouped observation independent of vehicle order.
	action capture_dtse {
		list<float> cells <- [];
			list<int> hits <- [];
		loop i from: 0 to: 479 { cells <- cells + 0.0; }
			loop i from: 0 to: 159 { hits <- hits + 0; }
		dtse_delay_sq <- 0.0;
		dtse_delay_sq_near <- 0.0;
		loop v over: active_vehicles {
			if (v.current_road != nil and v.uses_signal and v.approach_id >= 4 and v.approach_id <= 7 and
				road(v.current_road).is_outer_road and road(v.current_road).is_incoming) {
				float speed_fraction <- min([1.0, max([0.0, v.speed / (50.0 #km/#h)])]);
				float vehicle_delay_sq <- 1.0 - speed_fraction * speed_fraction;
				// Reward covers every incoming signal vehicle, as in the SUMO reference.
				dtse_delay_sq <- dtse_delay_sq + vehicle_delay_sq;
				float distance_from_stop <- distance_to(v.location, intersection_center) - distance_to_intersection_center;
				if (distance_from_stop >= 0.0 #m and distance_from_stop < 160.0 #m) {
					int row <- (v.approach_id - 4) * 2 + (v.maneuver = "left" ? 1 : 0);
					int col <- int(distance_from_stop / (8.0 #m));
					int index <- row * 20 + col;
					cells[index] <- 1.0;
						hits[index] <- hits[index] + 1;
						cells[160 + index] <- cells[160 + index] + speed_fraction;
					dtse_delay_sq_near <- dtse_delay_sq_near + vehicle_delay_sq;
				}
			}
		}
		// Average each cell once after accumulating every vehicle. Doing this
		// inside the vehicle loop makes the result depend on vehicle order/count.
		loop i from: 0 to: 159 {
			if (hits[i] > 0) { cells[160 + i] <- cells[160 + i] / hits[i]; }
		}
		loop row from: 0 to: 7 {
			int approach <- 4 + floor(row / 2);
			int served_phase <- (approach = 5 or approach = 7) ? (row mod 2 = 1 ? 2 : 0) :
				(row mod 2 = 1 ? 6 : 4);
			if (global_phase = served_phase and !dqn_all_red) {
				loop col from: 0 to: 19 { cells[320 + row * 20 + col] <- 1.0; }
			}
		}
		dtse_pack <- "" + cells;
	}

	// Research observation: [4 approaches E,N,W,S, 6 channels, 20 distance
	// cells, 16 lateral cells]. C0-C2 are occupied footprint fractions by
	// vehicle class; C3/C4 are footprint-weighted velocities; C5 is road mask.
	// The present road graph has straight, uniform 16-sublane approaches.
	action capture_spatial_state {
		if (spatial_capture_cycle = cycle) { return; }
		list<float> incoming_movement <- [];
		list<float> outgoing_queue <- [0.0,0.0,0.0,0.0];
		loop i from: 0 to: 7 { incoming_movement <- incoming_movement + 0.0; }
		list<float> pixels <- [];
		list<float> moving_area <- [];
		loop i from: 0 to: 7679 { pixels <- pixels + 0.0; }
		loop i from: 0 to: 1279 { moving_area <- moving_area + 0.0; }
		loop arm from: 0 to: 3 {
			loop h from: 0 to: 19 {
				loop w from: 0 to: 15 {
					pixels[(((arm * 6 + 5) * 20 + h) * 16 + w)] <- w < nb_lanes ? 1.0 : 0.0;
				}
			}
		}
		loop v over: active_vehicles {
			if (v.current_road != nil and v.location != nil and v.route_kind = "ground" and
				road(v.current_road).is_outer_road and !road(v.current_road).is_incoming and
				v.destination_approach >= 4 and v.destination_approach <= 7 and
				v.measured_speed < 5.0 #km/#h and
				sqrt((v.location.x - intersection_center.x)^2 +
					(v.location.y - intersection_center.y)^2) < distance_to_intersection_center + 80.0 #m) {
				outgoing_queue[v.destination_approach - 4] <- outgoing_queue[v.destination_approach - 4] + 1.0;
			}
			if (v.current_road != nil and v.location != nil and v.uses_signal and v.approach_id >= 4 and v.approach_id <= 7 and
				road(v.current_road).is_outer_road and road(v.current_road).is_incoming) {
				road r <- road(v.current_road);
				int arm <- v.approach_id - 4;
				float along <- sqrt((v.location.x - intersection_center.x)^2 +
					(v.location.y - intersection_center.y)^2) -
					distance_to_intersection_center - signal_node_to_stop_bar;
				if (along >= 0.0 #m and along < 160.0 #m and
					v.measured_speed < 5.0 #km/#h) {
					incoming_movement[arm * 2 + (v.maneuver = "left" ? 1 : 0)] <-
						incoming_movement[arm * 2 + (v.maneuver = "left" ? 1 : 0)] + 1.0;
				}
				float along_lo <- along - v.vehicle_length / 2.0;
				float along_hi <- along + v.vehicle_length / 2.0;
				float lane_center <- r.num_lanes -
					(v.current_lateral_shift - r.render_offset(v.location)) / r.sublane_width;
				float across_lo <- lane_center - v.vehicle_width / (2.0 * r.sublane_width);
				float across_hi <- lane_center + v.vehicle_width / (2.0 * r.sublane_width);
				int first_h <- max([0, int(floor(along_lo / (8.0 #m)))]);
				int last_h <- min([19, int(floor(along_hi / (8.0 #m)))]);
				int first_w <- max([0, int(floor(across_lo))]);
				int last_w <- min([15, int(floor(across_hi))]);
				if (first_h <= last_h and first_w <= last_w) {
					int type_channel <- v.vehicle_type = "motorcycle" ? 0 :
						(v.vehicle_type = "car" ? 1 : 2);
					float longitudinal_speed <- min([1.0, max([0.0, v.measured_speed / (50.0 #km/#h)])]);
					// Peak since the last decision avoids missing a sub-second lane
					// change between 10-second snapshots. Positive shift is left;
					// C4 follows the proposal convention: negative left, positive right.
					float transverse_speed <- max([-1.0, min([1.0,
						-v.recent_lateral_peak / (5.0 #m/#s)])]);
					loop h from: first_h to: last_h {
						float overlap_long <- max([0.0 #m, min([along_hi, (h + 1) * 8.0 #m]) -
							max([along_lo, h * 8.0 #m])]);
						loop w from: first_w to: last_w {
							float overlap_across <- max([0.0, min([across_hi, w + 1.0]) -
								max([across_lo, float(w)])]);
							float area <- overlap_long * overlap_across / (8.0 #m);
							int pixel <- (((arm * 6 + type_channel) * 20 + h) * 16 + w);
							int speed_pixel <- (arm * 20 + h) * 16 + w;
							pixels[pixel] <- min([1.0, pixels[pixel] + area]);
							pixels[(((arm * 6 + 3) * 20 + h) * 16 + w)] <-
								pixels[(((arm * 6 + 3) * 20 + h) * 16 + w)] + area * longitudinal_speed;
							pixels[(((arm * 6 + 4) * 20 + h) * 16 + w)] <-
								pixels[(((arm * 6 + 4) * 20 + h) * 16 + w)] + area * transverse_speed;
							moving_area[speed_pixel] <- moving_area[speed_pixel] + area;
						}
					}
				}
			}
		}
		// E,N,W,S are approach IDs 4..7. Pressure subtracts the queue at
		// each movement's actual exit; both approaches served by a phase add.
		spatial_pressure <- [
			incoming_movement[2] - outgoing_queue[3] + incoming_movement[6] - outgoing_queue[1],
			incoming_movement[3] - outgoing_queue[0] + incoming_movement[7] - outgoing_queue[2],
			incoming_movement[0] - outgoing_queue[2] + incoming_movement[4] - outgoing_queue[0],
			incoming_movement[1] - outgoing_queue[3] + incoming_movement[5] - outgoing_queue[1]
		];
		loop arm from: 0 to: 3 {
			loop h from: 0 to: 19 {
				loop w from: 0 to: 15 {
					int local_index <- (arm * 20 + h) * 16 + w;
					if (moving_area[local_index] > 0.0) {
						int speed_index <- (((arm * 6 + 3) * 20 + h) * 16 + w);
						int lateral_index <- (((arm * 6 + 4) * 20 + h) * 16 + w);
						pixels[speed_index] <- pixels[speed_index] / moving_area[local_index];
						pixels[lateral_index] <- pixels[lateral_index] / moving_area[local_index];
					}
				}
			}
		}
		spatial_pack <- "" + pixels;
		ask active_vehicles { recent_lateral_peak <- 0.0 #m/#s; }
		spatial_capture_cycle <- cycle;
	}

	// =====================================================
	// STATE OBSERVATION (dau vao cua DQN)
	// =====================================================
	list<road> ns_incoming_roads <- [];
	list<road> ew_incoming_roads <- [];
	int queue_ns <- 0;      // xe dang cho (speed < 1 km/h) tren duong NS vao
	int queue_ew <- 0;      // xe dang cho tren duong EW vao
	int approach_ns <- 0;   // xe dang di chuyen tren duong NS vao
	int approach_ew <- 0;   // xe dang di chuyen tren duong EW vao
	float rng_seed <- 1093.0; // Seed RNG truyen tu Python (fix: tham so 'seed'
	                          // dac biet cua server reload bi bo qua -> dung tham so thuong)

	init {
		// FIX bug seed: tham so dac biet "seed" truyen qua gama-server reload
		// khong duoc ap dung (tat ca episode trung lap). Truyen seed nhu mot
		// tham so thuong "rng_seed" roi gan vao RNG tai init, truoc khi bat
		// ky xe nao duoc sinh (sinh xe bat dau tu cycle 1).
		seed <- int(rng_seed);
		// Initialization moved to observe_state reflex to avoid nil agent errors in headless mode
	}

	// Goi tu Python qua gama-server: lenh "ask" (agent="world.0", args la JSON string):
	//   {"type":"ask","exp_id":...,"action":"set_dqn_action","agent":"world.0","args":"{\"a\": 1.0}"}
	action set_dqn_action(int a) {
		dqn_action <- a;
	}

	// =====================================================
	// REFLEX 1: Quan sat state moi cycle (den xanh + vang deu quan sat)
	// =====================================================
	reflex observe_state {
		// During Python training with the service guard disabled, queue/demand
		// snapshots are only consumed at decision ticks. Avoid scanning every
		// active vehicle on the other ~99 cycles between decisions. Keep the
		// per-cycle scan for the guarded policy and the interactive preview,
		// where phase aging and the HUD are expected to update continuously.
		bool refresh_observation <- dqn_service_guard or !python_controlled or dqn_waiting_action or
			(global_phase mod 2 = 0 and green_cycles >= min_green_cycles - 1 and
			 green_cycles mod min_green_cycles = min_green_cycles - 1);
		if (refresh_observation) {
			phase_demand <- [0,0,0,0];
			if (empty(ns_incoming_roads) and length(intersection) > 3) {
				ns_incoming_roads <- controller_roads(true,true);
				ew_incoming_roads <- controller_roads(false,true);
			}
			queue_ns <- 0;
			queue_ew <- 0;
			approach_ns <- 0;
			approach_ew <- 0;
			loop v over: active_vehicles {
				if (v.current_road != nil and v.uses_signal and distance_to(v.location,intersection_center) <= detector_distance) {
					road r <- road(v.current_road);
					bool is_ns <- ns_incoming_roads contains r;
					bool is_ew <- ew_incoming_roads contains r;
					if (is_ns or is_ew) {
						int movement <- (is_ns ? 0 : 2) + (v.maneuver = "left" ? 1 : 0);
						phase_demand[movement] <- phase_demand[movement] + 1;
						if (v.speed < 1.0 #km/#h) {
							if (is_ns) { queue_ns <- queue_ns + 1; } else { queue_ew <- queue_ew + 1; }
						} else {
							if (is_ns) { approach_ns <- approach_ns + 1; } else { approach_ew <- approach_ew + 1; }
						}
					}
				}
			}
			// Expose restrictions before action selection; Python uses the same
			// mask for exploration and Bellman bootstrap, so no action is relabelled.
			dqn_action_mask <- [1,1,1,1];
			int oldest <- -1;
			loop a from: 0 to: 3 {
				if (phase_demand[a] = 0 or (global_phase = a * 2 and !dqn_all_red)) {
					phase_wait_cycles[a] <- 0;
				} else { phase_wait_cycles[a] <- phase_wait_cycles[a] + 1; }
				if (phase_wait_cycles[a] >= max_phase_wait_cycles and
					(oldest < 0 or phase_wait_cycles[a] > phase_wait_cycles[oldest])) { oldest <- a; }
			}
			if (dqn_service_guard) {
				if (oldest >= 0) {
					dqn_action_mask <- [0,0,0,0]; dqn_action_mask[oldest] <- 1;
				} else if (global_phase mod 2 = 0 and green_cycles >= max_green_cycles - 1) {
					int current <- int(global_phase / 2);
					if (sum(phase_demand) > phase_demand[current]) { dqn_action_mask[current] <- 0; }
				}
			}
		}
	}

	// =====================================================
	// REFLEX 2: DQN CONTROLLER
	// =====================================================
	reflex dqn_controller {
		phase_timer <- phase_timer + step;
		if (dqn_all_red) {
			all_red_cycles <- all_red_cycles + 1;
			// Keep all approaches red until every admitted vehicle clears the
			// conflict area. A fixed 25-cycle cap could open a perpendicular
			// phase while the previous movement was still inside the junction.
			if (all_red_cycles >= 20 and signal_junction_clear()) {
				dqn_all_red <- false;
				all_red_cycles <- 0;
				global_phase <- pending_green_phase;
				green_cycles <- 0;
				phase_timer <- 0.0;
			}
		} else if (global_phase mod 2 = 1) {
			yellow_cycles <- yellow_cycles + 1;
			if (yellow_cycles >= yellow_len_cycles) {
				yellow_cycles <- 0;
				dqn_all_red <- true;
				// Keep the yellow phase index; elapsed time makes all shared GAMA
				// signal and vehicle rules show red during clearance.
				phase_timer <- signal_yellow_seconds;
			}
		} else {
			green_cycles <- green_cycles + 1;
			if (dqn_waiting_action and dqn_action >= 0) {
				current_action <- dqn_action;
				decision_count <- decision_count + 1;
				dqn_waiting_action <- false;
				dqn_action <- -1;
				pending_green_phase <- current_action * 2;
				if (pending_green_phase != global_phase) {
					global_phase <- global_phase + 1; // current green -> its yellow
					yellow_cycles <- 0;
					phase_timer <- 0.0;
				}
			} else if (!dqn_waiting_action and green_cycles >= min_green_cycles and
				green_cycles mod min_green_cycles = 0) {
				if (python_controlled) {
					dqn_waiting_action <- true;
				} else {
						// Preview serves all four movements and obeys the service guard.
						int selected <- -1;
						float best_score <- -1.0;
						loop a from: 0 to: 3 {
							float score <- phase_demand[a] + phase_wait_cycles[a] / 100.0;
							if (dqn_action_mask[a] = 1 and score > best_score) {
								selected <- a; best_score <- score;
							}
						}
						pending_green_phase <- selected * 2;
					current_action <- int(pending_green_phase / 2);
					if (pending_green_phase != global_phase) {
						global_phase <- global_phase + 1;
						yellow_cycles <- 0;
						phase_timer <- 0.0;
					}
				}
			}
		}
		if (!fast_training) {
			if (dqn_all_red) {
				signal_display_seconds <- int(max([0, 20 - all_red_cycles]) * step / (1 #s) + 0.999);
			} else if (global_phase mod 2 = 1) {
				signal_display_seconds <- int((yellow_len_cycles - yellow_cycles) * step / (1 #s) + 0.999);
			} else {
				signal_display_seconds <- int(max([0, min_green_cycles - (green_cycles mod min_green_cycles)]) * step / (1 #s) + 0.999);
			}
			signal_countdown <- "" + signal_display_seconds + "s";
		}
	}

	// =====================================================
	// REFLEX 3: Scheduler - bao Python can step bao nhieu cycle
	// de den dung diem quyet dinh ke tiep
	// =====================================================
	reflex update_schedule {
		if (dqn_waiting_action) {
			next_decision_in_cycles <- 1;
		} else if (dqn_all_red) {
			next_decision_in_cycles <- all_red_cycles >= 20 ? 1 :
				20 - all_red_cycles + min_green_cycles;
		} else if (global_phase mod 2 = 1) {
			next_decision_in_cycles <- yellow_len_cycles - yellow_cycles + 20 + min_green_cycles;
		} else {
			next_decision_in_cycles <- min_green_cycles - green_cycles mod min_green_cycles;
		}
	}

}

// Experiment chinh cho Python/gama-server.
experiment dqn_server_experiment type: gui autorun: false {
	parameter "Phase service guard" var: dqn_service_guard;
	parameter "Surrogate safety metrics" var: enable_surrogate_safety init: false;
	parameter "Demand CSV" var: demand_file_path;
	parameter "NS spawn interval" var: spawn_rate_NS;
	parameter "EW spawn interval" var: spawn_rate_EW;
	parameter "Python Controlled" var: python_controlled init: true;
	parameter "Seed" var: seed init: 1093.0;
	parameter "DQN_RNG_Seed" var: rng_seed init: 1093.0;
}

experiment dqn_gui_server_experiment type: gui {
	// Opening this experiment manually must remain live without a Python
	// client. External training/evaluation uses dqn_server_experiment above.
	parameter "Python Controlled" var: python_controlled init: false;
	parameter "Seed" var: seed init: 1093.0;
	float minimum_cycle_duration <- 0.033#s; // About 3 model seconds per wall second (step = 0.1 s)
	output synchronized: true {
		monitor "Xe hoàn thành mặt đất" value: total_vehicles_passed;
		monitor "Hàng đợi mặt đất" value: current_queue_length;
		monitor "Pha đèn" value: phase_names[global_phase];
		monitor "Đếm ngược pha" value: signal_countdown;
		monitor "Chờ trung bình (s)" value: average_waiting_time;
		display doan_3d type: opengl background: rgb(8,12,22) {
			camera 'default' location: {820,1210,220} target: {1000,1000,16};
			species doan_ground aspect: base;
			species traffic_signal_post aspect: doan_3d;
			species base_vehicle_species aspect: doan_3d;
		}

		display intersection_view background: rgb(18, 22, 28) {
			species doan_ground aspect: base;
			species road aspect: base;
			species traffic_signal_post aspect: base;
			species conflict_zone aspect: base;
			species base_vehicle_species aspect: base;
			graphics "HUD" {
				rgb phase_c <- dqn_all_red ? #red : (global_phase mod 2 = 0 ? #lime : #yellow);
				draw rectangle(310, 115) at: {165, 62} color: rgb(15, 23, 42, 215) border: rgb(71, 85, 105);
				draw "TRAFFIC-LIGHT ABM | DQN PYTORCH" at: {20, 22} color: #lime font: font("Helvetica", 13, #bold);
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " | Green: " + green_cycles + "c" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);
				draw "Selected phase: " + current_action + " | Next in: " + next_decision_in_cycles + "c" at: {20, 68} color: #orange font: font("Helvetica", 12, #plain);
				draw "Queue NS: " + queue_ns + " | EW: " + queue_ew at: {20, 88} color: rgb(56, 189, 248) font: font("Helvetica", 12, #plain);
				draw "Passed: " + total_vehicles_passed + " xe | Queue: " + current_queue_length + " | Avg Wait: " + (round(average_waiting_time*10.0)/10.0) + "s" at: {20, 108} color: rgb(226, 232, 240) font: font("Helvetica", 12, #plain);
			}
		}
	}
}

experiment dqn_gui_preview type: gui {
	parameter "Python Controlled" var: python_controlled init: false;
	float minimum_cycle_duration <- 0.033#s; // About 3 model seconds per wall second (step = 0.1 s)
	output synchronized: true {
		monitor "Xe hoàn thành mặt đất" value: total_vehicles_passed;
		monitor "Hàng đợi mặt đất" value: current_queue_length;
		monitor "Pha đèn" value: phase_names[global_phase];
		monitor "Đếm ngược pha" value: signal_countdown;
		monitor "Chờ trung bình (s)" value: average_waiting_time;
		display doan_3d type: opengl background: rgb(8,12,22) {
			camera 'default' location: {820,1210,220} target: {1000,1000,16};
			species doan_ground aspect: base;
			species traffic_signal_post aspect: doan_3d;
			species base_vehicle_species aspect: doan_3d;
		}

		display intersection_view background: rgb(18, 22, 28) {
			species doan_ground aspect: base;
			species road aspect: base;
			species traffic_signal_post aspect: base;
			species conflict_zone aspect: base;
			species base_vehicle_species aspect: base;
			graphics "HUD" {
				rgb phase_c <- dqn_all_red ? #red : (global_phase mod 2 = 0 ? #lime : #yellow);
				draw rectangle(310, 125) at: {165, 68} color: rgb(15, 23, 42, 215) border: rgb(71, 85, 105);
				draw "TRAFFIC-LIGHT ABM | DQN PYTORCH" at: {20, 22} color: #lime font: font("Helvetica", 13, #bold);
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " | Green: " + green_cycles + "c" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);
				draw "Selected phase: " + current_action + " | Dec: " + decision_count at: {20, 68} color: #orange font: font("Helvetica", 12, #plain);
				draw "Passed: " + total_vehicles_passed + " xe | Queue: " + current_queue_length + " | Avg Wait: " + (round(average_waiting_time*10.0)/10.0) + "s" at: {20, 88} color: rgb(226, 232, 240) font: font("Helvetica", 12, #plain);
				draw "TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | TTC<1s: " + severe_conflicts_count + " | HardBrakes: " + hard_braking_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s" at: {20, 108} color: rgb(248, 113, 113) font: font("Helvetica", 11, #plain);
			}
		}
		display performance_charts {
			chart "Performance Metrics (DQN-Python)" type: series {
				data "Queue Length" value: current_queue_length color: #crimson;
				data "Avg Waiting Time (s)" value: average_waiting_time color: #darkorange;
				data "Avg Speed (km/h)" value: average_speed color: #dodgerblue;
			}
		}
	}
}
