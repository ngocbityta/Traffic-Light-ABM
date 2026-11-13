/**
* Name: intersection_dqn_python
* Description: Deep Q-Learning (DQN) traffic signal controller driven by an EXTERNAL
*   Python agent (PyTorch) through gama-server (WebSocket, JSON protocol).
*
*   Architecture (GAMA + Python/PyTorch):
*     - GAMA (this file, headless server): giu toan bo logic thoi gian & an toan
*       (min through green 10s, max 60s, protected left 12s, yellow 3s,
*       decision moi 5s) nhu mo hinh
*       Q-Learning tabular de dam bao so sanh cong bang.
*     - Python (train/logic/): DQN agent nhan state tu GAMA, chon action
*       (0 = EXTEND, 1 = SWITCH) roi gui nguoc ve GAMA.
*
*   Handshake protocol (Python <- -> GAMA), moi diem quyet dinh:
*     1. Python GET  state_pack        (tai cycle co tick, dqn_waiting_action = true)
*     2. Python SET  dqn_action
*     3. Python step(1)                (controller ap dung action)
*     4. Python GET  next_decision_in_cycles
*     5. Python step(next_decision_in_cycles)  -> ve lai dung tick ke tiep
*
*   Counter nguyen (green_cycles / yellow_cycles) duoc dung thay float phase_timer
*   de tick quyet dinh hoan toan deterministic (tranh loi lam tron float cua 0.1s).
*   Timing van tuong duong: tick tai 10s, 15s, ..., 55s; force switch tai 60s.
*/

model intersection_dqn_legacy

import "../includes/collision_avoidance.gaml"

global {

	// =====================================================
	// TIMING & SAFETY CONSTRAINTS (giong intersection-qlearning.gaml)
	// =====================================================
	// 0/4: through green, 1/5: through yellow,
	// 2/6: protected left green, 3/7: left yellow.
	int green_cycles <- 0;            // so cycle da troi trong pha xanh hien tai
	int yellow_cycles <- 0;           // so cycle da troi trong pha vang hien tai
	int left_cycles <- 0;

	int min_green_cycles <- 100;      // min_green_time 10s / step 0.1s
	int max_green_cycles <- 600;      // max_green_time 60s / step 0.1s
	int yellow_len_cycles <- 30;      // yellow_time 3s / step 0.1s
		int clearance_len_cycles <- 20;   // all-red minimum, followed by geometry clearance
	int left_len_cycles <- 120;       // protected left green 12s
	int decision_cycles <- 50;        // decision_interval 5s / step 0.1s

	// =====================================================
	// PYTHON DQN AGENT INTERFACE
	// =====================================================
	int dqn_action <- -1;             // Python set truoc handshake: 0=EXTEND, 1=SWITCH
	bool dqn_waiting_action <- false; // true = controller dang cho action tu Python
	int next_decision_in_cycles <- 100; // Python step den tick ke tiep
	int current_action <- 0;          // action cuoi cung da ap dung
	int decision_count <- 0;          // tong so quyet dinh AI da thuc hien
	bool python_controlled <- true;   // false = chay heuristic fallback (demo khong can Python)

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
					if (v.speed < 1.0 #km/#h) {
						if (is_ns) { queue_ns <- queue_ns + 1; } else { queue_ew <- queue_ew + 1; }
					} else {
						if (is_ns) { approach_ns <- approach_ns + 1; } else { approach_ew <- approach_ew + 1; }
					}
				}
			}
		}
	}

	// =====================================================
	// REFLEX 2: DQN CONTROLLER
	// =====================================================
	reflex dqn_controller {
		phase_timer <- phase_timer + step;

			// The odd phase index remains during all-red; lamps use phase_timer.
		if (global_phase mod 2 = 1) {
			yellow_cycles <- yellow_cycles + 1;
				if (yellow_cycles >= yellow_len_cycles + clearance_len_cycles and signal_junction_clear()) {
				yellow_cycles <- 0;
				phase_timer <- 0.0;
				global_phase <- (global_phase + 1) mod 8;
				green_cycles <- 0;
				left_cycles <- 0;
			}
		}
		else if (global_phase = 2 or global_phase = 6) {
			left_cycles <- left_cycles + 1;
			if (left_cycles >= left_len_cycles) {
				left_cycles <- 0;
				phase_timer <- 0.0;
				yellow_cycles <- 0;
				global_phase <- global_phase + 1;
			}
		}
		// DQN chooses the duration of through green only.
		else {
			green_cycles <- green_cycles + 1;

			if (dqn_waiting_action) {
				// Handshake cycle: ap dung action Python vua gui
				if (dqn_action >= 0) {
					current_action <- dqn_action;
					decision_count <- decision_count + 1;
					dqn_waiting_action <- false;
					dqn_action <- -1;
					if (current_action = 1) {
						phase_timer <- 0.0;
						yellow_cycles <- 0;
						global_phase <- global_phase + 1; // GREEN -> YELLOW
					}
				}
					// Python không phản hồi: vẫn cưỡng chế đổi pha ở giới hạn 60 s.
						if (global_phase mod 2 = 0 and dqn_action < 0 and green_cycles >= max_green_cycles) {
						dqn_waiting_action <- false;
						phase_timer <- 0.0;
						yellow_cycles <- 0;
						global_phase <- global_phase + 1;
						green_cycles <- 0;
					}
			}
			else {
				// Tick dung tren luoi nguyen: 100, 150, 200, ... cycles (10s, 15s, 20s, ...)
				bool tick_now <- (green_cycles >= min_green_cycles) and
				                 ((green_cycles - min_green_cycles) mod decision_cycles = 0);
				bool force_switch <- green_cycles >= max_green_cycles;

				if (tick_now or force_switch) {
					if (force_switch) {
						// An toan: bat buoc doi pha o 60s, khong can hoi AI (nhu Q-learning)
						phase_timer <- 0.0;
						yellow_cycles <- 0;
						global_phase <- global_phase + 1;
						green_cycles <- 0;
					} else if (python_controlled) {
						dqn_waiting_action <- true;   // hoi Python
					} else {
						// Fallback heuristic kieu MaxPressure (demo GUI khong can Python)
						bool want_switch <- false;
						if (global_phase = 0 and queue_ew > queue_ns + 2) { want_switch <- true; }
						if (global_phase = 4 and queue_ns > queue_ew + 2) { want_switch <- true; }
						if (want_switch) {
							phase_timer <- 0.0;
							yellow_cycles <- 0;
							global_phase <- global_phase + 1;
							green_cycles <- 0;
						}
					}
				}
			}
		}
		if (global_phase mod 2 = 1) {
			int remaining <- max([0, yellow_len_cycles - yellow_cycles]);
			signal_display_seconds <- int(remaining * step / (1 #s) + 0.999);
			signal_countdown <- "" + int(remaining * step / (1 #s) + 0.999) + "s";
		} else if (global_phase = 2 or global_phase = 6) {
			signal_display_seconds <- int(max([0, left_len_cycles - left_cycles]) * step / (1 #s) + 0.999);
			signal_countdown <- "" + int(max([0, left_len_cycles - left_cycles]) * step / (1 #s) + 0.999) + "s";
		} else {
			signal_display_seconds <- int(max([0, max_green_cycles - green_cycles]) * step / (1 #s) + 0.999);
			signal_countdown <- "≤" + int(max([0, max_green_cycles - green_cycles]) * step / (1 #s) + 0.999) + "s";
		}
	}

	// =====================================================
	// REFLEX 3: Scheduler - bao Python can step bao nhieu cycle
	// de den dung diem quyet dinh ke tiep
	// =====================================================
	reflex update_schedule {
		if (dqn_waiting_action) {
			next_decision_in_cycles <- 1;   // handshake cycle
		}
		else if (global_phase mod 2 = 1) {
				int rem_yellow <- max([0, yellow_len_cycles + clearance_len_cycles - yellow_cycles]);
			// Through yellow is followed by protected left green and left yellow.
				next_decision_in_cycles <- rem_yellow = 0 ? 1 : rem_yellow +
				((global_phase = 1 or global_phase = 5) ?
						left_len_cycles + yellow_len_cycles + clearance_len_cycles : 0) + min_green_cycles;
		}
		else if (global_phase = 2 or global_phase = 6) {
			next_decision_in_cycles <- max([0, left_len_cycles - left_cycles]) +
					yellow_len_cycles + clearance_len_cycles + min_green_cycles;
		}
		else {
			// Pha xanh: tick ke tiep tren luoi 50 cycles, tinh bang so nguyen
			int next_tick;
			if (green_cycles < min_green_cycles) {
				next_tick <- min_green_cycles;
			} else {
				next_tick <- min_green_cycles + decision_cycles * (floor((green_cycles - min_green_cycles) / decision_cycles) + 1);
			}
			if (next_tick < max_green_cycles) {
				next_decision_in_cycles <- next_tick - green_cycles;
			} else {
				// Force switch tai 60s -> vang 30 + min green 100 cycles truoc tick ke tiep
				int rem_green <- max([0, max_green_cycles - green_cycles]);
					next_decision_in_cycles <- rem_green + yellow_len_cycles + clearance_len_cycles +
						left_len_cycles + yellow_len_cycles + clearance_len_cycles + min_green_cycles;
			}
		}
	}

	// A1+A4: Spawn and metrics are now centralized in collision_avoidance.gaml
	// Only verbose logging remains here (specific to DQN)
	reflex verbose_log when: (cycle mod 50 = 0 and cycle > 0) {
		write ">> Cycle: " + cycle + " | Phase: " + phase_names[global_phase] + " (" + (round(phase_timer*10.0)/10.0) + "s) | Action: " + (current_action = 0 ? "EXTEND" : "SWITCH") + " | Passed: " + total_vehicles_passed + " | Queue: " + current_queue_length + " | TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | Severe TTC: " + severe_conflicts_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s";
	}
}

// Experiment chinh cho Python/gama-server.
experiment dqn_server_experiment type: gui autorun: false {
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
				rgb phase_c <- global_phase mod 2 = 0 ? #lime : #yellow;
				draw rectangle(310, 115) at: {165, 62} color: rgb(15, 23, 42, 215) border: rgb(71, 85, 105);
				draw "TRAFFIC-LIGHT ABM | DQN PYTORCH" at: {20, 22} color: #lime font: font("Helvetica", 13, #bold);
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " | Green: " + green_cycles + "c" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);
				draw "Action: " + (current_action = 0 ? "EXTEND (0)" : "SWITCH (1)") + " | Next in: " + next_decision_in_cycles + "c" at: {20, 68} color: #orange font: font("Helvetica", 12, #plain);
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
				rgb phase_c <- global_phase mod 2 = 0 ? #lime : #yellow;
				draw rectangle(310, 125) at: {165, 68} color: rgb(15, 23, 42, 215) border: rgb(71, 85, 105);
				draw "TRAFFIC-LIGHT ABM | DQN PYTORCH" at: {20, 22} color: #lime font: font("Helvetica", 13, #bold);
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " | Green: " + green_cycles + "c / " + max_green_cycles + "c" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);
				draw "Action: " + (current_action = 0 ? "EXTEND (0)" : "SWITCH (1)") + " | Dec: " + decision_count at: {20, 68} color: #orange font: font("Helvetica", 12, #plain);
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
