/**
* Name: intersection_maxpressure
* Description: 4-way traffic intersection simulation controlled by MaxPressure adaptive traffic signal algorithm.
*/

model intersection_maxpressure

import "../includes/collision_avoidance.gaml"

global {
	// MAXPRESSURE CONTROLLER PARAMETERS
	// 0/4: through green, 1/5: through yellow,
	// 2/6: protected left green, 3/7: left yellow.
	// Timing constraints for stability & safety
	float min_green_time <- 10#s;   // Thời gian xanh tối thiểu để tránh đổi pha liên tục
	float max_green_time <- 60#s;   // Thời gian xanh tối đa để tránh bỏ đói (starvation)
	float yellow_time <- 3#s;       // Thời gian đèn vàng an toàn
	float left_green_time <- 12#s;  // Guaranteed protected left-turn service
	float check_interval <- 5#s;    // Chu kỳ đánh giá áp lực (mỗi 5 giây)
	
	// Road collections for fast pressure calculation
	list<road> ns_incoming_roads <- [];
	list<road> ns_outgoing_roads <- [];
	list<road> ew_incoming_roads <- [];
	list<road> ew_outgoing_roads <- [];
	
	float pressure_ns <- 0.0;
	float pressure_ew <- 0.0;
	
	init {
		// FIX (headless): mang duong chua duoc tao khi init cua model chay
		// -> intersection(1) = nil -> danh sach duong rong -> ap suat luon = 0
		// -> controller "mu", chi con force-switch theo max_green_time (60s).
		// Khoi tao muon duoc chuyen sang reflex init_road_lists ben duoi
		// (cach lam giong intersection-dqn-python.gaml).
	}

	// Khoi tao muon danh sach duong vao/ra khi mang duong da san sang
	reflex init_road_lists when: (empty(ns_incoming_roads) and length(intersection) > 3) {
		ns_incoming_roads <- controller_roads(true,true);
		ew_incoming_roads <- controller_roads(false,true);

		ns_outgoing_roads <- controller_roads(true,false);
		ew_outgoing_roads <- controller_roads(false,false);
	}
	
	reflex maxpressure_controller {
		phase_timer <- (round(phase_timer / step) + 1) * step;
		
		// Yellow phases clear the intersection before the next movement opens.
		if (global_phase mod 2 = 1) {
			if (phase_timer >= yellow_time + 2 #s and signal_junction_clear()) {
				phase_timer <- 0.0;
				global_phase <- (global_phase + 1) mod 8;
			}
		} else if (global_phase = 2 or global_phase = 6) {
			if (phase_timer >= left_green_time) {
				phase_timer <- 0.0;
				global_phase <- global_phase + 1;
			}
		}
		// MaxPressure chooses the length of each through green.
		else {
			bool check_now <- (phase_timer >= min_green_time) and 
			                  (round(phase_timer * 10) mod int(check_interval * 10) = 0);
			bool force_switch <- phase_timer >= max_green_time;
			
			if (check_now or force_switch) {
				int in_ns <- 0; int out_ns <- 0;
				int in_ew <- 0; int out_ew <- 0;
				int waiting_left <- 0;
				
				loop v over: active_vehicles {
					if (v.current_road != nil and distance_to(v.location,intersection_center) <= detector_distance) {
						road r <- road(v.current_road);
						if (r.is_outer_road and r.is_incoming and v.maneuver = "left" and v.speed < 1 #km/#h) {
							waiting_left <- waiting_left + 1;
						}
						if ((ns_incoming_roads contains r) and v.maneuver = "straight") { in_ns <- in_ns + 1; }
						else if ((ns_outgoing_roads contains r) and v.lowest_lane >= 4 and v.lowest_lane < 12) { out_ns <- out_ns + 1; }
						else if ((ew_incoming_roads contains r) and v.maneuver = "straight") { in_ew <- in_ew + 1; }
						else if ((ew_outgoing_roads contains r) and v.lowest_lane >= 4 and v.lowest_lane < 12) { out_ew <- out_ew + 1; }
					}
				}
				
				// Áp lực mạng lưới = Số xe vào - Số xe ra
				pressure_ns <- float(in_ns - out_ns);
				pressure_ew <- float(in_ew - out_ew);
				
				if (global_phase = 0) { // NS through
					if (force_switch or (pressure_ew > pressure_ns + 2.0) or
						(in_ns = 0 and (in_ew > 0 or waiting_left > 0))) {
						phase_timer <- 0.0;
						global_phase <- 1;
					}
				} else if (global_phase = 4) { // EW through
					if (force_switch or (pressure_ns > pressure_ew + 2.0) or
						(in_ew = 0 and (in_ns > 0 or waiting_left > 0))) {
						phase_timer <- 0.0;
						global_phase <- 5;
					}
				}
			}
		}
		if (global_phase mod 2 = 1) {
			float remaining <- max([0.0, yellow_time - phase_timer]);
			signal_display_seconds <- int(remaining / (1 #s) + 0.999);
			signal_countdown <- "" + int(remaining / (1 #s) + 0.999) + "s";
		} else if (global_phase = 2 or global_phase = 6) {
			signal_display_seconds <- int(max([0.0, left_green_time - phase_timer]) / (1 #s) + 0.999);
			signal_countdown <- "" + int(max([0.0, left_green_time - phase_timer]) / (1 #s) + 0.999) + "s";
		} else {
			signal_display_seconds <- int(max([0.0, max_green_time - phase_timer]) / (1 #s) + 0.999);
			signal_countdown <- "≤" + int(max([0.0, max_green_time - phase_timer]) / (1 #s) + 0.999) + "s";
		}
	}
	
	// A1+A4: Spawn and metrics are now centralized in collision_avoidance.gaml
	// Only verbose logging remains here (specific to MaxPressure)
	reflex verbose_log when: (cycle mod 50 = 0 and cycle > 0) {
		write ">> Cycle: " + cycle + " | Phase: " + phase_names[global_phase] + " (" + round(phase_timer*10.0)/10.0 + "s) | Passed: " + total_vehicles_passed + " | Queue: " + current_queue_length + " | Pressure NS: " + round(pressure_ns*10.0)/10.0 + " EW: " + round(pressure_ew*10.0)/10.0 + " | TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | Severe TTC: " + severe_conflicts_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s";
	}
}

experiment my_experiment type: gui {
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
				draw "TRAFFIC-LIGHT ABM | MAXPRESSURE" at: {20, 22} color: #lime font: font("Helvetica", 13, #bold);
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " (" + int(phase_timer) + "s)" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);
				draw "Pressure NS: " + (round(pressure_ns*10.0)/10.0) + " | EW: " + (round(pressure_ew*10.0)/10.0) at: {20, 68} color: rgb(56, 189, 248) font: font("Helvetica", 12, #plain);
				draw "Passed: " + total_vehicles_passed + " xe | Queue: " + current_queue_length + " | Avg Wait: " + (round(average_waiting_time*10.0)/10.0) + "s" at: {20, 88} color: rgb(226, 232, 240) font: font("Helvetica", 12, #plain);
				draw "TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | TTC<1s: " + severe_conflicts_count + " | HardBrakes: " + hard_braking_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s" at: {20, 108} color: rgb(248, 113, 113) font: font("Helvetica", 11, #plain);
			}
		}
		display performance_charts {
			chart "Performance Metrics (MaxPressure)" type: series {
				data "Queue Length (vehicles)" value: current_queue_length color: #crimson;
				data "Avg Waiting Time (s)" value: average_waiting_time color: #darkorange;
				data "Avg Speed (km/h)" value: average_speed color: #dodgerblue;
			}
		}
	}
}

experiment my_batch_experiment type: batch until: (cycle >= doan_horizon_cycles) {
	// Seed giam de 4 thuat toan (Fixed-Time/MaxPressure/Q-Learning/DQN) chay
	// cung luong yeu cau xe -> so sanh cong bang & kha lap lai
	parameter "Seed" var: seed init: 5001.0;
}

experiment python_parity type: gui autorun: false {
	parameter "Demand CSV" var: demand_file_path;
	parameter "Traffic seed" var: traffic_seed;
	parameter "Fast simulation" var: fast_training init: true;
}
