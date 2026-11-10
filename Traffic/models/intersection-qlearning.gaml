/**
* Name: intersection_qlearning
* Description: Q-Learning reinforcement learning traffic signal controller.
*/

model intersection_qlearning

import "../includes/collision_avoidance.gaml"

global {
	// =====================================================
	// Q-LEARNING REINFORCEMENT LEARNING CONTROLLER
	// =====================================================
	// 0/4: through green, 1/5: through yellow,
	// 2/6: protected left green, 3/7: left yellow.
	// Timing & Safety constraints
	float min_green_time <- 10#s;   // Thời gian xanh tối thiểu (tránh đổi liên tục)
	float max_green_time <- 60#s;   // Thời gian xanh tối đa (tránh starvation)
	float yellow_time <- 3#s;       // Thời gian đèn vàng an toàn
	float left_green_time <- 12#s;  // Guaranteed protected left-turn service
	float decision_interval <- 5#s; // AI quyết định mỗi 5 giây
	
	// Q-Learning Hyperparameters
	float alpha_lr <- 0.1;          // Tốc độ học (Learning rate)
	float gamma_discount <- 0.97;   // Discount per 5 simulation seconds
	float epsilon <- 0.5;           // Xác suất kháĐm phá (Exploration rate)
	float epsilon_decay <- 0.995;   // Giảm dần khám phá
	float epsilon_min <- 0.05;      // Mức khám phá tối thiểu
	
	// Q-Table: Map[State, [Q(Extend), Q(Switch)]]
	map<string, list<float>> q_table <- [];
	string prev_state <- "";
	int prev_action <- -1;
	int current_action <- 0;
	float last_reward <- 0.0;
	int q_updates_count <- 0;
	float prev_queue_integral <- 0.0;
	float prev_decision_time <- 0.0;
	
	// Road collections for queue observation
	list<road> ns_incoming_roads <- [];
	list<road> ew_incoming_roads <- [];
	int queue_ns <- 0;
	int queue_ew <- 0;

	init {
		// FIX (headless): khong build danh sach duong o day vi intersection = nil
		// khi init cua model chay (mang duong tao trong init cua include) ->
		// danh sach rong -> queue_ns/queue_ew luon = 0 -> khong gian state suy bien
		// (chi hoc trong 1-2 state thay vi 18). Khoi tao muon nhu model DQN.

		// Khởi tạo Q-Table với các giá trị mặc định cho 18 states (2 pha xanh x 3 NS x 3 EW)
		loop p over: [0, 4] {
			loop n_lvl over: [0, 1, 2] {
				loop e_lvl over: [0, 1, 2] {
					string st <- "" + p + "_" + n_lvl + "_" + e_lvl;
					q_table[st] <- [0.0, 0.0];
				}
			}
		}
	}

	// Khoi tao muon danh sach duong vao khi mang duong da san sang
	reflex init_road_lists when: (empty(ns_incoming_roads) and length(intersection) > 3) {
		ns_incoming_roads <- controller_roads(true,true);
		ew_incoming_roads <- controller_roads(false,true);
	}
	
	// =====================================================
	// Q-LEARNING CONTROLLER REFLEX
	// =====================================================
	reflex qlearning_controller {
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
		// Q-learning controls the through greens; left greens run for 12s.
		else {
			bool check_now <- (phase_timer >= min_green_time) and 
			                  (round(phase_timer * 10) mod int(decision_interval * 10) = 0);
			bool force_switch <- phase_timer >= max_green_time;
			
			if (check_now or force_switch) {
				// 1. Quan sát trạng thái môi trường
				queue_ns <- 0;
				queue_ew <- 0;
				loop v over: active_vehicles {
					if (v.current_road != nil and v.uses_signal and distance_to(v.location,intersection_center) <= detector_distance and v.speed < 1.0 #km/#h) {
						road r <- road(v.current_road);
						if (ns_incoming_roads contains r) {
							queue_ns <- queue_ns + 1;
						} else if (ew_incoming_roads contains r) {
							queue_ew <- queue_ew + 1;
						}
					}
				}
				
				int ns_lvl <- 0;
				if (queue_ns >= 15) { ns_lvl <- 2; }
				else if (queue_ns >= 5) { ns_lvl <- 1; }
				
				int ew_lvl <- 0;
				if (queue_ew >= 15) { ew_lvl <- 2; }
				else if (queue_ew >= 5) { ew_lvl <- 1; }
				
				// Green age distinguishes decisions with different legal durations.
				string curr_state <- "" + global_phase + "_" + ns_lvl + "_" + ew_lvl + "_" + int(phase_timer / decision_interval);
				
				if not (q_table contains_key curr_state) {
					q_table[curr_state] <- [0.0, 0.0];
				}
				
				// 2. Tính Reward từ bước trước & Cập nhật Q-Table (Bellman Equation)
				if (prev_state != "" and prev_action >= 0) {
					// Reward âm tỷ lệ thuận với hàng đợi (khuyến khích giải tỏa nhanh)
					float reward <- -(queue_time_integral - prev_queue_integral) / 50.0;
					last_reward <- reward;
					
					list<float> curr_q <- q_table[curr_state];
					float max_next_q <- force_switch ? curr_q[1] : max(curr_q);
					float discount <- gamma_discount ^ ((time - prev_decision_time) / (5 #s));
					
					list<float> prev_q <- q_table[prev_state];
					float old_val <- prev_q[prev_action];
					float new_val <- (1.0 - alpha_lr) * old_val + alpha_lr * (reward + discount * max_next_q);
					prev_q[prev_action] <- new_val;
					q_table[prev_state] <- prev_q;
					q_updates_count <- q_updates_count + 1;
				}
				
				// 3. Chọn Action theo Epsilon-Greedy
				int chosen_action <- 0; // 0: Extend (Giữ xanh), 1: Switch (Đổi đèn)
				if (force_switch) {
					chosen_action <- 1; // Bắt buộc đổi để tránh kẹt xe kéo dài
				} else {
					if (rnd(1.0) < epsilon) {
						chosen_action <- rnd(1); // Khám phá (Explore)
					} else {
						// Khai thác (Exploit)
						list<float> q_vals <- q_table[curr_state];
						if (q_vals[0] > q_vals[1]) {
							chosen_action <- 0;
						} else if (q_vals[1] > q_vals[0]) {
							chosen_action <- 1;
						} else {
							chosen_action <- rnd(1);
						}
					}
				}
				
				// Giảm dần tỷ lệ khám phá (Epsilon Decay)
				if (epsilon > epsilon_min) {
					epsilon <- max([epsilon_min, epsilon * epsilon_decay]);
				}
				
				// Lưu lại state và action để học ở bước kế tiếp
				prev_state <- curr_state;
				prev_action <- chosen_action;
				prev_queue_integral <- queue_time_integral;
				prev_decision_time <- time;
				current_action <- chosen_action;
				
				// 4. Thực thi Action
				if (chosen_action = 1) {
					phase_timer <- 0.0;
					global_phase <- global_phase + 1;
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
	// Only verbose logging remains here (specific to Q-Learning)
	reflex verbose_log when: (cycle mod 50 = 0 and cycle > 0) {
		write ">> Cycle: " + cycle + " | Phase: " + phase_names[global_phase] + " (" + (round(phase_timer*10.0)/10.0) + "s) | Action: " + (current_action = 0 ? "EXTEND" : "SWITCH") + " | Eps: " + (round(epsilon*100.0)/100.0) + " | Passed: " + total_vehicles_passed + " | Queue: " + current_queue_length + " | TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | Severe TTC: " + severe_conflicts_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s";
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
				draw "TRAFFIC-LIGHT ABM | Q-LEARNING" at: {20, 22} color: #cyan font: font("Helvetica", 13, #bold);
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " (" + int(phase_timer) + "s)" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);
				draw "Action: " + (current_action = 0 ? "EXTEND" : "SWITCH") + " | Eps: " + (round(epsilon*100.0)/100.0) + " | Q-Up: " + q_updates_count at: {20, 68} color: #cyan font: font("Helvetica", 12, #plain);
				draw "Passed: " + total_vehicles_passed + " xe | Queue: " + current_queue_length + " | Avg Wait: " + (round(average_waiting_time*10.0)/10.0) + "s" at: {20, 88} color: rgb(226, 232, 240) font: font("Helvetica", 12, #plain);
				draw "TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | TTC<1s: " + severe_conflicts_count + " | HardBrakes: " + hard_braking_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s" at: {20, 108} color: rgb(248, 113, 113) font: font("Helvetica", 11, #plain);
			}
		}
		display performance_charts {
			chart "Performance Metrics (Q-Learning)" type: series {
				data "Queue Length" value: current_queue_length color: #crimson;
				data "Avg Waiting Time (s)" value: average_waiting_time color: #darkorange;
				data "Avg Speed (km/h)" value: average_speed color: #dodgerblue;
				data "Epsilon" value: epsilon * 10 color: #violet;
			}
		}
	}
}

experiment my_batch_experiment type: batch until: (cycle >= doan_horizon_cycles) {
	// Seed giam de 4 thuat toan (Fixed-Time/MaxPressure/Q-Learning/DQN) chay
	// cung luong yeu cau xe -> so sanh cong bang & kha lap lai
	parameter "Seed" var: seed init: 5001.0;
}
