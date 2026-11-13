/** GUI demo of the trained 11-input DQN from fair_benchmark.py.
 * Start python3 scripts/demo_dqn.py before running my_experiment in GAMA Studio.
 */
model intersection_dqn_demo

import "intersection-benchmark.gaml"

global {
	bool demo_file_bridge <- false;
	bool demo_request_open <- false;
	int demo_request_serial <- 0;
	int demo_actions_received <- 0;
	string demo_status <- "Chờ Python";
	string demo_bridge_dir <- "../../../../train/results/demo_bridge/";

	// The benchmark model already measures the eight movement counts used by
	// the trained network. Only the GUI experiment enables this file bridge.
	reflex demo_bridge when: demo_file_bridge {
		if (dqn_waiting_action) {
			if (!demo_request_open) {
				if (file_exists(demo_bridge_dir + "ready.txt")) {
					demo_request_serial <- demo_request_serial + 1;
					string suffix <- "" + demo_request_serial;
					string action_path <- demo_bridge_dir + "action_" + suffix + ".txt";
					if (file_exists(action_path)) {
						bool removed <- delete_file(action_path);
					}
					string observation <- "" + cycle + "," + global_phase + "," + green_cycles
						+ "," + through_queue_ns + "," + through_queue_ew
						+ "," + through_approach_ns + "," + through_approach_ew
						+ "," + left_queue_ns + "," + left_queue_ew
						+ "," + downstream_ns + "," + downstream_ew;
					save observation to: (demo_bridge_dir + "request_" + suffix + ".txt") format: "text" rewrite: true;
					demo_request_open <- true;
					demo_status <- "Đang chờ quyết định DQN";
				} else {
					demo_status <- "Chưa chạy scripts/demo_dqn.py";
				}
			} else {
				string action_path <- demo_bridge_dir + "action_" + demo_request_serial + ".txt";
				if (file_exists(action_path)) {
					loop answer over: text_file(action_path) {
						if (answer = "0" or answer = "1") {
							dqn_action <- int(answer);
							demo_actions_received <- demo_actions_received + 1;
							demo_request_open <- false;
							demo_status <- "DQN: " + (dqn_action = 0 ? "GIỮ XANH" : "CHUYỂN PHA");
							break;
						}
					}
					if (!demo_request_open) {
						bool removed <- delete_file(action_path);
					}
				}
			}
		} else if (demo_request_open) {
			// The 60 s safety limit can end a decision before Python replies.
			demo_request_open <- false;
		}
	}
}

experiment my_experiment type: gui {
	parameter "Python Controlled" var: python_controlled init: true;
	parameter "Trained DQN demo" var: demo_file_bridge init: true;
	float minimum_cycle_duration <- 0.033#s; // About 3 model seconds per wall second (step = 0.1 s)
	output synchronized: true {
		monitor "Trạng thái DQN" value: demo_status;
		monitor "Quyết định DQN" value: demo_actions_received;
		monitor "Xe hoàn thành mặt đất" value: total_vehicles_passed;
		monitor "Hàng đợi mặt đất" value: current_queue_length;
		monitor "Pha đèn" value: phase_names[global_phase];
		monitor "Đếm ngược pha" value: signal_countdown;
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
				draw rectangle(360, 115) at: {190, 62} color: rgb(15, 23, 42, 215) border: rgb(71, 85, 105);
				draw "TRAFFIC-LIGHT ABM | TRAINED DQN" at: {20, 22} color: #lime font: font("Helvetica", 13, #bold);
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " | Green: " + green_cycles + "c" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);
				draw demo_status + " | Action: " + (current_action = 0 ? "EXTEND" : "SWITCH") at: {20, 68} color: #orange font: font("Helvetica", 12, #plain);
				draw "Queue NS: " + queue_ns + " | EW: " + queue_ew + " | DQN decisions: " + demo_actions_received at: {20, 88} color: rgb(56, 189, 248) font: font("Helvetica", 12, #plain);
				draw "Passed: " + total_vehicles_passed + " | Queue: " + current_queue_length at: {20, 108} color: rgb(226, 232, 240) font: font("Helvetica", 12, #plain);
			}
		}
	}
}
