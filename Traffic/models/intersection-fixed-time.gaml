/**
* Name: intersection_traffic
* Description: 4-way traffic intersection simulation with Fixed-Time baseline signal controller.
*              Enhanced with realistic mixed-traffic vehicles (cars, motorcycles, buses)
*              and rich urban road infrastructure visuals.
*/

model intersection_trafficn
import "../includes/collision_avoidance.gaml"

global {
	// Each axis: through green/yellow, then protected left green/yellow.
	float global_counter <- 0.0;
	list<float> phase_durations <- [30#s, 3#s, 12#s, 3#s,
		30#s, 3#s, 12#s, 3#s];

	reflex traffic_light_controller {
		signal_phase_schedule <- phase_durations;
		global_counter <- (round(global_counter / step) + 1) * step;
		phase_timer <- global_counter;
		float required <- phase_durations[global_phase] + (global_phase mod 2 = 1 ? 2 #s : 0 #s);
		if (global_counter >= required and (global_phase mod 2 = 0 or signal_junction_clear())) {
			global_counter <- 0.0;
			phase_timer <- 0.0;
			global_phase <- (global_phase + 1) mod length(phase_durations);
		}
		float remaining <- max([0.0, phase_durations[global_phase] - global_counter]);
		signal_display_seconds <- int(remaining / (1 #s) + 0.999);
		signal_countdown <- "" + int(remaining / (1 #s) + 0.999) + "s";
	}

	// A1+A4: Spawn and metrics are now centralized in collision_avoidance.gaml
	// Only verbose logging remains here (specific to Fixed-Time)
	reflex verbose_log when: (cycle mod 50 = 0 and cycle > 0) {
		int rem_time <- int(phase_durations[global_phase] - global_counter);
		write ">> Cycle: " + cycle + " | Phase: " + phase_names[global_phase] + " (Còn " + rem_time + "s / " + int(phase_durations[global_phase]) + "s) | Passed: " + total_vehicles_passed + " | Queue: " + current_queue_length + " | TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | Severe TTC: " + severe_conflicts_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s";
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
			// 1. Mat duong Asphalt & Via he
			species doan_ground aspect: base;
			species road aspect: base;
			// 2. Vach ke duong, vach di bo zebra, mui ten chi lan, yellow box
			// 3. Cot den tin hieu 3 mau phat sang
			species traffic_signal_post aspect: base;
			// 4. O xung dot tam ngat tu do PET
			species conflict_zone aspect: base;
			// 5. Phuong tien giao thong chi tiet (Xe may, O to, Bus)
			species base_vehicle_species aspect: base;

			// 6. BANG DIEU KHIEN HUD DARK GLASSMORPHISM
			graphics "HUD" {
				int rem_time <- int(phase_durations[global_phase] - global_counter);
				rgb phase_c <- global_phase mod 2 = 0 ? #lime : #yellow;

				// Khung HUD ban mo
				draw rectangle(310, 105) at: {165, 58} color: rgb(15, 23, 42, 215) border: rgb(71, 85, 105);

				// Tieu de
				draw "TRAFFIC-LIGHT ABM | FIXED-TIME" at: {20, 22} color: rgb(241, 196, 15) font: font("Helvetica", 13, #bold);

				// Pha den & Dem lui
				draw circle(4.5) at: {26, 42} color: phase_c;
				draw "Phase: " + phase_names[global_phase] + " (Còn " + rem_time + "s / " + int(phase_durations[global_phase]) + "s)" at: {38, 45} color: phase_c font: font("Helvetica", 14, #bold);

				// Chi so luu thong
				draw "Passed: " + total_vehicles_passed + " xe | Queue: " + current_queue_length + " | Avg Wait: " + (round(average_waiting_time*10.0)/10.0) + "s" at: {20, 68} color: rgb(226, 232, 240) font: font("Helvetica", 12, #plain);

				// Chi so an toan
				draw "TIT: " + (round(total_tit_sum*10.0)/10.0) + "s | TTC<1s: " + severe_conflicts_count + " | HardBrakes: " + hard_braking_count + " | PET: " + (round(min_pet_observed*10.0)/10.0) + "s" at: {20, 90} color: rgb(248, 113, 113) font: font("Helvetica", 11, #plain);
			}
		}
		display performance_charts {
			chart "Performance Metrics (Fixed-Time Baseline)" type: series {
				data "Queue Length (vehicles)" value: current_queue_length color: #crimson;
				data "Avg Waiting Time (s)" value: average_waiting_time color: #darkorange;
				data "Avg Speed (km/h)" value: average_speed color: #dodgerblue;
			}
		}
	}
}

experiment my_batch_experiment type: batch until: (cycle >= doan_horizon_cycles) {
	parameter "Seed" var: seed init: 5001.0;
}

experiment python_parity type: gui autorun: false {
	parameter "Demand CSV" var: demand_file_path;
	parameter "Traffic seed" var: traffic_seed;
	parameter "Fast simulation" var: fast_training init: true;
}
