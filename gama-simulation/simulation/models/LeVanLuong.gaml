/**
 * Name: LeVanLuong
 * Ngã tư Lê Văn Lương × Hoàng Đạo Thúy / Hoàng Minh Giám (cạnh 27 Lê Văn Lương, Thanh Xuân, Hà Nội).
 * Bố cục đường theo số đo ảnh vệ tinh (scripts/build_layout.py → includes/*.geojson), mô phỏng bằng advanced_driving:
 * mỗi chiều mỗi nhánh là 1 road với num_lanes = số sublane rộng 1,05 m (3 sublane ≈ 1 làn to); xe máy chiếm 1 sublane, ô tô 2 sublane.
 * Vùng xung đột = các road "connector" (trái / thẳng / phải) có hình học thật nối vạch dừng → đầu nhánh ra.
 */
model LeVanLuong

global {
	file roads_file <- geojson_file("../includes/roads.geojson");
	file areas_file <- geojson_file("../includes/areas.geojson");
	geometry shape <- envelope(areas_file);
	float step <- 0.2 #s;
	float traffic_seed <- -1.0; // Optional ordinary parameter for paired headless validation.
	string demand_file_path <- "";
	bool replay_demand <- false;
	matrix<float> replay_rows;
	int replay_cursor <- 0;
	int total_vehicles_created <- 0;
	int total_vehicles_passed <- 0;
	float sublane_width <- 1.05; // m — phải khớp SUBLANE_W trong scripts/build_layout.py (num_lanes đã tính sẵn từ số này)

	// ---- Nhu cầu (giả định, sửa theo số đếm thực tế) ----
	float flow_per_approach <- 2400.0; // xe/h mỗi nhánh vào
	float motorbike_ratio <- 0.85;
	float p_left <- 0.2;
	float p_right <- 0.2;

	// ---- Đèn 2 pha cố định (giả định): pha LVL (NE + SW), pha HĐT (NW + SE) ----
	float green_lvl <- 50.0 #s;
	float green_hdt <- 40.0 #s;
	float all_red <- 2.0 #s;
	float yellow_duration <- 4.0 #s;
	float cycle_length <- green_lvl + green_hdt + 2 * all_red; // "cycle" là biến có sẵn của GAMA
	string phase <- "LVL";
	string yellow_phase <- "";
	float phase_remaining <- 0.0 #s;
	bool train_mode <- false;
	bool dqn_waiting_action <- false;
	int dqn_action <- -1;
	int next_decision_in_cycles <- 50;
	int red_ticks_left <- 0;
	int yellow_ticks_left <- 0;
	string pending_phase <- "LVL";
	float queue_time_integral <- 0.0;
	float system_time_integral <- 0.0;
	float weighted_delay_integral <- 0.0;
	int green_ticks <- 0;

	graph road_network;
	list<road> approaches;
	map<string, int> front_row_max; // số sublane bị chiếm đồng thời ở hàng đầu (≤ 1,5 m trước vạch dừng), lớn nhất từ đầu mô phỏng

	init {
		if (traffic_seed >= 0.0) { seed <- int(traffic_seed); }
		create area from: areas_file with: [kind::string(read("kind"))];
		create road from: roads_file with: [rid::string(read("id")), kind::string(read("kind")), arm::string(read("arm")),
			to_arm::string(read("to_arm")), turn::string(read("turn")), num_lanes::int(read("num_lanes")), width::float(read("width"))] {
			maxspeed <- (kind = "connector" ? 40.0 : 60.0) * #km / #h;
		}
		// nút = đầu/cuối của các road (Python sinh tọa độ trùng khít nên không cần file nút riêng)
		loop p over: remove_duplicates(road accumulate [first(each.shape.points), last(each.shape.points)]) {
			create intersection with: [location::p];
		}
		road_network <- as_driving_graph(road, intersection);
		approaches <- road where (each.kind = "approach");
		ask approaches { // source_node / target_node do as_driving_graph gán sẵn
			ask intersection(target_node) {
				approach <- myself;
				stop << [];
			}
			front_row_max[arm] <- 0;
		}
	}

	reflex signal {
		if train_mode {
			if dqn_action >= 0 {
				string wanted <- dqn_action = 0 ? "LVL" : "HDT";
				if wanted != phase and red_ticks_left = 0 and yellow_ticks_left = 0 {
					pending_phase <- wanted;
					yellow_phase <- phase in ["LVL", "HDT"] ? phase : "";
					yellow_ticks_left <- yellow_phase != "" ? int(yellow_duration / step) : 0;
					red_ticks_left <- int(all_red / step);
				}
				dqn_action <- -1;
			}
			if yellow_ticks_left > 0 {
				phase <- "none";
				yellow_ticks_left <- yellow_ticks_left - 1;
				phase_remaining <- yellow_ticks_left * step;
				if yellow_ticks_left = 0 { yellow_phase <- ""; }
			} else if red_ticks_left > 0 {
				phase <- "none";
				red_ticks_left <- red_ticks_left - 1;
				phase_remaining <- red_ticks_left * step;
				if red_ticks_left = 0 { phase <- pending_phase; yellow_phase <- ""; }
			} else {
				phase_remaining <- next_decision_in_cycles * step;
			}
			if (cycle + 1) mod 50 = 0 { dqn_waiting_action <- true; }
			next_decision_in_cycles <- dqn_waiting_action ? 1 : 50 - ((cycle + 1) mod 50);
		} else {
			float t <- time - cycle_length * floor(time / cycle_length);
			phase <- t < green_lvl - yellow_duration ? "LVL" : (t < green_lvl ? "none" : (t < green_lvl + all_red ? "none" : (t < green_lvl + all_red + green_hdt - yellow_duration ? "HDT" : "none")));
			yellow_phase <- t >= green_lvl - yellow_duration and t < green_lvl ? "LVL" : (t >= green_lvl + all_red + green_hdt - yellow_duration and t < green_lvl + all_red + green_hdt ? "HDT" : "");
			if (yellow_phase = "LVL") { phase_remaining <- green_lvl - t; }
			else if (yellow_phase = "HDT") { phase_remaining <- green_lvl + all_red + green_hdt - t; }
			else if (phase = "LVL") { phase_remaining <- green_lvl - yellow_duration - t; }
			else if (phase = "HDT") { phase_remaining <- green_lvl + all_red + green_hdt - yellow_duration - t; }
			else if (t < green_lvl) { phase_remaining <- green_lvl - t; }
			else if (t < green_lvl + all_red) { phase_remaining <- green_lvl + all_red - t; }
			else if (t < green_lvl + all_red + green_hdt) { phase_remaining <- green_lvl + all_red + green_hdt - t; }
			else { phase_remaining <- cycle_length - t; }
		}
		if (phase in ["LVL", "HDT"]) {
			green_ticks <- green_ticks + 1;
		} else {
			green_ticks <- 0;
		}
		ask intersection where (each.approach != nil) {
			bool green <- (approach.arm in ["NE", "SW"]) ? (phase = "LVL") : (phase = "HDT");
			if green { stop[0] <- []; } else { stop[0] <- [approach]; }
		}
	}

	action set_dqn_action(int a) {
		dqn_action <- a;
		dqn_waiting_action <- false;
	}

	reflex load_replay_demand when: cycle = 0 and demand_file_path != "" {
		replay_demand <- true;
		replay_rows <- matrix<float>(csv_file(demand_file_path, ",", float, false));
	}

	reflex spawn_replay_demand when: replay_demand {
		loop while: replay_cursor < replay_rows.rows {
			// GAMA exposes cycle n after reflexes observed n-1.
			if (replay_rows[0, replay_cursor] > cycle + 1) { break; }
			int origin <- int(replay_rows[1, replay_cursor]);
			string arm_name <- ["NE", "SE", "SW", "NW"][origin];
			string turn_wanted <- ["left", "right", "through"][int(replay_rows[4, replay_cursor])];
			road incoming <- road first_with (each.rid = arm_name + "_in");
			road connector <- road first_with (each.kind = "connector" and each.arm = arm_name and each.turn = turn_wanted);
			road outgoing <- road first_with (each.kind = "exit" and each.arm = connector.to_arm);
			int occupied <- int(replay_rows[2, replay_cursor]) = 1 ? 2 : 1;
			int lane_max <- incoming.num_lanes - occupied;
			int requested_lane <- int(replay_rows[5, replay_cursor]);
			int right_last <- max([0, int(incoming.num_lanes / 4) - occupied]);
			int initial_lane <- turn_wanted = "left" ? max([requested_lane, lane_max - 2]) :
				(turn_wanted = "right" ? min([requested_lane, right_last]) : requested_lane);
			create vehicle with: [location::incoming.source_node.location,
				dest::intersection(outgoing.target_node),
				is_car::(int(replay_rows[2, replay_cursor]) = 1),
				maneuver::turn_wanted,
				demand_row_id::replay_cursor,
				requested_speed_kmh::replay_rows[3, replay_cursor],
				requested_lane::initial_lane];
			total_vehicles_created <- total_vehicles_created + 1;
			replay_cursor <- replay_cursor + 1;
		}
	}

	reflex spawn when: demand_file_path = "" {
		ask approaches {
			if flip(flow_per_approach / 3600 * step) {
				float r <- rnd(1.0);
				string turn_wanted <- r < p_left ? "left" : (r < p_left + p_right ? "right" : "through");
				road c <- road first_with (each.kind = "connector" and each.arm = arm and each.turn = turn_wanted);
				road ex <- road first_with (each.kind = "exit" and each.arm = c.to_arm);
				create vehicle with: [location::source_node.location, dest::intersection(ex.target_node),
					is_car::flip(1 - motorbike_ratio), maneuver::turn_wanted];
				total_vehicles_created <- total_vehicles_created + 1;
			}
		}
	}

	reflex measure_front_row {
		system_time_integral <- system_time_integral + length(vehicle) * step;
		loop v over: vehicle {
			if v.speed < 1.0 * #km / #h { queue_time_integral <- queue_time_integral + step; }
			weighted_delay_integral <- weighted_delay_integral +
				(1.0 - min([1.0, v.speed / v.max_speed])) * step;
		}
		ask approaches {
			int n <- front_row();
			if n > front_row_max[arm] { front_row_max[arm] <- n; }
		}
	}
}

species area {
	string kind;
	aspect default {
		switch kind {
			match "asphalt" { draw shape color: rgb(70, 70, 75); }
			match "median" { draw shape color: rgb(90, 140, 80); }
			match "crosswalk" { draw shape color: rgb(220, 220, 220, 0.55); }
			match "conflict" { draw shape color: rgb(255, 80, 60, 0.18) border: rgb(255, 80, 60); }
			match "stopline" { draw shape + 0.15 color: #white; }
			match "sublane" { draw shape color: rgb(255, 255, 255, 0.25); }
		}
	}
}

species road skills: [road_skill] {
	string rid;
	string kind;   // approach | exit | connector
	string arm;    // NE | SE | SW | NW (nhánh xuất phát với connector)
	string to_arm;
	string turn;   // left | through | right (connector)
	float width;

	// số sublane khác nhau đang bị chiếm bởi xe có đầu xe cách vạch dừng ≤ 1,5 m
	int front_row {
		list<int> occ <- [];
		loop v over: vehicle where (each.current_road = self and each.distance_to_current_target <= 1.5) {
			occ <- occ + range(v.lowest_lane, v.lowest_lane + v.num_lanes_occupied - 1);
		}
		return length(remove_duplicates(occ));
	}

	aspect default {
		if kind = "connector" { draw shape color: rgb(255, 255, 255, 0.15); }
	}

	aspect label {
		if kind != "connector" {
			draw arm + (kind = "approach" ? " vào: " : " ra: ") + num_lanes + " sublane (" + width + " m)"
				at: shape.points[0] color: #yellow font: font("Arial", 11, #bold) anchor: #center;
		}
	}
}

species intersection skills: [intersection_skill] {
	road approach; // khác nil ⇔ nút vạch dừng có đèn
	aspect default {
		if approach != nil {
			draw circle(1.2) color: empty(stop[0]) ? #green : #red border: #black;
		}
	}
}

species vehicle skills: [driving] {
	int demand_row_id <- -1;
	intersection dest;
	bool is_car;
	string maneuver <- "through";
	float requested_speed_kmh <- -1.0;
	int requested_lane <- -1;
	rgb color;

	init {
		color <- is_car ? rgb(80, 160, 255) : rgb(255, 210, 60); // gán trong init vì is_car đến từ with:
		num_lanes_occupied <- is_car ? 2 : 1;
		vehicle_length <- is_car ? 4.5 #m : 1.9 #m;
		max_speed <- (requested_speed_kmh > 0.0 ? requested_speed_kmh :
			(is_car ? gauss(50, 5) : gauss(40, 5))) * #km / #h;
		if (requested_lane >= 0) { lowest_lane <- requested_lane; }
		max_acceleration <- is_car ? 2.5 : 2.0;
		time_headway <- is_car ? 1.2 : 0.6;
		min_safety_distance <- is_car ? 1.0 : 0.3;
		lane_change_limit <- 1;
		linked_lane_limit <- 0;
		proba_use_linked_road <- 0.0;
		proba_respect_priorities <- 0.0;
		proba_block_node <- 0.0;
		proba_respect_stops <- [1.0];
		acc_gain_threshold <- 0.5;
		politeness_factor <- 0.1;
	}

	reflex choose_path when: final_target = nil {
		do compute_path graph: road_network target: dest;
	}

	reflex move when: final_target != nil {
		// Xe có thể spawn ở mọi sublane; giới hạn hành lang trong 100 m cuối trước vạch dừng.
		// Xe đi thẳng vì thế phải rời làn rẽ phải trước khi xếp hàng chờ đèn.
		if current_road != nil and road(current_road).kind = "approach" and distance_to_current_target <= 100.0 #m {
			int occupied <- num_lanes_occupied;
			int last_lane <- road(current_road).num_lanes - occupied;
			int right_last <- max([0, int(road(current_road).num_lanes / 4) - occupied]);
			int left_first <- max([0, last_lane - 2]);
			int first_lane <- maneuver = "right" ? 0 : (maneuver = "left" ? left_first : min([last_lane, int(road(current_road).num_lanes / 4)]));
			int final_lane <- maneuver = "right" ? min([last_lane, right_last]) :
				(maneuver = "left" ? last_lane : max([first_lane, left_first - occupied]));
			allowed_lanes <- range(first_lane, final_lane);
		} else if current_road != nil {
			allowed_lanes <- range(0, road(current_road).num_lanes - num_lanes_occupied);
		}
		do drive;
		if final_target = nil {
			total_vehicles_passed <- total_vehicles_passed + 1;
			do unregister;
			do die;
		}
	}

	// đường polyline nằm ở mép dải phân cách; sublane 0 sát bó vỉa bên phải
	point display_position {
		float dist <- (road(current_road).num_lanes - lowest_lane - mean(range(num_lanes_occupied - 1)) - 0.5) * sublane_width;
		return location + {cos(heading + 90) * dist, sin(heading + 90) * dist};
	}

	aspect default {
		if current_road != nil {
			draw rectangle(vehicle_length, is_car ? 1.7 : 0.7) at: display_position() rotate: heading color: color border: #black;
		}
	}
}

experiment main type: gui {
	parameter "Lưu lượng mỗi nhánh vào (xe/h)" var: flow_per_approach min: 0.0 max: 8000.0;
	parameter "Tỷ lệ xe máy" var: motorbike_ratio min: 0.0 max: 1.0;
	parameter "Tỷ lệ rẽ trái" var: p_left min: 0.0 max: 1.0;
	parameter "Tỷ lệ rẽ phải" var: p_right min: 0.0 max: 1.0;
	parameter "Xanh Lê Văn Lương (s)" var: green_lvl;
	parameter "Xanh Hoàng Đạo Thúy (s)" var: green_hdt;

	output {
		display "Ngã tư" type: 2d background: rgb(40, 45, 40) {
			species area;
			species road;
			species intersection;
			species vehicle;
			species road aspect: label;
		}
		monitor "Pha đèn" value: phase + (yellow_phase != "" ? " (vàng " + yellow_phase + ")" : "") + " · " + string(phase_remaining) + " s";
		monitor "Số xe" value: length(vehicle);
		monitor "NE vào: sublane / hàng đầu max" value: string((road first_with (each.rid = "NE_in")).num_lanes) + " / " + front_row_max["NE"];
		monitor "SE vào: sublane / hàng đầu max" value: string((road first_with (each.rid = "SE_in")).num_lanes) + " / " + front_row_max["SE"];
		monitor "SW vào: sublane / hàng đầu max" value: string((road first_with (each.rid = "SW_in")).num_lanes) + " / " + front_row_max["SW"];
		monitor "NW vào: sublane / hàng đầu max" value: string((road first_with (each.rid = "NW_in")).num_lanes) + " / " + front_row_max["NW"];
	}
}

// Pauseable headless reference for paired Python/GAMA metric comparisons.
experiment python_parity type: gui autorun: false {
	parameter "Traffic seed" var: traffic_seed;
	parameter "Paired demand CSV" var: demand_file_path;
	parameter "Flow per approach" var: flow_per_approach;
	parameter "Motorbike ratio" var: motorbike_ratio;
	parameter "Left turn share" var: p_left;
	parameter "Right turn share" var: p_right;
	parameter "LVL green" var: green_lvl;
	parameter "HDT green" var: green_hdt;
		parameter "All red (s)" var: all_red;
}

experiment dqn_server_experiment type: gui autorun: false {
	parameter "Traffic seed" var: traffic_seed;
	parameter "Paired demand CSV" var: demand_file_path;
	parameter "Training mode" var: train_mode;
}
