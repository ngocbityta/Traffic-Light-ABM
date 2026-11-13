/** Visual structures ported from the former DoAn project.
 * North/south coordinates are mirrored to Traffic's GAMA convention.
 * The network, signals and vehicles are supplied by the shared Traffic model.
 */
model doan_scenery

global {
	// Keep signal state available when this scenery model is validated on its own.
	int global_phase <- 0;
	float phase_timer <- 0.0;
	int signal_display_seconds <- 0;
	float signal_yellow_seconds <- 3#s;
	list<float> signal_phase_schedule <- [];
	action init_doan_scenery {
		create doan_ground;
		// 4 trụ đèn tín hiệu tại vạch dừng 4 nhánh Lê Văn Lương - Hoàng Đạo Thúy / Hoàng Minh Giám
		create traffic_signal_post with: (post_id: 1, direction_axis: 0, location: {1015,987}, heading: 143.4);
		create traffic_signal_post with: (post_id: 2, direction_axis: 0, location: {983,1019}, heading: 322.8);
		create traffic_signal_post with: (post_id: 3, direction_axis: 1, location: {1019,1022}, heading: 218.5);
		create traffic_signal_post with: (post_id: 4, direction_axis: 1, location: {981,983}, heading: 46.4);
	}
}

species doan_ground {
	aspect base {
		draw (square(2000) at_location {1000,1000}) at: {1000,1000,-0.5} color: rgb(15,20,30);

		// 4 Nhánh đường ngã tư Lê Văn Lương - Hoàng Đạo Thúy / Hoàng Minh Giám
		// Nhánh NE: Lê Văn Lương hướng Láng Hạ (bearing 53.4°)
		// Rộng 33.8m (vào 13.3m, dải phân cách 7.8m, ra 12.7m)
		draw (rectangle(1000, 33.8) at_location {1000 + 401.4, 1000 - 298.1})
			rotate: 53.4 - 90.0 depth: 0.6 color: rgb(35, 42, 55) border: rgb(0, 242, 254, 100);

		// Nhánh SW: Lê Văn Lương hướng Khuất Duy Tiến (bearing 232.8°)
		// Rộng 33.1m (vào 12.8m, dải phân cách 7.9m, ra 12.4m)
		draw (rectangle(1000, 33.1) at_location {1000 - 398.3, 1000 + 302.3})
			rotate: 232.8 - 90.0 depth: 0.6 color: rgb(35, 42, 55) border: rgb(0, 242, 254, 100);

		// Nhánh SE: Hoàng Đạo Thúy hướng Nhân Hòa (bearing 128.5°)
		// Rộng 25.2m (vào 13.5m, dải phân cách 2.5m, ra 9.2m)
		draw (rectangle(1000, 25.2) at_location {1000 + 391.3, 1000 + 311.3})
			rotate: 128.5 - 90.0 depth: 0.6 color: rgb(35, 42, 55) border: rgb(0, 242, 254, 100);

		// Nhánh NW: Hoàng Minh Giám hướng Trung Hòa (bearing 316.4°)
		// Rộng 24.4m (vào 11.9m, dải phân cách 2.6m, ra 9.9m)
		draw (rectangle(1000, 24.4) at_location {1000 - 344.8, 1000 - 362.1})
			rotate: 316.4 - 90.0 depth: 0.6 color: rgb(35, 42, 55) border: rgb(0, 242, 254, 100);

		// Vùng xung đột giao lộ trung tâm
		draw circle(35.0) at: {1000, 1000, 0.05} depth: 0.55 color: rgb(35, 42, 55);

		// Dải phân cách cứng (Medians)
		// Median NE (rộng 7.8m)
		draw line([{1000 + 17.4 * 0.8028 - 2.1 * -0.5962, 1000 + 17.4 * -0.5962 - 2.1 * -0.8028, 0.72},
			{1000 + 990.0 * 0.8028 - 2.1 * -0.5962, 1000 + 990.0 * -0.5962 - 2.1 * -0.8028, 0.72}])
			color: rgb(90, 140, 80) width: 7.8;

		// Median SW (rộng 7.9m)
		draw line([{1000 + 21.8 * -0.7965 + 1.05 * 0.6046, 1000 + 21.8 * 0.6046 + 1.05 * 0.7965, 0.72},
			{1000 + 990.0 * -0.7965 + 1.05 * 0.6046, 1000 + 990.0 * 0.6046 + 1.05 * 0.7965, 0.72}])
			color: rgb(90, 140, 80) width: 7.9;

		// Median SE (rộng 2.5m)
		draw line([{1000 + 28.3 * 0.7826 - 6.11 * 0.6225, 1000 + 28.3 * 0.6225 - 6.11 * -0.7826, 0.72},
			{1000 + 990.0 * 0.7826 - 6.11 * 0.6225, 1000 + 990.0 * 0.6225 - 6.11 * -0.7826, 0.72}])
			color: rgb(90, 140, 80) width: 2.5;

		// Median NW (rộng 2.6m)
		draw line([{1000 + 23.7 * -0.6896 + 1.4 * -0.7242, 1000 + 23.7 * -0.7242 + 1.4 * 0.6896, 0.72},
			{1000 + 990.0 * -0.6896 + 1.4 * -0.7242, 1000 + 990.0 * -0.7242 + 1.4 * 0.6896, 0.72}])
			color: rgb(90, 140, 80) width: 2.6;

		// 4 VẠCH SƠN DỪNG ĐÈN ĐỎ (STOP BARS) MÀU TRẮNG DÀY SÁT MÉP NGÃ TƯ
		// Stop bar SE (w_in = 13.5m)
		draw line([{1019.28, 1021.55, 0.75}, {1019.28 + 0.6225 * 13.5, 1021.55 - 0.7826 * 13.5, 0.75}])
			color: #white width: 2.0;

		// Stop bar NE (w_in = 13.3m)
		draw line([{1014.82, 986.75, 0.75}, {1014.82 - 0.5962 * 13.3, 986.75 - 0.8028 * 13.3, 0.75}])
			color: #white width: 2.0;

		// Stop bar NW (w_in = 11.9m)
		draw line([{980.53, 983.47, 0.75}, {980.53 - 0.7242 * 11.9, 983.47 + 0.6896 * 11.9, 0.75}])
			color: #white width: 2.0;

		// Stop bar SW (w_in = 12.8m)
		draw line([{983.19, 1019.04, 0.75}, {983.19 + 0.6046 * 12.8, 1019.04 + 0.7965 * 12.8, 0.75}])
			color: #white width: 2.0;

		// (Làn rẽ phải nằm trong vùng xung đột giao lộ, không vẽ tách nhánh riêng bên ngoài)
		// (Không có làn quay đầu tại ngã tư)
	}
}


// =========================================================================
// LOÀI TÁC TỬ: TRAFFIC SIGNAL POST (ĐÈN THẲNG VÀ RẼ TRÁI RIÊNG)
// =========================================================================
species traffic_signal_post {
	int post_id;
	int direction_axis; // 0: NS, 1: EW
	float heading <- 0.0;

	// movement 0: straight; movement 1: protected left turn.
	rgb get_active_light(int movement) {
		int green_phase <- direction_axis = 0 ? (movement = 0 ? 0 : 2) :
			(movement = 0 ? 4 : 6);
		if (global_phase = green_phase) { return #lime; }
		if (global_phase = green_phase + 1 and phase_timer < signal_yellow_seconds) { return #yellow; }
		return #red;
	}

	int red_wait_seconds(int movement) {
		if (length(signal_phase_schedule) != 8) { return -1; }
		int green_phase <- direction_axis = 0 ? (movement = 0 ? 0 : 2) :
			(movement = 0 ? 4 : 6);
		float remaining <- max([0.0, signal_phase_schedule[global_phase] - phase_timer]);
		int next_phase <- (global_phase + 1) mod 8;
		loop i from: 0 to: 7 {
			if (next_phase != green_phase) {
				remaining <- remaining + signal_phase_schedule[next_phase];
				next_phase <- (next_phase + 1) mod 8;
			}
		}
		return min([99,int(remaining / (1#s) + 0.999)]);
	}

	aspect base {
		draw circle(1.0) at: location color: rgb(56, 66, 80) border: #black;
		// Keep the two heads in a fixed screen order at every approach:
		// protected left on the left, through on the right.
		point board_center <- location;
		if (post_id = 1) { board_center <- location + {7.0, -10.0}; }
		else if (post_id = 2) { board_center <- location + {-7.0, -4.0}; }
		else if (post_id = 3) { board_center <- location + {-23.0, 8.0}; }
		else { board_center <- location + {23.0, -9.0}; }
		draw line([location, board_center]) color: rgb(90, 103, 120);
		draw line([board_center + {-6.3, 0.0}, board_center + {6.3, 0.0}])
			color: rgb(90, 103, 120);
		loop movement over: [0, 1] {
			float side <- movement = 1 ? -6.3 : 6.3;
			rgb movement_color <- movement = 0 ? rgb(93, 204, 252) : rgb(255, 194, 86);
			point box_loc <- board_center + {side, 0.0};
			draw rectangle(3.0, 7.2) at: box_loc color: rgb(15, 20, 29)
				border: movement_color;
			rgb active_c <- get_active_light(movement);
			point red_pt <- box_loc + {0.0, -2.2};
			point yel_pt <- box_loc;
			point grn_pt <- box_loc + {0.0, 2.2};
			if (active_c = #red) { draw circle(1.15) at: red_pt color: rgb(255, 60, 60, 65); }
			if (active_c = #yellow) { draw circle(1.15) at: yel_pt color: rgb(255, 205, 50, 65); }
			if (active_c = #lime) { draw circle(1.15) at: grn_pt color: rgb(50, 245, 125, 65); }
			draw circle(active_c = #red ? 0.85 : 0.65) at: red_pt
				color: (active_c = #red ? rgb(255, 65, 65) : rgb(72, 28, 31)) border: #black;
			draw circle(active_c = #yellow ? 0.85 : 0.65) at: yel_pt
				color: (active_c = #yellow ? rgb(255, 208, 42) : rgb(76, 64, 29)) border: #black;
			draw circle(active_c = #lime ? 0.85 : 0.65) at: grn_pt
				color: (active_c = #lime ? rgb(48, 239, 113) : rgb(23, 72, 46)) border: #black;
			point timer_box <- box_loc + {0.0, 6.0};
			draw rectangle(3.5, 2.3) at: timer_box color: rgb(12, 16, 24)
				border: rgb(100, 114, 133);
			int head_seconds <- active_c = #red ? red_wait_seconds(movement) : signal_display_seconds;
			string shown_time <- head_seconds < 0 ? "-" : "" + head_seconds;
			draw shown_time at: timer_box - {0.8, -0.45} color: active_c
				font: font("Helvetica", 12, #bold);
		}
	}

	aspect doan_3d {
		// One mast carries separate protected-left and through heads over the approach.
		float arm_x <- direction_axis = 0 ? (location.x < 1000 ? 20.0 : -20.0) : 0.0;
		float arm_y <- direction_axis = 1 ? (location.y < 1000 ? 20.0 : -20.0) : 0.0;
		float arm_sign <- direction_axis = 0 ? (location.x < 1000 ? 1.0 : -1.0) :
			(location.y < 1000 ? 1.0 : -1.0);
		draw cylinder(1.0, 15.0) at: location + {0,0,7.5}
			color: rgb(105,115,124) border: rgb(140,151,160);
		draw box(direction_axis = 0 ? 20.0 : 1.0,
			direction_axis = 1 ? 20.0 : 1.0, 0.9)
			at: location + {arm_x / 2.0, arm_y / 2.0, 14.2}
			color: rgb(105,115,124);
		loop movement over: [0,1] {
			// The left-turn head sits nearer the centreline; both face approaching cars.
			float lateral <- arm_sign * (movement = 1 ? 9.5 : -9.5);
			point head <- location + {arm_x + (direction_axis = 0 ? lateral : 0.0),
				arm_y + (direction_axis = 1 ? lateral : 0.0), 15.3};
			draw box(direction_axis = 0 ? 17.0 : 2.7,
				direction_axis = 1 ? 17.0 : 2.7, 4.5)
				at: head color: rgb(20,24,27) border: rgb(80,89,95);
			// Raised countdown housing is unobstructed in the default oblique view.
			draw box(direction_axis = 0 ? 0.8 : 2.7,
				direction_axis = 1 ? 0.8 : 2.7, 4.2)
				at: head + {0,0,3.8} color: rgb(88,97,104);
			draw box(direction_axis = 0 ? 7.4 : 2.7,
				direction_axis = 1 ? 7.4 : 2.7, 6.4)
				at: head + {0,0,6.5} color: rgb(9,13,17) border: rgb(138,148,154);
			rgb active_c <- get_active_light(movement);
			// Opposite approaches share one phase, so both faces remain readable in 3D.
			loop face over: [-1.0,1.0] {
				float visual_sign <- direction_axis = 0 ? -face : face;
				float face_depth <- face * 1.5;
				// Red, amber, green are fixed left to right on a dark housing.
				loop lamp over: [0,1,2] {
					float lamp_u <- visual_sign * (-6.3 + lamp * 3.4);
					point lens <- head + (direction_axis = 0 ?
						{lamp_u,face_depth,0.0} : {face_depth,lamp_u,0.0});
					rgb lens_color <- lamp = 0 ?
						(active_c = #red ? rgb(255,46,37) : rgb(75,22,20)) :
						(lamp = 1 ? (active_c = #yellow ? rgb(255,192,35) : rgb(72,54,18)) :
						(active_c = #lime ? rgb(36,228,85) : rgb(18,64,33)));
					draw sphere(1.55) at: lens color: rgb(5,7,9);
					// In GAMA sphere() uses a radius; offset the lens beyond its bezel.
					draw sphere(1.25) at: lens + (direction_axis = 0 ?
						{0.0,face * 1.05,0.0} : {face * 1.05,0.0,0.0})
						color: lens_color;
				}
				// The timer uses the opposite horizontal orientation from the lamps.
				// Apply it to both digit order and each seven-segment glyph.
				int head_seconds <- active_c = #red ? red_wait_seconds(movement) : signal_display_seconds;
				if (head_seconds >= 0) {
					int seconds <- min([99,max([0,head_seconds])]);
					float digit_sign <- -visual_sign;
					loop digit_index over: [0,1] {
						int digit <- digit_index = 0 ? int(seconds / 10) : seconds mod 10;
						float digit_u <- digit_sign * (digit_index = 0 ? -1.45 : 1.45);
						loop segment over: [0,1,2,3,4,5,6] {
							bool lit <- (segment = 0 and digit != 1 and digit != 4) or
								(segment = 1 and digit != 5 and digit != 6) or
								(segment = 2 and digit != 2) or
								(segment = 3 and digit != 1 and digit != 4 and digit != 7) or
								(segment = 4 and (digit = 0 or digit = 2 or digit = 6 or digit = 8)) or
								(segment = 5 and (digit = 0 or digit = 4 or digit = 5 or digit = 6 or digit = 8 or digit = 9)) or
								(segment = 6 and digit != 0 and digit != 1 and digit != 7);
							if (lit) {
								float seg_u <- digit_u + digit_sign * ((segment = 1 or segment = 2) ? 0.78 :
									((segment = 4 or segment = 5) ? -0.78 : 0.0));
								float seg_z <- segment = 0 ? 2.15 : (segment = 3 ? -2.15 :
									(segment = 6 ? 0.0 : ((segment = 1 or segment = 5) ? 1.08 : -1.08)));
								float seg_width <- (segment = 0 or segment = 3 or segment = 6) ? 1.52 : 0.28;
								float seg_height <- (segment = 0 or segment = 3 or segment = 6) ? 0.28 : 1.75;
								point seg_pos <- head + (direction_axis = 0 ?
									{seg_u,face * 1.72,6.5 + seg_z} : {face * 1.72,seg_u,6.5 + seg_z});
								draw box(direction_axis = 0 ? seg_width : 0.16,
									direction_axis = 1 ? seg_width : 0.16, seg_height)
									at: seg_pos color: active_c;
							}
						}
					}
				}
			}
		}
	}
}
