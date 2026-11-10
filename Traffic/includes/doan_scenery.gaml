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
		create traffic_signal_post with: (post_id: 1, direction_axis: 0, location: {967,953}, heading: 180.0);
		create traffic_signal_post with: (post_id: 2, direction_axis: 0, location: {1033,1047}, heading: 0.0);
		create traffic_signal_post with: (post_id: 3, direction_axis: 1, location: {953,1033}, heading: 90.0);
		create traffic_signal_post with: (post_id: 4, direction_axis: 1, location: {1047,967}, heading: 270.0);
	}
}

species doan_ground {
	aspect base {
		draw (square(2000) at_location {1000,1000}) at: {1000,1000,-0.5} color: rgb(15,20,30);

		// Trục Ngang (Đông - Tây): 4 LÀN XE MỖI CHIỀU (Rộng 60m)
		draw (rectangle(2000,60) at_location {1000,1000}) depth: 0.6 color: rgb(35, 42, 55) border: rgb(0, 242, 254, 100);

		// Trục Dọc (Bắc - Nam): 4 LÀN XE MỖI CHIỀU (Rộng 60m)
		draw (rectangle(60,2000) at_location {1000,1000}) depth: 0.6 color: rgb(35, 42, 55) border: rgb(0, 242, 254, 100);

				// VẠCH TIM ĐƯỜNG ĐÔI MÀU VÀNG NÉT LIỀN (DOUBLE YELLOW MEDIAN DIVIDERS)
				// Đường Nguyễn Trãi (Y = 1000.0)
				draw line([{0, 1000.6, 0.72}, {950, 1000.6, 0.72}]) color: rgb(255, 215, 0) width: 2.0;
				draw line([{0, 999.4, 0.72}, {950, 999.4, 0.72}]) color: rgb(255, 215, 0) width: 2.0;
				draw line([{1050, 1000.6, 0.72}, {2000, 1000.6, 0.72}]) color: rgb(255, 215, 0) width: 2.0;
				draw line([{1050, 999.4, 0.72}, {2000, 999.4, 0.72}]) color: rgb(255, 215, 0) width: 2.0;

				// Đường Khuất Duy Tiến - Nguyễn Xiển (X = 1000.0)
				draw line([{999.4, 2000, 0.72}, {999.4, 1050, 0.72}]) color: rgb(255, 215, 0) width: 2.0;
				draw line([{1000.6, 2000, 0.72}, {1000.6, 1050, 0.72}]) color: rgb(255, 215, 0) width: 2.0;
				draw line([{999.4, 950, 0.72}, {999.4, 0, 0.72}]) color: rgb(255, 215, 0) width: 2.0;
				draw line([{1000.6, 950, 0.72}, {1000.6, 0, 0.72}]) color: rgb(255, 215, 0) width: 2.0;

				// 4 VẠCH SƠN DỪNG ĐÈN ĐỎ (STOP BARS) MÀU TRẮNG DÀY SÁT MÉP NGÃ TƯ (ĐÚNG NỬA ĐƯỜNG ĐI VÀO THEO RHT)
				// Nhánh Tây: vạch X=955, phần xe đi Đông.
				draw box(1.8, 25.0, 0.2) at: {955, 1015, 0.75} color: #white;
				// Nhánh Đông: vạch X=1045, phần xe đi Tây.
				draw box(1.8, 25.0, 0.2) at: {1045, 985, 0.75} color: #white;
				// Nhánh Bắc: vạch Y=955, phần xe đi Nam.
				draw box(25.0, 1.8, 0.2) at: {985, 955, 0.75} color: #white;
				// Nhánh Nam: vạch Y=1045, phần xe đi Bắc.
				draw box(25.0, 1.8, 0.2) at: {1015, 1045, 0.75} color: #white;

				// Subtle dashed guides follow the same cubic Bézier control points
				// as the four protected left-turn connectors in the road graph.
				list<point> turn_starts <- [{1040.5,998.0},{998.0,959.5},
					{959.5,1002.0},{1002.0,1040.5}];
				list<point> turn_ends <- [{1002.0,1040.5},{1040.5,1002.0},
					{1002.0,959.5},{959.5,998.0}];
				list<point> turn_controls_1 <- [{1015.5,998.0},{998.0,984.5},
					{984.5,1002.0},{1002.0,1015.5}];
				list<point> turn_controls_2 <- [{1002.0,1015.5},{1015.5,1002.0},
					{1002.0,984.5},{984.5,998.0}];
				loop turn_index from: 0 to: 3 {
					loop dash from: 0 to: 9 {
						float t0 <- dash / 10.0;
						float t1 <- t0 + 0.06;
						point a <- turn_starts[turn_index] * ((1-t0)^3) +
							turn_controls_1[turn_index] * (3*(1-t0)^2*t0) +
							turn_controls_2[turn_index] * (3*(1-t0)*t0*t0) +
							turn_ends[turn_index] * (t0^3);
						point b <- turn_starts[turn_index] * ((1-t1)^3) +
							turn_controls_1[turn_index] * (3*(1-t1)^2*t1) +
							turn_controls_2[turn_index] * (3*(1-t1)*t1*t1) +
							turn_ends[turn_index] * (t1^3);
						draw line([a + {0,0,0.95},b + {0,0,0.95}])
							color: rgb(225,225,210,125) width: 1.2;
					}
				}

				// (Đã bỏ vẽ vạch qua đường cho người đi bộ theo yêu cầu của người dùng)

				// 4 NHÁNH RẼ PHẢI ƯU TIÊN 2 LÀN RIÊNG BIỆT TẠI 4 GÓC PHẦN TƯ (RHT VIỆT NAM)
				// Góc Tây-Nam (Rẽ từ Nguyễn Trãi đi Đông sang Nguyễn Xiển đi Nam)
				draw line([{935, 1026, 0.6}, {974, 1065, 0.6}]) color: rgb(35, 42, 55) width: 22.0;
				draw line([{935, 1026, 0.8}, {974, 1065, 0.8}]) color: #white width: 2.0;
				draw line([{938, 1029, 0.85}, {971, 1062, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;
				draw line([{932, 1023, 0.85}, {977, 1068, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;

				// Góc Tây-Bắc (Rẽ từ Khuất Duy Tiến đi Nam sang Nguyễn Trãi đi Tây)
				draw line([{974, 935, 0.6}, {935, 974, 0.6}]) color: rgb(35, 42, 55) width: 22.0;
				draw line([{974, 935, 0.8}, {935, 974, 0.8}]) color: #white width: 2.0;
				draw line([{971, 938, 0.85}, {938, 971, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;
				draw line([{977, 932, 0.85}, {932, 977, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;

				// Góc Đông-Bắc (Rẽ từ Nguyễn Trãi đi Tây sang Khuất Duy Tiến đi Bắc)
				draw line([{1065, 974, 0.6}, {1026, 935, 0.6}]) color: rgb(35, 42, 55) width: 22.0;
				draw line([{1065, 974, 0.8}, {1026, 935, 0.8}]) color: #white width: 2.0;
				draw line([{1062, 971, 0.85}, {1029, 938, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;
				draw line([{1068, 977, 0.85}, {1023, 932, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;

				// Góc Đông-Nam (Rẽ từ Nguyễn Xiển đi Bắc sang Nguyễn Trãi đi Đông)
				draw line([{1026, 1065, 0.6}, {1065, 1026, 0.6}]) color: rgb(35, 42, 55) width: 22.0;
				draw line([{1026, 1065, 0.8}, {1065, 1026, 0.8}]) color: #white width: 2.0;
				draw line([{1029, 1062, 0.85}, {1062, 1029, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;
				draw line([{1023, 1068, 0.85}, {1068, 1023, 0.85}]) color: rgb(255, 255, 255, 160) width: 1.0;

				// 4 VÒNG QUAY ĐẦU CHỮ U VÀNG (ĐẶT NGAY TRƯỚC VẠCH DỪNG 955 & 1045)
				// 1. U-Turn Bắc (Đón từ Tây X=995 vòng sang Đông X=1005 đi ra hướng Bắc)
				draw line([{995, 930, 0.72}, {995, 940, 0.72}, {1005, 940, 0.72}, {1005, 930, 0.72}]) color: rgb(255, 215, 0) width: 4.0;

				// 2. U-Turn Nam (Đón từ Đông X=1005 vòng sang Tây X=995 đi ra hướng Nam)
				draw line([{1005, 1070, 0.72}, {1005, 1060, 0.72}, {995, 1060, 0.72}, {995, 1070, 0.72}]) color: rgb(255, 215, 0) width: 4.0;

				// 3. U-Turn Tây (Đón từ Nam Y=995 vòng sang Bắc Y=1005 đi ra hướng Tây)
				draw line([{930, 1005, 0.72}, {940, 1005, 0.72}, {940, 995, 0.72}, {930, 995, 0.72}]) color: rgb(255, 215, 0) width: 4.0;

				// 4. U-Turn Đông (Đón từ Bắc Y=1005 vòng sang Nam Y=995 đi ra hướng Đông)
				draw line([{1070, 995, 0.72}, {1060, 995, 0.72}, {1060, 1005, 0.72}, {1070, 1005, 0.72}]) color: rgb(255, 215, 0) width: 4.0;

				// VẠCH PHÂN LÀN TRẮNG NÉT ĐỨT (LÀN ĐƯỜNG NGUYỄN TRÃI)
				loop y_off over: [977.0, 984.0, 991.0, 1009.0, 1016.0, 1023.0] {
					draw line([{0.0, y_off, 0.7}, {2000.0, y_off, 0.7}]) color: rgb(255, 255, 255, 160) width: 1.5;
				}

				// VẠCH PHÂN LÀN TRẮNG NÉT ĐỨT (LÀN ĐƯỜNG KHUẤT DUY TIẾN - NGUYỄN XIỂN)
				loop x_off over: [977.0, 984.0, 991.0, 1009.0, 1016.0, 1023.0] {
					draw line([{x_off, 0.0, 0.7}, {x_off, 2000.0, 0.7}]) color: rgb(255, 255, 255, 160) width: 1.5;
				}

				// MŨI TÊN CHỈ HƯỚNG TRÊN MẶT ĐƯỜNG (LÀN VÀO HƯỚNG VÀO NGÃ TƯ, LÀN RA HƯỚNG RA KHỎI NGÃ TƯ)
				// 1. Nhánh Tây (Nguyễn Trãi): Nam đi Đông (vào), Bắc đi Tây (ra khỏi ngã tư)
				draw line([{880, 1015, 0.72}, {900, 1015, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Vào
				draw line([{900, 985, 0.72}, {880, 985, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Ra khỏi ngã tư
				// 2. Nhánh Đông (Nguyễn Trãi): Bắc đi Tây (vào), Nam đi Đông (ra khỏi ngã tư)
				draw line([{1120, 985, 0.72}, {1100, 985, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Vào
				draw line([{1100, 1015, 0.72}, {1120, 1015, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Ra khỏi ngã tư
				// 3. Nhánh Bắc (Khuất Duy Tiến): Tây đi Nam (vào), Đông đi Bắc (ra khỏi ngã tư)
				draw line([{985, 880, 0.72}, {985, 900, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Vào
				draw line([{1015, 900, 0.72}, {1015, 880, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Ra khỏi ngã tư
				// 4. Nhánh Nam (Nguyễn Xiển): Đông đi Bắc (vào), Tây đi Nam (ra khỏi ngã tư)
				draw line([{1015, 1120, 0.72}, {1015, 1100, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Vào
				draw line([{985, 1100, 0.72}, {985, 1120, 0.72}]) color: rgb(255, 255, 255, 200) width: 2.0; // Ra khỏi ngã tư

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
