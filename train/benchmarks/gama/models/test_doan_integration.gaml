/** Continuous integration audit of the actual DoAn network and controller. */
model test_doan_integration
import "intersection-fixed-time.gaml"

global {
	int audit_horizon <- 6000;
	int audit_errors <- 0;
	int audit_red_entries <- 0;
	int audit_overlaps <- 0;
	int audit_lane_errors <- 0;
	int audit_green_idle <- 0;
	int audit_phase_errors <- 0;
	int audit_unsafe_merges <- 0;
	int audit_empty_changes <- 0;
	int audit_wrong_routes <- 0;
	int audit_jumps <- 0;
	int audit_stopbar_violations <- 0;
	int audit_remote_idle <- 0;
	bool audit_force_merge_stall <- false;
	bool audit_force_overtaken_owner <- false;
	base_vehicle_species audit_overtaken_front <- nil;
	base_vehicle_species audit_overtaken_rear <- nil;
	intersection audit_overtaken_merge <- nil;
	int audit_overtaken_injected_cycle <- -1;
	bool audit_overtaken_released <- false;
	bool audit_overtaken_reassigned <- false;
	base_vehicle_species audit_frozen_owner <- nil;
	intersection audit_frozen_merge <- nil;
	float audit_frozen_original_speed <- 0.0;
	bool audit_merge_reassigned <- false;
	float audit_reassign_distance <- 0.0;
	bool audit_owner_resumed <- false;
	bool audit_owner_crossed_merge <- false;
	float audit_min_acc <- 0.0;
	float audit_max_z <- 0.0;
	float audit_min_z <- 0.0;
	int audit_ramp_samples <- 0;
	int audit_prev_phase <- -1;
	int audit_phase_start <- 0;
	int audit_phase_changes <- 0;
	bool audit_direct_phase <- false;
	map<string, road> audit_last_road <- [];
	map<string, int> audit_last_lane <- [];
	map<string, point> audit_last_position <- [];
	map<string, int> audit_idle_ticks <- [];
	map<string, int> audit_unexplained_idle_ticks <- [];
	map<string, int> audit_merge_wait_ticks <- [];
	map<string, int> audit_slow_ticks <- [];
	map<string, int> audit_follow_delay_ticks <- [];
	int audit_slow_reports <- 0;
	int audit_merge_wait_reports <- 0;
	map<string, int> audit_movement_entries <- [];
	map<string, bool> audit_reported_pairs <- [];

	bool body_overlap(base_vehicle_species v, base_vehicle_species w) {
		point pv <- v.compute_position(); point pw <- w.compute_position();
		if (abs(pv.z - pw.z) > 2.5) { return false; }
		float av <- v.heading + v.current_yaw_deviation;
		float aw <- w.heading + w.current_yaw_deviation;
		point uv <- {cos(av),sin(av)}; point vv <- {-sin(av),cos(av)};
		point uw <- {cos(aw),sin(aw)}; point vw <- {-sin(aw),cos(aw)};
		point d <- pw - pv;
		// Separating-axis test on the rendered, oriented vehicle bodies.
		loop axis over: [uv,vv,uw,vw] {
			float separation <- abs(d.x * axis.x + d.y * axis.y);
			float rv <- 0.45 * (v.vehicle_length * abs(uv.x * axis.x + uv.y * axis.y) +
				v.vehicle_width * abs(vv.x * axis.x + vv.y * axis.y));
			float rw <- 0.45 * (w.vehicle_length * abs(uw.x * axis.x + uw.y * axis.y) +
				w.vehicle_width * abs(vw.x * axis.x + vw.y * axis.y));
			if (separation >= rv + rw) { return false; }
		}
		return true;
	}

	// Exercise a stalled reservation on the flyover ramp merge. The owner
	// remains outside the conflict zone; a vehicle on the other incoming edge
	// must eventually receive the slot without an overlap.
	reflex force_merge_stall when: audit_force_merge_stall and !audit_merge_reassigned {
		if (audit_frozen_owner = nil) {
			loop merge over: intersection where (each.node_key = "F_NS1" or each.node_key = "F_SN1") {
				loop owner over: merge.merge_owners {
					if (owner.current_target = merge and owner.distance_to_current_target >
						merge.merge_clearance + owner.vehicle_length / 2.0 + 5.0 #m and
						!empty(active_vehicles where (each != owner and each.current_target = merge and
						 each.current_road != owner.current_road and each.distance_to_current_target < 120 #m and
						 each.lowest_lane < owner.lowest_lane + owner.num_lanes_occupied and
						 each.lowest_lane + each.num_lanes_occupied > owner.lowest_lane))) {
						audit_frozen_owner <- owner;
						audit_frozen_merge <- merge;
						audit_frozen_original_speed <- owner.original_max_speed;
						ask owner { original_max_speed <- 0.01 #km/#h; }
						write "DOAN_FORCED_MERGE_STALL," + cycle + "," + owner.name + "," + merge.node_key;
						break;
					}
				}
				if (audit_frozen_owner != nil) { break; }
			}
		} else if (!(audit_frozen_merge.merge_owners contains audit_frozen_owner) and
			audit_frozen_owner.current_target = audit_frozen_merge and
			audit_frozen_owner.measured_speed < 0.5 #km/#h and
			!empty(audit_frozen_merge.merge_owners where (each.current_road != audit_frozen_owner.current_road))) {
			audit_merge_reassigned <- true;
			audit_reassign_distance <- audit_frozen_owner.distance_to_current_target;
			write "DOAN_MERGE_REASSIGNED," + cycle + "," + audit_frozen_owner.name + "," +
				audit_frozen_merge.node_key + "," + audit_frozen_merge.merge_owners;
			ask audit_frozen_owner { original_max_speed <- audit_frozen_original_speed; }
		}
	}

	reflex observe_merge_recovery when: audit_force_merge_stall and audit_merge_reassigned and !audit_owner_crossed_merge {
		if (!audit_owner_resumed and (audit_frozen_owner.current_target != audit_frozen_merge or
			audit_frozen_owner.distance_to_current_target < audit_reassign_distance - 5.0 #m)) {
			audit_owner_resumed <- true;
			write "DOAN_MERGE_OWNER_RESUMED," + cycle + "," + audit_frozen_owner.name;
		}
		if (audit_frozen_owner.current_target != audit_frozen_merge) {
			audit_owner_crossed_merge <- true;
			write "DOAN_MERGE_OWNER_CROSSED," + cycle + "," + audit_frozen_owner.name;
		}
	}

	// Reproduce a reservation left on a vehicle behind the first car in its
	// lane. The stale rear slot must be released at the next arbitration; the
	// front car may still wait for a legitimate vehicle on the merging ramp.
	reflex force_overtaken_owner when: audit_force_overtaken_owner and cycle > 650 and
		audit_overtaken_injected_cycle < 0 {
		loop merge over: intersection where (each.node_key = "F_NS1" or each.node_key = "F_SN1") {
			list<base_vehicle_species> same_edge <- active_vehicles where
				(each.current_target = merge and each.current_road != nil and
				 road(each.current_road).road_kind = "flyover" and each.distance_to_current_target < 120 #m and
				 each.distance_to_current_target > 25 #m);
			if (length(same_edge) >= 2) {
				base_vehicle_species front <- same_edge with_min_of each.distance_to_current_target;
				list<base_vehicle_species> rears <- same_edge where
					(each != front and each.current_road = front.current_road and
					 each.distance_to_current_target > front.distance_to_current_target + 5 #m and
					 each.lowest_lane < front.lowest_lane + front.num_lanes_occupied and
					 each.lowest_lane + each.num_lanes_occupied > front.lowest_lane);
				if (!empty(rears)) {
					base_vehicle_species rear <- rears with_min_of each.distance_to_current_target;
					ask merge { merge_owners <- [rear]; }
					audit_overtaken_front <- front;
					audit_overtaken_rear <- rear;
					audit_overtaken_merge <- merge;
					audit_overtaken_injected_cycle <- cycle;
					write "DOAN_OVERTAKEN_INJECTED," + cycle + "," + merge.node_key + "," + front.name + "," + rear.name;
					break;
				}
			}
		}
	}

	reflex observe_overtaken_owner when: audit_force_overtaken_owner and
		audit_overtaken_injected_cycle >= 0 and !audit_overtaken_reassigned {
		if (!audit_overtaken_released and
			!(audit_overtaken_merge.merge_owners contains audit_overtaken_rear)) {
			audit_overtaken_released <- true;
			write "DOAN_OVERTAKEN_RELEASED," + cycle + "," + audit_overtaken_rear.name;
		}
		if (audit_overtaken_merge.merge_owners contains audit_overtaken_front) {
			audit_overtaken_reassigned <- true;
			write "DOAN_OVERTAKEN_REASSIGNED," + cycle + "," + audit_overtaken_front.name;
		}
	}

	reflex audit_doan {
		if (cycle = 1) {
			write "DOAN_SIGNAL_DISPLAY," + global_phase + "," + signal_countdown;
			loop r over: road {
				if (r.source_node = nil or r.target_node = nil or r.num_lanes < 2) { audit_errors <- audit_errors + 1; }
			}
			if (length(road) != 64 or length(intersection) != 56) { audit_errors <- audit_errors + 1; }
			write "DOAN_SEED," + seed;
			write "DOAN_TOPOLOGY," + length(road) + "," + length(intersection) + "," + audit_errors;
		}
		if (audit_prev_phase >= 0 and global_phase != audit_prev_phase) {
			int phase_ticks <- cycle - audit_phase_start;
			bool invalid_phase <- audit_direct_phase ?
				(audit_prev_phase mod 2 = 0 ?
					global_phase != (audit_prev_phase + 1) mod 8 : global_phase mod 2 != 0) :
				global_phase != (audit_prev_phase + 1) mod 8;
			bool invalid_duration <- audit_direct_phase ?
				(audit_prev_phase mod 2 = 1 ? phase_ticks < 49 : phase_ticks < 99) :
				(audit_prev_phase mod 2 = 1 ? phase_ticks < 49 :
					(phase_ticks < 99 or phase_ticks > 601));
			if (invalid_phase or invalid_duration) {
				audit_phase_errors <- audit_phase_errors + 1;
			}
			audit_phase_changes <- audit_phase_changes + 1;
			audit_phase_start <- cycle;
		}
		audit_prev_phase <- global_phase;
		loop v over: active_vehicles {
			if (v.current_road != nil and v.final_target != nil) {
				road r <- road(v.current_road); string k <- v.name;
				if (v.current_acc < audit_min_acc and v.current_acc < -12) {
					write "DOAN_BRAKE," + cycle + "," + k + "," + v.route_kind + "," + r.road_key + "," + v.current_acc + "," + v.speed + "," + v.distance_to_current_target + "," + v.maneuver + "," + v.lowest_lane + "," + v.max_speed;
					loop w over: v.nearby_vehicles(50 #m) where (each != v and
						(each.current_road = r or (v.next_road != nil and each.current_road = road(v.next_road)))) {
						write "DOAN_BRAKE_NEIGHBOR," + cycle + "," + w.name + "," +
							road(w.current_road).road_key + "," + w.distance_to_current_target + "," +
							w.measured_speed + "," + w.lowest_lane + "," + w.num_lanes_occupied;
					}
				}
				audit_min_acc <- min([audit_min_acc,v.current_acc]);
				audit_max_z <- max([audit_max_z,v.surface_height()]);
				audit_min_z <- min([audit_min_z,v.surface_height()]);
				if (r.road_kind = "ramp" and v.surface_height() > 2 and v.surface_height() < 48) { audit_ramp_samples <- audit_ramp_samples + 1; }
				if (v.lowest_lane < 0 or v.lowest_lane + v.num_lanes_occupied > r.num_lanes) { audit_lane_errors <- audit_lane_errors + 1; }
				if (v.uses_signal and r.is_incoming and
					intersection(r.target_node).is_traffic_signal and v.measured_speed < 0.5 #km/#h and
					v.distance_to_current_target < signal_node_to_stop_bar + v.vehicle_length / 2.0 + 0.9 #m) {
					audit_stopbar_violations <- audit_stopbar_violations + 1;
					if (audit_stopbar_violations <= 10) {
						write "DOAN_STOPBAR," + cycle + "," + k + "," + r.road_key + "," + v.distance_to_current_target;
					}
				}
				point pos <- v.compute_position();
				if (audit_last_position contains_key k and distance_to(pos,audit_last_position[k]) > 3.5 #m) {
					audit_jumps <- audit_jumps + 1;
					if (audit_jumps <= 10) { write "DOAN_JUMP," + cycle + "," + k + "," + v.route_kind + "," + r.road_key + "," + distance_to(pos,audit_last_position[k]) + "," + audit_last_position[k] + "," + pos + "," + v.heading + "," + v.current_lateral_shift; }
				}
				audit_last_position[k] <- pos;
				if (audit_last_road contains_key k and audit_last_road[k] != r) {
					if (r.is_signal_connector) {
						bool valid <- v.maneuver = "left" ?
							((v.approach_ns and (global_phase = 2 or
								(global_phase = 3 and phase_timer < signal_yellow_seconds))) or
							 (!v.approach_ns and (global_phase = 6 or
								(global_phase = 7 and phase_timer < signal_yellow_seconds)))) :
							((v.approach_ns and (global_phase = 0 or
								(global_phase = 1 and phase_timer < signal_yellow_seconds))) or
							 (!v.approach_ns and (global_phase = 4 or
								(global_phase = 5 and phase_timer < signal_yellow_seconds))));
						if ((!valid and !v.yellow_commit) or
							(v.maneuver = "left" and !v.left_committed)) {
							audit_red_entries <- audit_red_entries + 1;
							write "DOAN_RED," + cycle + "," + v.name + "," + r.road_key + "," +
								v.maneuver + "," + global_phase + "," + v.left_committed;
						}
					}
					if (r.road_kind = "through" or r.road_kind = "left" or r.road_kind = "right") {
						string expected <- v.maneuver = "straight" ? "through" : v.maneuver;
						if (r.road_kind != expected or r.approach_id != v.approach_id) { audit_wrong_routes <- audit_wrong_routes + 1; }
						audit_movement_entries[r.road_key] <- (audit_movement_entries contains_key r.road_key ? audit_movement_entries[r.road_key] : 0) + 1;
					}
				}
				list<base_vehicle_species> nearby <- v.nearby_vehicles(65 #m);
				bool blocked <- false;
				float nearest_leader_gap <- 1e9 #m;
				float nearest_leader_speed <- 0.0 #m/#s;
				string nearest_leader_name <- "";
				loop w over: nearby {
					if (w.current_road = r and w != v) {
						bool shared <- w.lowest_lane < v.lowest_lane + v.num_lanes_occupied and w.lowest_lane + w.num_lanes_occupied > v.lowest_lane;
						float route_gap <- v.distance_to_current_target - w.distance_to_current_target -
							(v.vehicle_length + w.vehicle_length) / 2.0;
						if (shared and route_gap >= 0.0 #m and route_gap < nearest_leader_gap) {
							nearest_leader_gap <- route_gap;
							nearest_leader_speed <- w.measured_speed;
							nearest_leader_name <- w.name;
						}
						if (shared and route_gap > -1.0 #m and route_gap < 35 #m) { blocked <- true; }
						float along <- (w.location.x - v.location.x) * cos(v.heading) + (w.location.y - v.location.y) * sin(v.heading);
						if (shared and along < 0 and audit_last_road contains_key k and audit_last_road[k] = r and
							audit_last_lane[k] != v.lowest_lane) {
							float gap <- -along - (v.vehicle_length + w.vehicle_length) / 2;
							if (gap < 3 or (w.speed > v.speed and gap / (w.speed - v.speed) < 1.5)) { audit_unsafe_merges <- audit_unsafe_merges + 1; }
						}
					}
					if (v.next_road != nil) {
						list<road> lookahead_roads <- [road(v.next_road)];
						if (road(v.next_road).shape.perimeter < 30.0 #m) {
							intersection mid_node <- intersection(road(v.next_road).target_node);
							road edge_after <- first(mid_node.roads_out where (road(each).road_kind = "ground" and road(each).is_incoming));
							if (edge_after != nil) {
								lookahead_roads <- lookahead_roads + edge_after;
							}
						}
						loop next_r over: lookahead_roads {
							if (w.current_road = next_r) {
								bool shared_next <- w.lowest_lane < v.lowest_lane + v.num_lanes_occupied and
									w.lowest_lane + w.num_lanes_occupied > v.lowest_lane;
								float dist_to_r <- next_r = road(v.next_road) ?
									v.distance_to_current_target :
									v.distance_to_current_target + road(v.next_road).shape.perimeter;
								float next_progress <- next_r.shape.perimeter - w.distance_to_current_target;
								float route_gap <- dist_to_r + next_progress -
									(v.vehicle_length + w.vehicle_length) / 2.0;
								if (shared_next and route_gap >= 0.0 #m and route_gap < nearest_leader_gap) {
									nearest_leader_gap <- route_gap;
									nearest_leader_speed <- w.measured_speed;
									nearest_leader_name <- w.name;
								}
								if (shared_next and route_gap > -1.0 #m and route_gap < 35 #m) { blocked <- true; }
							}
						}
					}
					if (v.name < w.name and distance_to(pos,w.compute_position()) < 13 and body_overlap(v,w)) {
						audit_overlaps <- audit_overlaps + 1;
						string pair_key <- v.name + ":" + w.name;
						if (!(audit_reported_pairs contains_key pair_key)) {
							audit_reported_pairs[pair_key] <- true;
							write "DOAN_OVERLAP," + cycle + "," + v.name + "," + w.name + "," + r.road_key + "," + road(w.current_road).road_key + "," + v.maneuver + "," + w.maneuver;
							write "DOAN_OVERLAP_DETAIL," + cycle + "," + pos + "," + w.compute_position() + "," +
								v.lowest_lane + "," + w.lowest_lane + "," + v.distance_to_current_target + "," +
								w.distance_to_current_target + "," + v.measured_speed + "," + w.measured_speed;
						}
					}
				}
				bool delayed_follower <- v.measured_speed < 0.5 #km/#h and v.max_speed > 3.0 #m/#s and
					nearest_leader_gap > 8.0 #m and nearest_leader_gap < 35.0 #m and
					nearest_leader_speed > 2.0 #m/#s;
				int follow_delay_ticks <- delayed_follower ?
					(audit_follow_delay_ticks contains_key k ? audit_follow_delay_ticks[k] + 1 : 1) : 0;
				audit_follow_delay_ticks[k] <- follow_delay_ticks;
				if (follow_delay_ticks = 20) {
					write "DOAN_FOLLOW_DELAY," + cycle + "," + k + "," + r.road_key + "," +
						nearest_leader_name + "," + nearest_leader_gap + "," + nearest_leader_speed + "," +
						v.measured_speed + "," + v.max_speed;
				}
				bool signal_hold <- v.uses_signal and r.is_incoming and
					intersection(r.target_node).is_traffic_signal and
					v.distance_to_current_target <= signal_node_to_stop_bar +
						v.vehicle_length / 2.0 + stop_bar_clearance + 1.0 #m;
				bool merge_hold <- v.current_target != nil and intersection(v.current_target).managed_merge and
					!(intersection(v.current_target).merge_owners contains v) and
					v.distance_to_current_target <= intersection(v.current_target).merge_clearance +
						v.vehicle_length / 2.0 + v.min_safety_distance + 5.0 #m;
				bool ahead_merge_hold <- false;
				if (v.next_road != nil) {
					intersection ahead <- intersection(road(v.next_road).target_node);
					float distance_to_ahead <- v.distance_to_current_target + road(v.next_road).shape.perimeter;
					ahead_merge_hold <- ahead.managed_merge and
						!(ahead.merge_owners contains v) and distance_to_ahead <= ahead.merge_clearance +
							v.vehicle_length / 2.0 + v.min_safety_distance + 5.0 #m;
				}
				bool slow <- v.measured_speed < 0.5 #km/#h and v.distance_to_current_target > 2 #m;
				int slow_ticks <- slow ? (audit_slow_ticks contains_key k ? audit_slow_ticks[k] + 1 : 1) : 0;
				audit_slow_ticks[k] <- slow_ticks;
				if ((slow_ticks = 30 or slow_ticks = 300 or slow_ticks = 600) and audit_slow_reports < 120) {
					audit_slow_reports <- audit_slow_reports + 1;
					write "DOAN_SLOW," + cycle + "," + k + "," + r.road_key + "," +
						v.route_kind + "," + v.maneuver + "," + v.lowest_lane + "," +
						v.distance_to_current_target + "," + v.max_speed + "," +
						v.speed + "," + v.measured_speed + "," + v.original_max_speed + "," +
						intersection(v.current_target).node_key + "," +
						(v.next_road = nil ? "nil" : road(v.next_road).road_key) + "," +
						global_phase + "," + blocked + "," + signal_hold + "," +
						merge_hold + "," + ahead_merge_hold + "," + nearest_leader_name + "," +
						nearest_leader_gap + "," + nearest_leader_speed;
					if (merge_hold) {
						intersection held <- intersection(v.current_target);
						loop owner over: held.merge_owners {
							write "DOAN_SLOW_OWNER," + cycle + "," + held.node_key + "," + owner.name + "," +
								(owner.current_road = nil ? "nil" : road(owner.current_road).road_key) + "," +
								owner.distance_to_current_target + "," + owner.measured_speed + "," + owner.lowest_lane + "," +
								(owner.current_target = nil ? "nil" : intersection(owner.current_target).node_key);
						}
					}
				}
					bool unexplained_idle <- cycle > 20 and time - v.spawn_time > 5#s and
						v.measured_speed < 0.5 #km/#h and !blocked and !signal_hold and !merge_hold and !ahead_merge_hold and
						v != audit_frozen_owner and
					v.distance_to_current_target > 2.0 #m;
				int unexplained_ticks <- unexplained_idle ?
					(audit_unexplained_idle_ticks contains_key k ? audit_unexplained_idle_ticks[k] + 1 : 1) : 0;
				audit_unexplained_idle_ticks[k] <- unexplained_ticks;
					if (unexplained_ticks = 30) {
						audit_remote_idle <- audit_remote_idle + 1;
						if (audit_remote_idle <= 20) {
							write "DOAN_REMOTE_IDLE," + cycle + "," + k + "," + r.road_key + "," +
								v.maneuver + "," + v.distance_to_current_target + "," + v.max_speed + "," +
								v.speed + "," + v.measured_speed + "," + intersection(v.current_target).node_key + "," +
								(v.next_road = nil ? "nil" : road(v.next_road).road_key);
						}
					}
					bool stalled_at_merge <- v.measured_speed < 0.5 #km/#h and !blocked and
						!signal_hold and (merge_hold or ahead_merge_hold);
					int merge_ticks <- stalled_at_merge ?
						(audit_merge_wait_ticks contains_key k ? audit_merge_wait_ticks[k] + 1 : 1) : 0;
					audit_merge_wait_ticks[k] <- merge_ticks;
					if (merge_ticks >= 300 and merge_ticks mod 100 = 0 and audit_merge_wait_reports < 20) {
						audit_merge_wait_reports <- audit_merge_wait_reports + 1;
						intersection held_merge <- merge_hold ? intersection(v.current_target) :
							intersection(road(v.next_road).target_node);
						write "DOAN_MERGE_WAIT," + cycle + "," + k + "," + r.road_key + "," +
							held_merge.node_key + "," + v.distance_to_current_target + "," +
							v.max_speed + "," + v.speed + "," + v.lowest_lane + "," + held_merge.merge_owners;
						loop owner over: held_merge.merge_owners {
							write "DOAN_MERGE_OWNER," + cycle + "," + held_merge.node_key + "," +
								owner.name + "," + road(owner.current_road).road_key + "," +
								owner.distance_to_current_target + "," + owner.measured_speed + "," + owner.lowest_lane + "," +
								(owner.current_target = nil ? "nil" : intersection(owner.current_target).node_key) + "," +
								distance_to(owner.location,held_merge.location);
						}
					}
				if (audit_last_road contains_key k and audit_last_road[k] = r and audit_last_lane[k] != v.lowest_lane and
					empty(nearby where (each != v and each.current_road = r and distance_to(each.location,v.location) < 30))) {
					audit_empty_changes <- audit_empty_changes + 1;
				}
				bool green <- v.maneuver = "left" ? ((v.approach_ns and global_phase = 2) or (!v.approach_ns and global_phase = 6)) :
					((v.approach_ns and global_phase = 0) or (!v.approach_ns and global_phase = 4));
				// Check the actual signal head. A vehicle on _in_2 can wait at
				// the upstream U-turn reservation or behind a vehicle on _in_3.
				bool idle <- v.uses_signal and r.is_incoming and
					intersection(r.target_node).is_traffic_signal and green and
					phase_timer > 3 and !blocked and
					distance_to(v.location,intersection_center) < distance_to_intersection_center + 20 and junction_clear() and v.measured_speed < 0.5 #km/#h and
					!(v.maneuver = "left" and !v.left_committed);
				int ticks <- idle ? (audit_idle_ticks contains_key k ? audit_idle_ticks[k] + 1 : 1) : 0;
				audit_idle_ticks[k] <- ticks;
				if (ticks = 20) { audit_green_idle <- audit_green_idle + 1; write "DOAN_IDLE," + cycle + "," + k + "," + r.road_key; }
				audit_last_road[k] <- r; audit_last_lane[k] <- v.lowest_lane;
			}
		}
		if (cycle mod 500 = 0 or cycle = audit_horizon - 1) {
			if (total_vehicles_created != total_vehicles_passed + length(active_vehicles where (each.route_kind = "ground")) or
				bypass_vehicles_created != bypass_vehicles_passed + length(active_vehicles where (each.route_kind != "ground"))) {
				audit_errors <- audit_errors + 1;
			}
			if (audit_min_z < -60.001 or audit_max_z > 50.001) { audit_errors <- audit_errors + 1; }
			write "DOAN_AUDIT," + cycle + "," + audit_errors + "," + failed_route_count + "," + audit_red_entries + "," + audit_overlaps + "," + audit_lane_errors + "," + audit_green_idle + "," + audit_phase_errors + "," + audit_unsafe_merges + "," + audit_empty_changes + "," + audit_wrong_routes + "," + audit_jumps + "," + audit_min_acc + "," + total_vehicles_created + "," + total_vehicles_passed + "," + bypass_vehicles_created + "," + bypass_vehicles_passed + "," + left_turn_completed_count + "," + right_turn_completed_count + ",0," + audit_phase_changes + "," + audit_min_z + "," + audit_max_z + "," + audit_ramp_samples + "," + audit_stopbar_violations + ",0,0,0," + audit_remote_idle;
			write "DOAN_ROUTES," + route_completed;
			write "DOAN_MOVEMENTS," + audit_movement_entries;
		}
	}
}

experiment doan_audit_batch type: batch until: cycle >= audit_horizon {
	parameter "Seed" var: traffic_seed init: 5001.0;
	parameter "NS interval" var: spawn_rate_NS init: 100;
	parameter "EW interval" var: spawn_rate_EW init: 100;
}
