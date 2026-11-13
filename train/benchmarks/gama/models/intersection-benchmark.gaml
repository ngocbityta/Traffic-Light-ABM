/** Shared signal schedule and replayed demand for all four Python policies. */
model intersection_benchmark
import "intersection-dqn-legacy.gaml"

global {
	int through_queue_ns <- 0;
	int through_queue_ew <- 0;
	int through_approach_ns <- 0;
	int through_approach_ew <- 0;
	int left_queue_ns <- 0;
	int left_queue_ew <- 0;
	int downstream_ns <- 0;
	int downstream_ew <- 0;

	reflex observe_benchmark_movements {
		through_queue_ns <- 0; through_queue_ew <- 0;
		through_approach_ns <- 0; through_approach_ew <- 0;
		left_queue_ns <- 0; left_queue_ew <- 0;
		downstream_ns <- 0; downstream_ew <- 0;
		loop v over: active_vehicles {
			if (v.current_road != nil and distance_to(v.location, intersection_center) <= detector_distance) {
				road r <- road(v.current_road);
				if (r.is_outer_road and r.is_incoming and v.uses_signal) {
					if (v.maneuver = "straight") {
						if (v.speed < 1 #km/#h) {
							if (v.approach_ns) { through_queue_ns <- through_queue_ns + 1; }
							else { through_queue_ew <- through_queue_ew + 1; }
						} else {
							if (v.approach_ns) { through_approach_ns <- through_approach_ns + 1; }
							else { through_approach_ew <- through_approach_ew + 1; }
						}
					} else if (v.maneuver = "left" and v.speed < 1 #km/#h) {
						if (v.approach_ns) { left_queue_ns <- left_queue_ns + 1; }
						else { left_queue_ew <- left_queue_ew + 1; }
					}
				} else if (r.is_outer_road and !r.is_incoming and v.lowest_lane >= 4 and v.lowest_lane < 12) {
					if (r.approach_id = 5 or r.approach_id = 7) { downstream_ns <- downstream_ns + 1; }
					else { downstream_ew <- downstream_ew + 1; }
				}
			}
		}
	}
}

experiment benchmark_server type: gui autorun: false {
	parameter "Demand CSV" var: demand_file_path;
	parameter "RNG seed" var: rng_seed;
	parameter "Python controlled" var: python_controlled init: true;
}
