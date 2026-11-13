/**
* Name: surrogate_safety_measures
* Description: Surrogate Safety Measures library.
* Note: Core SSM definitions and conflict zones are integrated into collision_avoidance.gaml.
*/
model surrogate_safety_measures

import "collision_avoidance.gaml"

global {
}

experiment test_ssm type: gui {
	output {
		display view background: rgb(18, 22, 28) {
			species road aspect: base;
			species traffic_signal_post aspect: base;
			species conflict_zone aspect: base;
			species base_vehicle_species aspect: base;
		}
	}
}
