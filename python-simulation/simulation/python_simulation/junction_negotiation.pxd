# Native field layout of the negotiation records (read ~10^5 times a tick).
cdef class Zone:
    cdef public double a_in
    cdef public double a_out
    cdef public double b_in
    cdef public double b_out
    cdef public object kind

cdef class Agent:
    cdef public object vid
    cdef public object movement
    cdef public object arm
    cdef public object turn
    cdef public double front
    cdef public double length
    cdef public double speed
    cdef public double accel
    cdef public double decel
    cdef public double max_decel
    cdef public double min_gap
    cdef public double desired
    cdef public bint motorbike
    cdef public bint on_junction
    cdef public bint arm_green
    cdef public double wait_time
    cdef public object group_size
    cdef public object group_moto
    cdef public object road
    cdef public double hold_time
    cdef public object commit_time
    cdef public bint passive
    cdef public bint stalled
    cdef public double lateral_shift
    cdef public double half_width
    cdef public double connector_shift
    cdef public object connector_span
    cdef public object xy
    cdef public object heading
    cdef public object units
