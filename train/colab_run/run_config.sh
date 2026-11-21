# Settings shared by the G4 launcher, the Colab side and the Mac launcher.
# Override any of them in the environment, e.g. G4_HOURS=6.5.
RUN_NAME=${RUN_NAME:-rainbow_r1_5m}
TOTAL_STEPS=${TOTAL_STEPS:-5000000}   # must stay the same for the whole run
G4_HOURS=${G4_HOURS:-7.0}             # training time per G4 session (CU ~8.7/h)
G4_WORKERS=${G4_WORKERS:-46}
G4_DEVICE=${G4_DEVICE:-cuda}
MAC_WORKERS=${MAC_WORKERS:-6}
MAC_HOURS=${MAC_HOURS:-168}           # the Mac stops after this or at TOTAL_STEPS
SYNC_MINUTES=${SYNC_MINUTES:-15}      # Colab -> Google Drive backup
DOWNLOAD_MINUTES=${DOWNLOAD_MINUTES:-60}  # Colab -> this Mac backup
DRIVE_SUBDIR="Traffic-Light-ABM/runs"
# Learning settings: identical on every machine (a resume refuses changes).
TRAIN_ARGS=(--scenario lvl --backend python --algorithm rainbow --reward-mode r1
            --rollout-mode async --learner thread --rollout-decisions 16
            --episode-horizon 8000 --total-steps "$TOTAL_STEPS" --warmup 2000
            --validate-every 250
            # Demand: each episode draws 50-130 % of 2400 veh/h per arm and a
            # 0-20 % heavier Hoang Dao Thuy axis (NW-SE), fixed for the episode.
            --demand-scale-min 0.5 --demand-scale-max 1.3 --axis-imbalance-max 0.2
            # Heavy levels reach returns near -7000: widen C51, same 40-unit atoms.
            --c51-v-min -8000 --c51-atoms 201
            # Queue counts reach ~300 per axis at 1.3x (linear clips at 100): log scale.
            --queue-feature-scale log)
