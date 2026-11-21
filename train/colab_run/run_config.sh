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
            --validate-every 250)
