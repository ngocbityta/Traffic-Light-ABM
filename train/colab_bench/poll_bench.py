import glob, os
log = open("/content/bench/run.log").read().splitlines() if os.path.exists("/content/bench/run.log") else []
print("\n".join(log[-6:]))
for path in sorted(glob.glob("/content/gama/results/*.console.log")):
    lines = [l for l in open(path).read().splitlines() if l.startswith("[Rollout")]
    if lines:
        print(os.path.basename(path), lines[-1][:90])
if any(l.startswith("BENCH_DONE") for l in log):
    print("BENCH_DONE")
