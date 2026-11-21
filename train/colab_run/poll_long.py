import glob, os
log = open("/content/long.log").read().splitlines() if os.path.exists("/content/long.log") else []
print("\n".join(log[-4:]))
for path in glob.glob("/content/runs/*/console.log"):
    rows = [l for l in open(path).read().splitlines() if l.startswith("[Rollout")]
    if rows:
        print(rows[-1][:110])
if os.path.exists("/content/sync.log"):
    sync = open("/content/sync.log").read().splitlines()
    if sync:
        print(sync[-1])
if any(l.startswith("LONG_DONE") for l in log):
    print("LONG_DONE")
