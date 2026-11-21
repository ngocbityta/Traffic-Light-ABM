import os, subprocess
os.makedirs("/content/bench", exist_ok=True)
subprocess.Popen(["bash", "/content/gama_src/train/colab_bench/remote_bench.sh"],
                 stdout=open("/content/bench/run.log", "w"), stderr=subprocess.STDOUT,
                 cwd="/content", start_new_session=True)
print("bench started")
