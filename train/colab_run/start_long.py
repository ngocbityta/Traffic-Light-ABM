import os, subprocess
for flag in ("/content/TRAIN_DONE",):
    if os.path.exists(flag):
        os.remove(flag)
subprocess.run("mkdir -p /content/gama_src && tar xzf /content/gama_src.tgz -C /content/gama_src",
               shell=True, check=True)
subprocess.Popen(["bash", "/content/gama_src/train/colab_run/remote_long.sh"],
                 stdout=open("/content/long.log", "w"), stderr=subprocess.STDOUT,
                 cwd="/content", start_new_session=True)
print("long run started")
