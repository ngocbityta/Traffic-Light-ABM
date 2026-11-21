import subprocess
subprocess.run("cd /content/gama && cp /content/bench/run.log results/ && "
               "tar czf /content/bench_results.tgz --exclude='training_state.pt' "
               "--exclude='checkpoint.pt' --exclude='demand' results", shell=True, check=True)
print("packed")
