import subprocess

subprocess.run("kadi-apy config create", shell=True)
subprocess.run("uvicorn app:app --host 0.0.0.0 --port 7860", shell=True)