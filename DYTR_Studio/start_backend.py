import os
import sys
import subprocess
import time
import urllib.request
from pathlib import Path

studio_dir = Path(__file__).resolve().parent
if not (studio_dir / "server" / "main.py").exists() and (studio_dir.parent / "server" / "main.py").exists():
    studio_dir = studio_dir.parent

dytr_candidates = [
    studio_dir.parent / "dytr" / "src",
    studio_dir / "dytr" / "src",
    Path.cwd() / "dytr" / "src",
    Path.cwd().parent / "dytr" / "src",
]
dytr_src = dytr_candidates[0]
for d in dytr_candidates:
    if d.exists():
        dytr_src = d
        break

conda_pl_candidates = [
    Path(r"C:\ProgramData\anaconda3\envs\pl\python.exe"),
    Path(r"C:\Users\M.D\anaconda3\envs\pl\python.exe"),
]
python_exe = str(conda_pl_candidates[0])
conda_env_dir = conda_pl_candidates[0].parent
for cand in conda_pl_candidates:
    if cand.exists():
        python_exe = str(cand)
        conda_env_dir = cand.parent
        break

# Kill any lingering process on port 8000
try:
    cmd_find = 'powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"'
    res = subprocess.check_output(cmd_find, shell=True, text=True).strip()
    for pid in res.split():
        if pid.isdigit() and int(pid) > 0:
            subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True)
            print(f"Freed port 8000 (terminated PID {pid})")
except Exception as e:
    pass

env = os.environ.copy()
if conda_env_dir:
    env["CONDA_PREFIX"] = str(conda_env_dir)
    conda_paths = [
        str(conda_env_dir),
        str(conda_env_dir / "Scripts"),
        str(conda_env_dir / "Library" / "bin"),
        str(conda_env_dir / "Library" / "usr" / "bin"),
        str(conda_env_dir / "Library" / "mingw-w64" / "bin"),
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(conda_paths + [existing_path])

env["PYTHONPATH"] = f"{dytr_src}{os.pathsep}{studio_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)

log_path = studio_dir / "backend_server.log"
log_file = open(log_path, "a", encoding="utf-8")

cmd = [
    python_exe,
    "-m",
    "uvicorn",
    "server.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--app-dir",
    str(studio_dir)
]

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

print("Launching detached FastAPI server on port 8000...")
proc = subprocess.Popen(
    cmd,
    env=env,
    cwd=str(studio_dir),
    stdout=log_file,
    stderr=subprocess.STDOUT,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
)

print(f"Spawned server process PID: {proc.pid}")

# Wait up to 6 seconds for server to be responsive
started = False
for _ in range(12):
    time.sleep(0.5)
    try:
        with urllib.request.urlopen("http://localhost:8000/api/v1/simulations/capabilities", timeout=1) as resp:
            if resp.status == 200:
                started = True
                break
    except Exception:
        pass

if started:
    print("SUCCESS: FastAPI server is actively listening on http://localhost:8000!")
else:
    print("WARNING: Server did not respond within timeout. Check backend_server.log")
