# Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
# Unified Launcher for DYTR Studio (FastAPI Backend + React Frontend)

import os
import sys
import subprocess
import time
from pathlib import Path

def find_studio_dir() -> Path:
    current = Path(__file__).resolve().parent
    if (current / "server" / "main.py").exists():
        return current
    if (current / "main.py").exists() and current.name == "server":
        return current.parent
    if (current / "DYTR_Studio").exists():
        return current / "DYTR_Studio"
    if (current / "DyTrStudio").exists():
        return current / "DyTrStudio"
    return current

def kill_port(port: int):
    """Ensure port is freed before launching."""
    if os.name == "nt":
        try:
            cmd = f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"'
            output = subprocess.check_output(cmd, shell=True, text=True).strip()
            if output:
                for pid in output.split():
                    if pid.isdigit() and int(pid) > 0:
                        subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True)
                        print(f"  Freed port {port} (terminated PID {pid})")
        except Exception:
            pass

STUDIO_DIR = find_studio_dir()

def main():
    print("=" * 70)
    print("  DYTR Studio: Interactive Educational & Continual Learning Platform")
    print("  Powered by DyTR Core Library")
    print(f"  Working Directory: {STUDIO_DIR}")
    print("=" * 70)

    # Free ports from any dangling old server processes
    kill_port(8000)
    kill_port(5173)

    # 1. Start FastAPI Backend Server
    # Dynamic & Portable Python Interpreter Resolution:
    # 1) Specific Conda 'pl' paths (if available on local Windows installation)
    # 2) Active Conda environment ($CONDA_PREFIX)
    # 3) Current Python runtime (sys.executable)
    # 4) Fallback to system python
    conda_pl_candidates = [
        Path(r"C:\ProgramData\anaconda3\envs\pl\python.exe"),
        Path(r"C:\Users\M.D\anaconda3\envs\pl\python.exe"),
        Path.home() / "anaconda3" / "envs" / "pl" / ("python.exe" if os.name == "nt" else "bin/python"),
        Path.home() / "miniconda3" / "envs" / "pl" / ("python.exe" if os.name == "nt" else "bin/python"),
    ]

    python_exe = None
    conda_env_dir = None
    for cand in conda_pl_candidates:
        if cand.exists():
            python_exe = str(cand)
            conda_env_dir = cand.parent if cand.parent.name != "bin" else cand.parent.parent
            break

    # If conda env is active in environment
    if not python_exe and os.environ.get("CONDA_PREFIX"):
        active_conda = Path(os.environ["CONDA_PREFIX"])
        active_py = active_conda / ("python.exe" if os.name == "nt" else "bin/python")
        if active_py.exists():
            python_exe = str(active_py)
            conda_env_dir = active_conda

    # Default fallback to current executable
    if not python_exe:
        if conda_pl_candidates[0].exists() or conda_pl_candidates[1].exists():
            python_exe = str(conda_pl_candidates[0] if conda_pl_candidates[0].exists() else conda_pl_candidates[1])
            conda_env_dir = Path(python_exe).parent
        else:
            python_exe = sys.executable

    print(f"  Using Python Interpreter: {python_exe}")
    backend_cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--app-dir",
        str(STUDIO_DIR),
        "--reload",
    ]
    env = os.environ.copy()

    # Prepend Conda environment binaries to PATH to ensure PyTorch and CUDA C++ extensions resolve
    if conda_env_dir and Path(conda_env_dir).exists():
        env["CONDA_PREFIX"] = str(conda_env_dir)
        conda_paths = [
            str(conda_env_dir),
            str(conda_env_dir / "Scripts"),
            str(conda_env_dir / "bin"),
            str(conda_env_dir / "Library" / "bin"),
            str(conda_env_dir / "Library" / "usr" / "bin"),
            str(conda_env_dir / "Library" / "mingw-w64" / "bin"),
        ]
        valid_conda_paths = [p for p in conda_paths if Path(p).exists()]
        existing_path = env.get("PATH", "")
        if valid_conda_paths:
            env["PATH"] = os.pathsep.join(valid_conda_paths + [existing_path])

    # Dynamic & Portable dytr library location candidates
    dytr_candidates = [
        STUDIO_DIR.parent / "dytr" / "src",
        STUDIO_DIR / "dytr" / "src",
        Path.cwd() / "dytr" / "src",
        Path.cwd().parent / "dytr" / "src",
    ]
    dytr_src = None
    for d in dytr_candidates:
        if d.exists():
            dytr_src = str(d)
            break
    if not dytr_src:
        dytr_src = str(dytr_candidates[0])

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{dytr_src}{os.pathsep}{str(STUDIO_DIR)}{os.pathsep}{existing_pythonpath}".strip(os.pathsep)

    print("\n[1/2] Launching FastAPI Backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(STUDIO_DIR), env=env)

    # 2. Start Vite Frontend Dev Server
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    print("[2/2] Launching React Vite Frontend Dev Server ...")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(STUDIO_DIR))

    print("\nDYTR Studio is running!")
    print("  Frontend UI: http://localhost:5173")
    print("  Backend API: http://localhost:8000/docs")
    print("  Press Ctrl+C to terminate both servers.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down DYTR Studio servers...")
        if os.name == "nt":
            subprocess.run(f"taskkill /F /T /PID {backend_proc.pid}", shell=True, capture_output=True)
            subprocess.run(f"taskkill /F /T /PID {frontend_proc.pid}", shell=True, capture_output=True)
        else:
            backend_proc.terminate()
            frontend_proc.terminate()
        print("Shutdown complete.")
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
