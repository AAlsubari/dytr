# Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
# Unified Launcher for DyTrStudio (FastAPI Backend + React Frontend)

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
    if (current / "DyTrStudio").exists():
        return current / "DyTrStudio"
    return current

STUDIO_DIR = find_studio_dir()

def main():
    print("=" * 70)
    print("  DyTrStudio: Interactive Educational & Continual Learning Platform")
    print("  Powered by DyTR Core Library")
    print(f"  Working Directory: {STUDIO_DIR}")
    print("=" * 70)

    # 1. Start FastAPI Backend Server
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

    print(f"\n[1/2] Launching FastAPI Backend with {python_exe} on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(STUDIO_DIR), env=env)

    # 2. Start Vite Frontend Dev Server
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    print("[2/2] Launching React Vite Frontend Dev Server ...")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(STUDIO_DIR))

    print("\nDyTrStudio is running!")
    print("  Frontend UI: http://localhost:5173")
    print("  Backend API: http://localhost:8000/docs")
    print("  Press Ctrl+C to terminate both servers.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down DyTrStudio servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
