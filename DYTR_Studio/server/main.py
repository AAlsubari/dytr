# Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FastAPI Server Entry Point for DyTrStudio.

Author: Jamal Al-Mansoub
Email: Jamalalmansoub@gmail.com
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure server package and parent directory are on sys.path
SERVER_DIR = Path(__file__).resolve().parent
STUDIO_DIR = SERVER_DIR.parent
PROJECT_ROOT = STUDIO_DIR.parent
DYTR_SRC = PROJECT_ROOT / "dytr" / "src"

for path_entry in [str(DYTR_SRC), str(STUDIO_DIR), str(SERVER_DIR), str(PROJECT_ROOT)]:
    if path_entry not in sys.path and Path(path_entry).exists():
        sys.path.insert(0, path_entry)

from server.router.simulations import router as simulations_router
from server.router.models import router as models_router
from server.router.datasets import router as datasets_router
from server.router.training import router as training_router
from server.router.inference import router as inference_router
from server.router.education import router as education_router
from server.router.evaluation import router as evaluation_router
from server.router.deployments import router as deployments_router
from server.router.problem import router as problem_router
from server.services.dytr_bridge import DYTR_AVAILABLE, DYTR_IMPORT_ERROR

app = FastAPI(
    title="DYTR Studio API",
    description="Backend services for DYTR Studio: Interactive Visual & Continual Learning Platform for DyTR.",
    version="0.1.3",
)

# Enable CORS for React Vite frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(simulations_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")
app.include_router(training_router, prefix="/api/v1")
app.include_router(inference_router, prefix="/api/v1")
app.include_router(education_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(deployments_router, prefix="/api/v1")
app.include_router(problem_router, prefix="/api/v1")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint with dynamic dytr library status and diagnostics."""
    import sys
    import traceback
    from pathlib import Path

    current_file = Path(__file__).resolve()
    # Ensure dytr/src is in sys.path dynamically and portably across any environment
    candidates = [
        current_file.parents[1] / "dytr" / "src",
        current_file.parents[2] / "dytr" / "src",
        Path.cwd() / "dytr" / "src",
        Path.cwd().parent / "dytr" / "src",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    is_available = False
    error_msg = ""
    try:
        # pyrefly: ignore [missing-import]
        import dytr
        # pyrefly: ignore [missing-import]
        from dytr.core.model import DynamicTransformer
        is_available = True
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"

    return {
        "status": "online",
        "service": "DYTR Studio Backend",
        "version": "0.1.3",
        "author": "Jamal Al-Mansoub",
        "dytr_available": is_available,
        "engine_mode": "live" if is_available else "simulation",
        "dytr_error": error_msg or "None (DyTR Core Loaded)",
        "python_interpreter": sys.executable,
    }


if __name__ == "__main__":
    import uvicorn
    # Use app_dir to guarantee uvicorn locates server.main regardless of invocation CWD
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(STUDIO_DIR))

