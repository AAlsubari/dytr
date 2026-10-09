# Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
# FastAPI Router for DyTrStudio Problem Framing & Scientific Diagnostics
# Decouples static domain data and provides RESTful endpoints

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/problem", tags=["problem"])

DB_PATH = Path(__file__).resolve().parents[2] / "mock-server" / "db.json"

def get_db():
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="Mock database db.json not found")
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read db.json: {str(e)}")

@router.get("/scenarios")
async def get_scenarios():
    """Retrieve pre-formulated problem scenario templates."""
    db = get_db()
    return db.get("problem_scenarios", [])

@router.get("/diagnostic-questions")
async def get_diagnostic_questions():
    """Retrieve the AI necessity diagnostic question definitions."""
    db = get_db()
    return db.get("diagnostic_questions", [])

@router.get("/task-types")
async def get_task_types():
    """Retrieve supported DyTR neural network task architectures."""
    db = get_db()
    return db.get("task_types", [])

@router.get("/foundation-models")
async def get_foundation_models():
    """Retrieve foundation backbone choices for hypothesis framing."""
    db = get_db()
    return db.get("foundation_models", [])
