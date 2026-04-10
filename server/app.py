# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the My Env Environment.

This module creates an HTTP server that exposes the MyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import MyAction, MyObservation
    from .my_env_environment import MyEnvironment
except ModuleNotFoundError:
    from models import MyAction, MyObservation
    from server.my_env_environment import MyEnvironment


# Create the app with web interface and README integration
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="my_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# At the bottom of server/app.py, after app = create_app(...)

from fastapi.responses import JSONResponse
from fastapi import Request

@app.get("/tasks")
async def get_tasks():
    return JSONResponse([
        {"id": "easy", "description": "Low complexity emergency scenarios", "max_steps": 5, "has_grader": True},
        {"id": "medium", "description": "Moderate emergency scenarios with trade-offs", "max_steps": 7, "has_grader": True},
        {"id": "hard", "description": "High complexity emergency scenarios with severe conditions", "max_steps": 10, "has_grader": True}
    ])

@app.post("/grader")
async def grade(request: Request):
    body = await request.json()
    reward = body.get("reward", 0.0)
    score = max(0.0, min(1.0, float(reward)))
    return JSONResponse({"score": score, "has_grader": True})

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
