---
app_port: 8000
base_path: /web
colorFrom: blue
colorTo: red
emoji: 🚑
pinned: false
sdk: docker
tags:
- openenv
- reinforcement-learning
- emergency-dispatch
- road-accident
- llm-agent
title: Rapid Rescue AI Environment
---

# 🚑 Rapid Rescue — Road Accident Dispatch Environment

An OpenEnv reinforcement learning environment where an AI agent acts as
an emergency dispatch controller, making real-time decisions about which
road accident zones to send ambulances to.

Built for the **Meta PyTorch Hackathon x Scaler School of Technology**,
this environment is directly inspired by a real accident detection
system that uses **YOLOv8 + OpenCV** to monitor CCTV footage and
automatically detect road accidents — this environment provides the
intelligent dispatch layer that decides *what to do* after an accident
is detected.

---

## 🌍 Real-World Motivation

Road accidents are one of the leading causes of preventable deaths
worldwide. The first hour after a road accident — often called the
**Golden Hour** — is the most critical window for saving lives. Every
minute of delay in emergency dispatch directly impacts survival outcomes.

Current emergency response systems rely heavily on human operators to
monitor feeds, assess severity, and decide which units to dispatch.
While human judgment is valuable, it is prone to errors under pressure —
missed alerts, incorrect prioritization, and delayed decisions are
common in high-stress, multi-incident scenarios.

Automating this process end-to-end, from detection to dispatch, removes
the bottleneck of human reaction time and ensures optimal resource
allocation even when multiple incidents occur simultaneously.

Research into existing smart city infrastructure in India reveals a
critical gap: most AI-equipped CCTV systems currently deployed only
detect traffic violations such as helmet non-compliance, overspeeding,
and seatbelt violations. Accident detection and emergency dispatch
optimization remain largely unaddressed by existing deployments.

To address this, a prior project was built that acts as a layer on top
of existing CCTV infrastructure — processing live footage server-side
to detect road accidents using YOLOv8 + OpenCV, assess their severity,
and automatically dispatch alerts to emergency services via Telegram,
including the accident photo, location, and risk level.

This environment builds on that foundation by replacing the hardcoded
dispatch logic in that system with a trained AI agent that learns to
make optimal dispatch decisions — accounting for casualty count,
severity, distance, and ambulance availability — in real time.

---

## 🔗 Connection to Accident Detection System

This environment is the dispatch intelligence layer for a broader
end-to-end emergency response pipeline:

```
CCTV Feed
    ↓
YOLOv8 + OpenCV (accident detection & severity assessment)
    ↓
Zone Data: {casualties, severity, location}
    ↓
Rapid Rescue Environment (this project)
    ↓
RL Agent Dispatch Decision: "Send Ambulance 2 to Zone 3"
    ↓
Emergency Alert with optimized dispatch info
```

The detection system was built during a hackathon at IIIT Kottayam,
using YOLOv8n to classify accident severity from live CCTV footage and
dispatch alerts via Telegram. This environment replaces the hardcoded
dispatch logic with a learned optimal policy.

The integration is a work in progress — the current environment uses
simulated zone data, with live detection pipeline integration planned
as future work.

---

## 🎮 Environment Overview

### Accident Zone Types

Each zone represents a real road accident scenario:

| Type | Severity Modifier | Description |
|---|:---:|---|
| `minor_collision` | +0.0 | Low impact, minimal injuries |
| `intersection_crash` | +0.2 | Moderate impact at junction |
| `major_collision` | +0.3 | High impact, multiple injuries |
| `highway_accident` | +0.4 | High speed, critical injuries |
| `multi_vehicle_pileup` | +0.5 | Mass casualty risk |

### Task Difficulty Levels

| Task | Zones | Ambulances | Severity Range | Worsening Rate | Max Steps |
|---|:---:|:---:|:---:|:---:|:---:|
| `easy` | 3 | 3 | 3–5 | 0.05/step | 8 |
| `medium` | 4 | 3 | 2–4 | 0.15/step | 9 |
| `hard` | 5 | 3 | 2–5 | 0.35/step | 10 |

### Ambulances

- Each ambulance has a speed multiplier (0.8–1.2×)
- Faster ambulances return sooner from dispatch
- Agent implicitly learns to prefer faster ambulances

---

## 🧠 Agent Behavior

The agent receives the full state as a formatted string:

```
=== Road Accident Dispatch | Task: HARD | Step: 3/10 ===

ACCIDENT ZONES:
  Zone 0 [highway_accident]: CLEARED
  Zone 1 [multi_vehicle_pileup]: HIGH | casualties=15, severity=3.5/5.0, distance=3km
  Zone 2 [major_collision]: CRITICAL | casualties=13, severity=4.5/5.0, distance=10km
  Zone 3 [major_collision]: CRITICAL | casualties=11, severity=4.5/5.0, distance=7km
  Zone 4 [major_collision]: CRITICAL | casualties=14, severity=5.0/5.0, distance=3km

AMBULANCES:
  Ambulance 0 [speed=0.9]: AVAILABLE
  Ambulance 1 [speed=0.9]: EN ROUTE (returns in 1 steps)
  Ambulance 2 [speed=1.0]: AVAILABLE

Total casualties rescued this episode: 9
```

The agent returns a single zone index. A well-trained agent learns to:

- Prioritize CRITICAL zones over MODERATE
- Prefer closer zones when severity is equal
- Avoid dispatching to CLEARED zones
- Manage ambulance availability across steps

---

## 🏆 Reward Design

| Event | Reward |
|---|:---:|
| Casualties saved × severity × 2 | positive |
| Critical zone bonus (severity ≥ 4.0) | +2.0 |
| First response bonus (new zone) | +1.0 |
| Distance penalty | −0.3 × km |
| Zone reaches max severity (5.0) | −1.0 |
| Wasted dispatch (empty zone) | −1.5 |
| No ambulance available | −2.0 |
| All zones cleared early | +3.0 + 0.5 × steps remaining |

All rewards are normalized to **[0.0, 1.0]** range and fire every step,
providing a dense signal throughout the episode rather than sparse
end-of-episode feedback.

---

## 📊 Grading

Each task uses a `python` grader with task-specific logic that scores
episode rewards on a 0.0–1.0 scale:

| Task | Grader Logic |
|---|---|
| `easy` | Pure average — rewards consistent performance across all steps |
| `medium` | Weighted average (70%) + peak reward (30%) — rewards consistency and peak decision quality |
| `hard` | Average minus wasted step penalty — penalizes failed dispatches under pressure |

---

## 📈 Sample Results

Running the baseline LLM agent (Qwen2.5-72B-Instruct):

```
[START] task=easy   → score=0.867
[START] task=medium → score=0.807
[START] task=hard   → score=0.534
```

The score ordering reflects genuine difficulty progression. Hard mode
demonstrates the resource pressure — the agent must triage 5 rapidly
deteriorating zones with limited ambulances and high distance penalties,
resulting in some unavoidable wasted steps.

---

## 🚀 Quick Start

### Run the Demo

```bash
python3 demo.py
```

Runs a full demonstration across all three difficulty levels using a
priority-based dispatch agent. No API keys required.
oad accidents are one of the leading causes of preventable deaths
worldwide. The first hour after a road accident — often called the
**Golden Hour** — is the most critical window for saving lives. Every
minute of delay in emergency dispatch directly impacts survival outcomes.

Current emergency response systems rely heavily on human operators to
monitor feeds, assess severity, and decide which units to dispatch.
While human judgment is valuable, it is prone to errors under pressure —
missed alerts, incorrect prioritization, and delayed decisions are
common in high-stress, multi-incident scenarios.

Automating this process end-to-end, from detection to dispatch, removes
the bottleneck of human reaction time and ensures optimal resource
allocation even when multiple incidents occur simultaneously.

Research into existing smart city infrastructure in India reveals a
critical gap: most AI-equipped CCTV systems currently deployed only
detect traffic violations such as helmet non-compliance, overspeeding,
and seatbelt violations. Accident detection and emergency dispatch
optimization remain largely unaddressed by existing deployments.

To address this, a prior project was built that acts as a layer on top
of existing CCTV infrastructure — processing live footage server-side
to detect road accidents using YOLOv8 + OpenCV, assess their severity,
and automatically dispatch alerts to emergency services via Telegram,
including the accident photo, location, and risk level.

This environment builds on that foundation by replacing the hardcoded
dispatch logic in that system with a trained AI agent that learns to
make optimal dispatch decisions — accounting for casualty count,
severity, distance, and ambulance availability — in real time.
### Run the Baseline LLM Agent

```bash
# Set environment variables first
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python3 inference.py
```

### Use Directly in Python

```python
# Run from project root after: uv sync or pip install -e .
from models import MyAction
from server.my_env_environment import MyEnvironment

env = MyEnvironment()

# Run easy episode
result = env.reset(task_id="easy")
print(result.echoed_message)

for step in range(5):
    action = MyAction(message="0")  # dispatch to zone 0
    result = env.step(action)
    print(f"Reward: {result.reward:.3f} | Done: {result.done}")
```

### Connect to Deployed Server

```python
# Run from project root after: uv sync or pip install -e .
from client import MyEnv
from models import MyAction

env = MyEnv(base_url="https://rvhul-my-env.hf.space")
result = env.reset(task_id="hard")
result = env.step(MyAction(message="2"))
```

---

## 🛠️ Local Development

```bash
# Install dependencies
uv sync

# Run server locally
uvicorn server.app:app --reload --port 8000

# Run demo (no API key needed)
python3 demo.py

# Run baseline inference
python3 inference.py

# Validate OpenEnv spec
openenv validate
```

---

## 📁 Project Structure

```
my_env/
├── README.md                    # Documentation
├── openenv.yaml                 # OpenEnv spec (tasks + graders)
├── Dockerfile                   # Container configuration
├── inference.py                 # Baseline LLM agent (Qwen2.5-72B)
├── demo.py                      # Standalone demo, no API key needed
├── grader.py                    # Task grading functions (easy/medium/hard)
├── models.py                    # Action + Observation models
├── client.py                    # HTTP client for remote environment
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package configuration
└── server/
    ├── app.py                   # FastAPI server entry point
    ├── my_env_environment.py    # Core environment logic
    └── Dockerfile               # Container configuration
```

---

## 📋 OpenEnv Spec

```yaml
tasks:
  - id: easy    # 3 zones, close distances, forgiving conditions
  - id: medium  # 4 zones, moderate pressure and deterioration
  - id: hard    # 5 zones, far distances, rapid deterioration
```

---

## 🔮 Future Work

- Integrate live zone data from YOLOv8 accident detection pipeline
- Train a dedicated RL policy (PPO/DQN) to replace LLM agent
- Add fire detection to the accident detection system using models from Hugging Face
- Add fire engine and police dispatch as additional action types
- Use latitude and longitude of specific CCTV cameras to provide Google Maps location to emergency services
- Multi-agent coordination between ambulances

---

*Built with OpenEnv · Powered by Meta PyTorch ·
Inspired by real-world CCTV accident detection*
