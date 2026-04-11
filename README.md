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

# 🚑 Rapid Rescue --- Road Accident Dispatch Environment

An OpenEnv reinforcement learning environment where an AI agent acts as
an emergency dispatch controller, making real-time decisions about which
road accident zones to send ambulances to.

Built for the **Meta PyTorch Hackathon x Scaler School of Technology**,
this environment is directly inspired by a real accident detection
system that uses **YOLOv8 + OpenCV** to monitor CCTV footage and
automatically detect road accidents --- this environment provides the
intelligent dispatch layer that decides *what to do* after an accident
is detected.

------------------------------------------------------------------------

## 🌍 Real-World Motivation

Modern accident detection systems can identify incidents on CCTV feeds
and assess their severity. But the dispatch decision --- *which
ambulance to send, to which zone, in what order* --- is still largely
manual or rule-based.

This environment trains an AI agent to make those dispatch decisions
optimally, considering: - How many casualties are at each zone -
Severity of each accident (minor collision → multi-vehicle pileup) -
Distance from available ambulances - Resource constraints (limited
ambulances, travel time) - Dynamic deterioration --- zones worsen every
step if unattended

The agent learns that a critical highway accident 4km away may need
priority over a moderate collision 1km away, and that wasting an
ambulance on an already-cleared zone has real consequences.

------------------------------------------------------------------------

## 🔗 Connection to Accident Detection System

This environment is the dispatch intelligence layer for a broader
end-to-end emergency response pipeline:

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

The detection system was built by me during a hackathon at IIIT
Kottayam, using YOLOv8n to classify accident severity from live CCTV
footage and dispatch alerts via Telegram. This environment replaces the
hardcoded dispatch logic with a learned optimal policy.

------------------------------------------------------------------------

## 🎮 Environment Overview

### Accident Zone Types

Each zone represents a real road accident scenario:

  Type                      Severity Modifier  Description
  ------------------------ ------------------- --------------------------------
  `minor_collision`               +0.0         Low impact, minimal injuries
  `intersection_crash`            +0.2         Moderate impact at junction
  `major_collision`               +0.3         High impact, multiple injuries
  `highway_accident`              +0.4         High speed, critical injuries
  `multi_vehicle_pileup`          +0.5         Mass casualty risk

### Task Difficulty Levels

  Task        Zones   Ambulances   Severity   Worsening Rate   Max Steps
  ---------- ------- ------------ ---------- ---------------- -----------
  `easy`        3         3          1--2       0.10/step          8
  `medium`      4         3          2--4       0.15/step          9
  `hard`        5         3          3--5       0.25/step         10

### Ambulances

-   Each ambulance has a speed multiplier (0.8--1.2×)
-   Faster ambulances return sooner from dispatch
-   Agent implicitly learns to prefer faster ambulances

------------------------------------------------------------------------

## 🧠 Agent Behavior

The agent receives the full state as a formatted string:

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

The agent returns a single zone index. A well-trained agent learns to: -
Prioritize CRITICAL zones over MODERATE - Prefer closer zones when
severity is equal - Avoid dispatching to CLEARED zones - Manage
ambulance availability across steps

------------------------------------------------------------------------

## 🏆 Reward Design

  Event                                              Reward
  -------------------------------------- ------------------------------
  Casualties saved × severity × 2                   positive
  Critical zone bonus (severity ≥ 4.0)                +2.0
  First response bonus (new zone)                     +1.0
  Distance penalty                                 −0.3 × km
  Zone reaches max severity (5.0)                     −1.0
  Wasted dispatch (empty zone)                        −1.5
  No ambulance available                              −2.0
  All zones cleared early                 +3.0 + 0.5 × steps remaining

All rewards are normalized to **\[0.0, 1.0\]** range.

------------------------------------------------------------------------

## 📊 Sample Results

Running the baseline LLM agent (Qwen2.5-72B-Instruct): \[START\]
task=easy → score=0.359 \[START\] task=medium → score=0.807 \[START\]
task=hard → score=0.798

Hard mode demonstrates intelligent multi-zone prioritization --- the
agent correctly skips low-severity zones and focuses ambulances on
critical accidents first.

------------------------------------------------------------------------

## 🚀 Quick Start

``` python
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

``` python
from client import MyEnv
from models import MyAction

env = MyEnv(base_url="https://rvhul-my-env.hf.space")
result = env.reset(task_id="hard")
result = env.step(MyAction(message="2"))
```

------------------------------------------------------------------------

## 🛠️ Local Development

``` bash
# Install dependencies
uv sync

# Run server locally
uvicorn server.app:app --reload --port 8000

# Run baseline inference
python3 inference.py

# Validate OpenEnv spec
openenv validate
```

------------------------------------------------------------------------

## 📁 Project Structure

my_env/ ├── README.md \# This file ├── openenv.yaml \# OpenEnv spec
(tasks + graders) ├── inference.py \# Baseline LLM agent ├── grader.py
\# Task grading functions ├── models.py \# Action + Observation models
├── client.py \# HTTP client for remote env └── server/ ├── app.py \#
FastAPI server ├── my_env_environment.py \# Core environment logic └──
Dockerfile

------------------------------------------------------------------------

## 📋 OpenEnv Spec

``` yaml
tasks:
  - id: easy    # 3 zones, forgiving conditions
  - id: medium  # 4 zones, moderate pressure
  - id: hard    # 5 zones, rapid deterioration
```

Each task uses a `python` grader that scores episode rewards on a
0.0--1.0 scale.

------------------------------------------------------------------------

## 🔮 Future Work

-   Integrate live zone data from YOLOv8 accident detection pipeline
-   Train a dedicated RL policy (PPO/DQN) to replace LLM agent
-   Add fire detection to the accident detection system using models from hugging face
-   Add fire engine and police dispatch as additional action types
-   Use lattitude and longitude of the specific cctv cameras to give google maps location to emergency services
-   Multi-agent coordination between ambulances

------------------------------------------------------------------------
