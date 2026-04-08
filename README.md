---

title: Rapid Rescue AI Environment
emoji: 🚑
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
- openenv

---

# Rapid Rescue AI Environment

A real-world inspired environment where an AI agent must allocate ambulances efficiently across multiple emergency zones.

The goal is to maximize rescue effectiveness under resource constraints such as limited ambulances, varying severity, and distance.

---

## Quick Start

The environment simulates emergency response decisions.

Example usage:

```python
from models import MyAction
from server.my_env_environment import MyEnvironment

env = MyEnvironment()

result = env.reset()
print(result.echoed_message)

# Example interaction loop
for i in range(5):
    action = MyAction(message=str(i))  # choosing zone index
    result = env.step(action)
    print(result.echoed_message, result.reward)
```

---

## Building the Docker Image

```bash
docker build -t openenv-my -f server/Dockerfile .
```

---

## Deploying to Hugging Face Spaces

```bash
openenv push
```

After deployment, your space will be available at:

https://huggingface.co/spaces/<repo-id>

---

## Environment Details

### Action

**MyAction**

* `message` (str): zone index selected by the agent

---

### Observation

**MyObservation**

* `echoed_message` (str): current state of zones and ambulances
* `message_length` (int): not used for logic (legacy field)
* `reward` (float): reward obtained from the action
* `done` (bool): whether episode is finished
* `metadata` (dict): includes step count and additional info

---

## Reward Function

Reward is based on:

* ✅ Number of people rescued
* ✅ Higher severity zones give higher reward
* ❌ Distance penalty (longer travel reduces reward)
* ❌ Penalty for invalid actions or wasted moves

---

## Environment Dynamics

* Zones worsen over time (people increase, severity rises)
* Ambulances become unavailable after dispatch
* Agent must balance urgency vs distance
* Limited ambulances introduce resource constraints

---

## Advanced Usage

### Connecting to Deployed Server

```python
from models import MyAction
from client import MyEnv

env = MyEnv(base_url="<ENV_HTTP_URL>")

result = env.reset()
result = env.step(MyAction(message="0"))
```

---

## Development & Testing

### Run Locally

```bash
uvicorn server.app:app --reload
```

---

## Project Structure

```
my_env/
├── README.md
├── openenv.yaml
├── inference.py
├── models.py
├── client.py
└── server/
    ├── app.py
    ├── my_env_environment.py
    └── Dockerfile
```

---

## Real-World Relevance

This environment models:

* Ambulance allocation systems
* Disaster response optimization
* Resource-constrained decision making

---

## Notes

Designed to evaluate AI agents in dynamic, high-stakes environments with trade-offs between speed, priority, and resource availability.
