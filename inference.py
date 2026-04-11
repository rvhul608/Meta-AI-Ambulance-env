import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from server.my_env_environment import MyEnvironment
from models import MyAction

load_dotenv()

# ---------------- ENV ---------------- #

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("MY_ENV_TASK", "rescue")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "my_env")

MAX_STEPS = 10
TEMPERATURE = 0.3
MAX_TOKENS = 50
SUCCESS_SCORE_THRESHOLD = 0.2

# ---------------- PROMPT ---------------- #

SYSTEM_PROMPT = textwrap.dedent("""
You are an emergency road accident dispatch system.

You are given accident zones with:
- casualty count (people needing help)
- severity (1.0-5.0, higher is worse)
- distance in km (lower is faster response)
- accident type

You also have ambulances with availability status.

Your job: choose ONE zone index to dispatch an available ambulance to.

Rules:
- Return ONLY a single integer (the zone index)
- Only choose zones where people > 0 (not CLEARED)
- Only dispatch if at least one ambulance is AVAILABLE
- Prioritize: highest severity first, then most casualties, then closest distance
- Consider ALL zones including the highest numbered ones
""").strip()

# ---------------- LOGGING ---------------- #

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------- MODEL ---------------- #

def get_action(client, state, last_reward):
    try:
        # Parse all zones from state
        active_zones = []
        lines = state.split("\n")
        for line in lines:
            if "Zone " in line and "CLEARED" not in line and "casualties=" in line:
                try:
                    idx = int(line.strip().split("Zone ")[1].split(" ")[0])
                    severity = float(line.split("severity=")[1].split("/")[0])
                    casualties = int(line.split("casualties=")[1].split(",")[0])
                    distance = int(line.split("distance=")[1].replace("km","").strip())
                    active_zones.append((idx, severity, casualties, distance))
                except:
                    continue

        if not active_zones:
            return "0"

        # Check if any ambulance is available
        ambulances_available = "AVAILABLE" in state

        if not ambulances_available:
            # All busy — still need to return something, pick best zone for next step
            active_zones.sort(key=lambda x: (x[1], x[2], -x[3]), reverse=True)
            return str(active_zones[0][0])

        # Sort by: severity desc, casualties desc, distance asc
        active_zones.sort(key=lambda x: (x[1], x[2], -x[3]), reverse=True)
        best_zone = str(active_zones[0][0])

        # Ask LLM with clear context
        zone_summary = ", ".join([
            f"Zone {z[0]}(sev={z[1]:.1f},cas={z[2]},dist={z[3]}km)"
            for z in active_zones
        ])

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"{state}\n"
                    f"Last reward: {last_reward:.2f}\n"
                    f"Active zones: {zone_summary}\n"
                    f"Best zone by priority: {best_zone}\n"
                    f"Reply with ONLY a single digit zone index:"
                )},
            ],
            temperature=0.0,
            max_tokens=5,
        )

        text = response.choices[0].message.content.strip()
        # Extract first digit found
        for char in text:
            if char.isdigit():
                zone_ids = [str(z[0]) for z in active_zones]
                if char in zone_ids:
                    return char

        # Fallback to best computed zone
        return best_zone

    except Exception:
        # Even on exception, try to return best zone not just "0"
        try:
            active_zones.sort(key=lambda x: (x[1], x[2], -x[3]), reverse=True)
            return str(active_zones[0][0])
        except:
            return "0"

# ---------------- MAIN ---------------- #
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MyEnvironment()

    all_rewards = []
    all_steps = 0

    for task_id in ["easy", "medium", "hard"]:
        rewards = []
        steps_taken = 0

        log_start(task_id, BENCHMARK, MODEL_NAME)

        try:
            result = env.reset(task_id=task_id)
            last_state = result.echoed_message
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action = get_action(client, last_state, last_reward)
                result = env.step(MyAction(message=action))

                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step

                log_step(step, action, reward, done, None)

                last_state = result.echoed_message
                last_reward = reward

                if done:
                    break

        except Exception as e:
            log_step(0, "0", 0.0, True, str(e))

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

        log_end(success, steps_taken, score, rewards)

        all_rewards.extend(rewards)
        all_steps += steps_taken

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    main()