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
You are an emergency response system.

You are given multiple zones with:
- number of people
- severity
- distance

Choose the best zone index to send an ambulance.

Rules:
- Return ONLY a number (0,1,2,...)
- Prioritize higher severity and more people
- Prefer closer zones if similar severity
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
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{state}\nLast reward: {last_reward:.2f}"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = response.choices[0].message.content.strip()
        text = text.split()[0]

        return text if text.isdigit() else "0"

    except Exception:
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