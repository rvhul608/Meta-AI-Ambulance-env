"""
demo.py - Rapid Rescue Environment Demo

Shows the environment in action across all three difficulty levels.
Run: python3 demo.py
"""
from server.my_env_environment import MyEnvironment
from models import MyAction

def run_demo(task_id: str, steps: int = 3):
    print(f"\n{'='*60}")
    print(f"  RAPID RESCUE DEMO — Task: {task_id.upper()}")
    print(f"{'='*60}")
    
    env = MyEnvironment()
    obs = env.reset(task_id=task_id)
    print(obs.echoed_message)
    
    # Simple priority-based dispatch for demo
    for step in range(steps):
        # Parse active zones and pick highest severity
        lines = obs.echoed_message.split("\n")
        best_zone = "0"
        best_severity = -1
        for line in lines:
            if "Zone" in line and "CLEARED" not in line and "severity=" in line:
                try:
                    idx = int(line.strip().split("Zone ")[1].split(" ")[0])
                    severity = float(line.split("severity=")[1].split("/")[0])
                    if severity > best_severity:
                        best_severity = severity
                        best_zone = str(idx)
                except:
                    continue

        print(f"\n>>> Dispatching to Zone {best_zone}")
        obs = env.step(MyAction(message=best_zone))
        print(obs.echoed_message)
        print(f"Reward: {obs.reward:.3f} | Done: {obs.done}")

        if obs.done:
            print("\n✅ All zones cleared!")
            break

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_demo(task_id=task, steps=5)