def grade_easy(trajectory: dict = None) -> float:
    trajectory = trajectory or {}
    rewards = trajectory.get("rewards", [])
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return max(0.0, min(1.0, score))

def grade_medium(trajectory: dict = None) -> float:
    trajectory = trajectory or {}
    rewards = trajectory.get("rewards", [])
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return max(0.0, min(1.0, score))

def grade_hard(trajectory: dict = None) -> float:
    trajectory = trajectory or {}
    rewards = trajectory.get("rewards", [])
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return max(0.0, min(1.0, score))