def grade_easy(trajectory: dict = None) -> float:
    trajectory = trajectory or {}
    rewards = trajectory.get("rewards", [])
    if not rewards:
        return 0.0
    # Easy: reward consistency matters, penalize variance
    avg = sum(rewards) / len(rewards)
    return max(0.0, min(1.0, avg))

def grade_medium(trajectory: dict = None) -> float:
    trajectory = trajectory or {}
    rewards = trajectory.get("rewards", [])
    if not rewards:
        return 0.0
    # Medium: reward peak performance + consistency
    avg = sum(rewards) / len(rewards)
    peak = max(rewards)
    return max(0.0, min(1.0, (avg * 0.7) + (peak * 0.3)))

def grade_hard(trajectory: dict = None) -> float:
    trajectory = trajectory or {}
    rewards = trajectory.get("rewards", [])
    if not rewards:
        return 0.0
    # Hard: penalize any zero rewards heavily
    avg = sum(rewards) / len(rewards)
    zero_penalty = rewards.count(0.0) * 0.05
    return max(0.0, min(1.0, avg - zero_penalty))