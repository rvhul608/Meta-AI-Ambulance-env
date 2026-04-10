def grade_easy(rewards):
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return max(0.0, min(1.0, score))

def grade_medium(rewards):
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return max(0.0, min(1.0, score))

def grade_hard(rewards):
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return max(0.0, min(1.0, score))