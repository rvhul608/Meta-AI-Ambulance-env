from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyAction, MyObservation
except ImportError:
    from models import MyAction, MyObservation


# Accident type profiles: (severity_modifier, people_modifier, description)
ACCIDENT_PROFILES = {
    "minor_collision":      (0.0,  0,  "Low impact, minimal injuries expected"),
    "major_collision":      (0.3,  1,  "High impact, multiple injuries likely"),
    "multi_vehicle_pileup": (0.5,  2,  "Multiple vehicles, mass casualty risk"),  # was 1.0
    "highway_accident":     (0.4,  1,  "High speed impact, critical injuries"),   # was 0.8
    "intersection_crash":   (0.2,  1,  "Moderate impact at junction"),            # was 0.3
}


class MyEnvironment(Environment):
    """
    Road Accident Emergency Dispatch Environment.

    Simulates an AI-powered emergency dispatch system for road accidents.
    The agent acts as a dispatch controller, deciding which accident zone
    to send ambulances to based on casualty count, severity, and distance.

    Inspired by real-world accident detection systems using computer vision
    to monitor CCTV feeds and automatically assess incident severity.

    Tasks:
        easy:   3 zones, lower severity, slower deterioration
        medium: 4 zones, moderate severity, faster deterioration
        hard:   5 zones, high severity, rapid deterioration

    Reward signal encourages:
        - Prioritizing high severity zones
        - Saving more casualties per dispatch
        - Efficient use of limited ambulance resources
        - Clearing all zones before max steps
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASK_CONFIGS = {
        "easy": {
            "num_zones": 3,
            "num_ambulances": 3,
            "people_range": (2, 5),
            "severity_range": (1, 3),
            "distance_range": (2, 6),
            "worsening_rate": 0.05,
            "max_steps": 8,
            "accident_types": ["minor_collision", "major_collision", "intersection_crash"],
        },
        "medium": {
            "num_zones": 4,
            "num_ambulances": 3,
            "people_range": (4, 7),
            "severity_range": (2, 4),
            "distance_range": (3, 8),
            "worsening_rate": 0.15,
            "max_steps": 9,
            "accident_types": ["major_collision", "highway_accident", "intersection_crash", "multi_vehicle_pileup"],
        },
        "hard": {
            "num_zones": 5,
            "num_ambulances": 3,
            "people_range": (6, 10),
            "severity_range": (2, 5),
            "distance_range": (3, 10),
            "worsening_rate": 0.25,
            "max_steps": 10,
            "accident_types": ["highway_accident", "multi_vehicle_pileup", "major_collision"],
        },
    }

    def __init__(self):
        """Initialize the dispatch environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.max_steps = 10
        self.task_id = "easy"
        self.zones = []
        self.ambulances = []
        self._task_config = self.TASK_CONFIGS["easy"]
        self._people_saved_total = 0

    # ---------------- RESET ---------------- #
    def reset(self, task_id: str = "easy") -> MyObservation:
        """
        Reset environment for a new episode.

        Args:
            task_id: Difficulty level - 'easy', 'medium', or 'hard'

        Returns:
            Initial observation with current zone and ambulance state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self.task_id = task_id
        self._task_config = self.TASK_CONFIGS.get(task_id, self.TASK_CONFIGS["easy"])
        self._people_saved_total = 0

        # Vary seed per episode so agent must learn general strategy
        random.seed(self._reset_count * 7 + 42)

        cfg = self._task_config
        self.max_steps = cfg["max_steps"]

        # Create accident zones
        self.zones = []
        for _ in range(cfg["num_zones"]):
            accident_type = random.choice(cfg["accident_types"])
            profile = ACCIDENT_PROFILES[accident_type]
            severity_mod, people_mod, _ = profile

            self.zones.append({
                "type": accident_type,
                "people": random.randint(*cfg["people_range"]) + people_mod,
                "severity": min(5.0, float(random.randint(*cfg["severity_range"])) + severity_mod),
                "distance": random.randint(*cfg["distance_range"]),
                "attended_before": False,
            })

        # Ambulances with slight speed variation
        self.ambulances = [
            {
                "available": True,
                "busy_time": 0,
                "speed": round(random.uniform(0.8, 1.2), 1),
            }
            for _ in range(cfg["num_ambulances"])
        ]

        return MyObservation(
            echoed_message=self._format_state(),
            message_length=0,
            done=False,
            reward=0.0,
        )

    # ---------------- STEP ---------------- #
    def step(self, action: MyAction) -> MyObservation:
        """
        Execute one dispatch decision.

        Args:
            action: Agent's chosen zone index as a string message

        Returns:
            Observation with updated state, reward, and done flag
        """
        self._state.step_count += 1
        message = action.message
        length = len(message)

        # Parse zone index from action
        try:
            zone_idx = int(message.strip())
        except (ValueError, AttributeError):
            zone_idx = -1

        raw_reward = 0.0
        cfg = self._task_config

        # -------- Validate action -------- #
        if zone_idx < 0 or zone_idx >= len(self.zones):
            raw_reward -= 1.0

        else:
            zone = self.zones[zone_idx]

            if zone["people"] == 0:
                # Penalize wasting ambulance on cleared zone
                raw_reward -= 1.5

            else:
                available = [a for a in self.ambulances if a["available"]]

                if not available:
                    # All ambulances busy
                    raw_reward -= 2.0

                else:
                    # Dispatch fastest available ambulance
                    ambulance = max(available, key=lambda a: a["speed"])

                    # Save up to 3 casualties
                    people_saved = min(zone["people"], 3)
                    zone["people"] -= people_saved
                    self._people_saved_total += people_saved

                    # Core reward: casualties saved × severity
                    raw_reward += people_saved * zone["severity"] * 2.0

                    # Urgency bonus for critical accidents
                    if zone["severity"] >= 4.0:
                        raw_reward += 2.0

                    # Distance cost — farther zones cost more response time
                    raw_reward -= zone["distance"] * 0.3

                    # First response bonus
                    if not zone["attended_before"]:
                        raw_reward += 1.0
                        zone["attended_before"] = True

                    # Ambulance travels to scene
                    ambulance["available"] = False
                    ambulance["busy_time"] = max(1, int(zone["distance"] / (2 * ambulance["speed"])))

        # -------- Update ambulances -------- #
        for amb in self.ambulances:
            if not amb["available"]:
                amb["busy_time"] -= 1
                if amb["busy_time"] <= 0:
                    amb["available"] = True

        # -------- Accident zones worsen over time -------- #
        for z in self.zones:
            if z["people"] > 0:
                z["people"] += 1
                z["severity"] = min(z["severity"] + cfg["worsening_rate"], 5.0)

                # Critical penalty — situation out of control
                if z["severity"] >= 5.0:
                    raw_reward -= 1.0

        # -------- All clear bonus -------- #
        all_clear = all(z["people"] == 0 for z in self.zones)
        if all_clear:
            steps_remaining = self.max_steps - self._state.step_count
            raw_reward += 3.0 + steps_remaining * 0.5

        # -------- Done condition -------- #
        done = self._state.step_count >= self.max_steps or all_clear

        # -------- Normalize reward to [0.0, 1.0] -------- #
        max_possible = 35.0
        min_possible = -3.0
        reward = (raw_reward - min_possible) / (max_possible - min_possible)
        reward = max(0.0, min(1.0, reward))

        return MyObservation(
            echoed_message=self._format_state(),
            message_length=length,
            done=done,
            reward=reward,
            metadata={
                "original_message": message,
                "step": self._state.step_count,
                "raw_reward": round(raw_reward, 3),
                "people_saved_total": self._people_saved_total,
                "task_id": self.task_id,
            },
        )

    # ---------------- STATE FORMAT ---------------- #
    def _format_state(self) -> str:
        """Format current state as readable string for the agent."""
        lines = [
            f"=== Road Accident Dispatch | Task: {self.task_id.upper()} | "
            f"Step: {self._state.step_count}/{self.max_steps} ==="
        ]

        lines.append("\nACCIDENT ZONES:")
        for i, z in enumerate(self.zones):
            if z["people"] == 0:
                lines.append(f"  Zone {i} [{z['type']}]: CLEARED")
            else:
                if z["severity"] >= 4.5:
                    urgency = "CRITICAL"
                elif z["severity"] >= 3.0:
                    urgency = "HIGH"
                else:
                    urgency = "MODERATE"
                lines.append(
                    f"  Zone {i} [{z['type']}]: {urgency} | "
                    f"casualties={z['people']}, "
                    f"severity={z['severity']:.1f}/5.0, "
                    f"distance={z['distance']}km"
                )

        lines.append("\nAMBULANCES:")
        for i, a in enumerate(self.ambulances):
            if a["available"]:
                lines.append(f"  Ambulance {i} [speed={a['speed']}]: AVAILABLE")
            else:
                lines.append(
                    f"  Ambulance {i} [speed={a['speed']}]: "
                    f"EN ROUTE (returns in {a['busy_time']} steps)"
                )

        lines.append(f"\nTotal casualties rescued this episode: {self._people_saved_total}")
        return "\n".join(lines)

    # ---------------- STATE ---------------- #
    @property
    def state(self) -> State:
        return self._state