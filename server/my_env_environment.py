from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyAction, MyObservation
except ImportError:
    from models import MyAction, MyObservation


class MyEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize environment"""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.max_steps = 10

        self.zones = []
        self.ambulances = []

    # ---------------- RESET ---------------- #
    def reset(self, task_id: str = "easy") -> MyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        random.seed(42)

    # -------- Task configurations -------- #
        if task_id == "easy":
            num_zones = 3
            num_ambulances = 2
            people_range = (2, 5)
            severity_range = (2, 3)

        elif task_id == "medium":
            num_zones = 4
            num_ambulances = 2
            people_range = (3, 8)
            severity_range = (2, 4)

        else:  # hard
            num_zones = 5
            num_ambulances = 2
            people_range = (5, 10)
            severity_range = (3, 5)

    # rest of reset stays the same...
        # -------- Create zones -------- #
        self.zones = [
            {
                "people": random.randint(*people_range),
                "severity": random.randint(*severity_range),
                "distance": random.randint(3, 10),
            }
            for _ in range(num_zones)
        ]

        # -------- Create ambulances -------- #
        self.ambulances = [
            {"available": True, "busy_time": 0}
            for _ in range(num_ambulances)
        ]

        return MyObservation(
            echoed_message=self._format_state(),
            message_length=0,
            done=False,
            reward=0.0,
        )

    # ---------------- STEP ---------------- #
    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # -------- Parse action -------- #
        try:
            zone_idx = int(message)
        except:
            zone_idx = 0

        reward = 0.0

        # -------- Validate action -------- #
        if zone_idx < 0 or zone_idx >= len(self.zones):
            reward -= 2
        else:
            zone = self.zones[zone_idx]

            # -------- Find available ambulance -------- #
            ambulance = next((a for a in self.ambulances if a["available"]), None)

            if ambulance is None:
                reward -= 3
            else:
                # -------- Rescue logic -------- #
                people_saved = min(zone["people"], 3)
                zone["people"] -= people_saved

                # -------- Reward calculation -------- #
                reward += (people_saved * zone["severity"] * 2)
                reward -= (zone["distance"] * 0.5)

                if people_saved == 0:
                    reward -= 2

                # -------- Ambulance becomes busy -------- #
                ambulance["available"] = False
                ambulance["busy_time"] = int(zone["distance"] / 2) + 1

        # -------- Update ambulances -------- #
        for amb in self.ambulances:
            if not amb["available"]:
                amb["busy_time"] -= 1
                if amb["busy_time"] <= 0:
                    amb["available"] = True

        # -------- Zones worsen over time -------- #
        for z in self.zones:
            if z["people"] > 0:
                z["people"] += 1
                z["severity"] = min(z["severity"] + 0.2, 5)

        # -------- Done condition -------- #
        done = (
            self._state.step_count >= self.max_steps
            or all(z["people"] == 0 for z in self.zones)
        )
        max_possible = 30.0
        min_possible = -3.0
        reward = (reward - min_possible) / (max_possible - min_possible)
        reward = max(0.0, min(1.0,reward))
 
        return MyObservation(
            echoed_message=self._format_state(),
            message_length=length,
            done=done,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    # ---------------- STATE FORMAT ---------------- #
    def _format_state(self):
        zone_info = []
        for i, z in enumerate(self.zones):
            zone_info.append(
                f"Zone {i}: people={z['people']}, severity={z['severity']:.1f}, distance={z['distance']}"
            )

        amb_info = []
        for i, a in enumerate(self.ambulances):
            status = "available" if a["available"] else f"busy({a['busy_time']})"
            amb_info.append(f"Ambulance {i}: {status}")

        return "\n".join(zone_info + amb_info)

    # ---------------- STATE ---------------- #
    @property
    def state(self) -> State:
        return self._state