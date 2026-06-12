"""
Blackboard: shared memory for the drone swarm.
Drones read/write here to coordinate without direct communication.
"""
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ZoneAssignment:
    drone_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    assigned_at: float = field(default_factory=time.time)
    completed: bool = False


@dataclass
class DroneStatus:
    drone_id: int
    x: float
    y: float
    battery: float
    role: str
    current_task: str
    last_update: float = field(default_factory=time.time)
    victims_found: int = 0


@dataclass
class Message:
    sender_id: int
    content: str
    timestamp: float = field(default_factory=time.time)
    message_type: str = "info"  # info / alert / request / response


class Blackboard:
    """
    Shared memory (blackboard pattern) for swarm coordination.
    Thread-safe for async drone agents.
    """

    def __init__(self, map_width: int, map_height: int):
        self._lock = threading.Lock()
        self.map_width = map_width
        self.map_height = map_height

        # Zone assignments: which drone covers which sector
        self.zone_assignments: Dict[int, ZoneAssignment] = {}

        # Drone statuses
        self.drone_statuses: Dict[int, DroneStatus] = {}

        # Known victim locations (found by any drone)
        self.found_victims: List[Dict] = []

        # Explored cells (fraction of map explored)
        self.explored_cells: set = set()

        # Message log (inter-agent communication)
        self.messages: List[Message] = []

        # LLM decision log (for visualization)
        self.llm_decisions: List[Dict] = []

        # Global strategy from command center
        self.global_strategy: str = "Sector sweep - prioritize high-density zones"

        # Simulation stats
        self.sim_start: float = time.time()
        self.total_llm_calls: int = 0

    # ─── Zone management ───────────────────────────────────────────────────

    def assign_zone(self, drone_id: int, x_min: float, y_min: float,
                    x_max: float, y_max: float):
        with self._lock:
            self.zone_assignments[drone_id] = ZoneAssignment(
                drone_id, x_min, y_min, x_max, y_max
            )

    def get_zone(self, drone_id: int) -> Optional[ZoneAssignment]:
        with self._lock:
            return self.zone_assignments.get(drone_id)

    def mark_zone_complete(self, drone_id: int):
        with self._lock:
            if drone_id in self.zone_assignments:
                self.zone_assignments[drone_id].completed = True

    def get_unassigned_area(self, num_drones: int) -> List[Dict]:
        """Divide map into sectors for initial assignment."""
        cols = 2 if num_drones <= 4 else 3
        rows = (num_drones + cols - 1) // cols
        sector_w = self.map_width / cols
        sector_h = self.map_height / rows
        sectors = []
        for r in range(rows):
            for c in range(cols):
                sectors.append({
                    "x_min": c * sector_w,
                    "y_min": r * sector_h,
                    "x_max": (c + 1) * sector_w,
                    "y_max": (r + 1) * sector_h,
                })
        return sectors[:num_drones]

    # ─── Drone status ───────────────────────────────────────────────────────

    def update_drone(self, drone_id: int, x: float, y: float,
                     battery: float, role: str, task: str, victims_found: int = 0):
        with self._lock:
            self.drone_statuses[drone_id] = DroneStatus(
                drone_id, x, y, battery, role, task,
                victims_found=victims_found
            )
            # Mark cells as explored
            self.explored_cells.add((int(x), int(y)))

    def get_all_drone_statuses(self) -> List[DroneStatus]:
        with self._lock:
            return list(self.drone_statuses.values())

    # ─── Victim reporting ───────────────────────────────────────────────────

    def report_victim(self, victim_dict: Dict):
        with self._lock:
            # Avoid duplicates
            ids = [v["id"] for v in self.found_victims]
            if victim_dict["id"] not in ids:
                self.found_victims.append(victim_dict)

    def get_found_victims(self) -> List[Dict]:
        with self._lock:
            return list(self.found_victims)

    # ─── Messaging ──────────────────────────────────────────────────────────

    def post_message(self, sender_id: int, content: str,
                     message_type: str = "info"):
        with self._lock:
            msg = Message(sender_id, content, message_type=message_type)
            self.messages.append(msg)
            # Keep last 50 messages
            if len(self.messages) > 50:
                self.messages = self.messages[-50:]

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        with self._lock:
            return list(self.messages[-n:])

    # ─── LLM decision log ───────────────────────────────────────────────────

    def log_llm_decision(self, drone_id: int, reasoning: str, action: str):
        with self._lock:
            self.llm_decisions.append({
                "drone_id": drone_id,
                "reasoning": reasoning,
                "action": action,
                "timestamp": time.time(),
            })
            self.total_llm_calls += 1
            if len(self.llm_decisions) > 100:
                self.llm_decisions = self.llm_decisions[-100:]

    def get_recent_decisions(self, n: int = 5) -> List[Dict]:
        with self._lock:
            return list(self.llm_decisions[-n:])

    # ─── Stats ──────────────────────────────────────────────────────────────

    def coverage_percent(self) -> float:
        total_cells = self.map_width * self.map_height
        return min(100.0, len(self.explored_cells) / total_cells * 100)

    def elapsed_time(self) -> float:
        return time.time() - self.sim_start

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "found_victims": len(self.found_victims),
                "coverage_pct": self.coverage_percent(),
                "elapsed_sec": round(self.elapsed_time(), 1),
                "total_llm_calls": self.total_llm_calls,
                "active_drones": len(self.drone_statuses),
                "messages_sent": len(self.messages),
            }
