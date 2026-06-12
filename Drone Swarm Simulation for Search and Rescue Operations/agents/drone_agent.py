"""
LLM Drone Agent: each drone is an autonomous agent powered by Claude.
Decisions are made based on local sensor data + blackboard state.
"""
import math
import random
import time
import json
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.map import SARMap
from agents.blackboard import Blackboard


DRONE_ROLES = {
    0: "Scout",       # fast, explores new areas
    1: "Rescuer",     # slow, thorough search
    2: "Relay",       # stays elevated, coordinates
    3: "Scout",
    4: "Rescuer",
}

ROLE_COLORS = {
    "Scout":   (100, 200, 255),   # cyan-blue
    "Rescuer": (255, 180, 50),    # amber
    "Relay":   (180, 255, 100),   # lime
}

ROLE_SPEEDS = {
    "Scout":   0.18,
    "Rescuer": 0.10,
    "Relay":   0.07,
}


@dataclass
class DroneState:
    x: float
    y: float
    battery: float = 100.0
    victims_found: int = 0
    current_task: str = "initializing"
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    in_danger_zone: bool = False


class LLMDroneAgent:
    """
    Autonomous drone agent. Every N steps it calls the LLM to decide
    its next waypoint / strategy based on:
      - Its own state (position, battery, role)
      - Blackboard state (other drones, found victims, coverage)
      - Local sensor scan (nearby clear/blocked cells)
    """

    LLM_CALL_INTERVAL = 8   # steps between LLM calls (balance cost vs reactivity)

    def __init__(self, drone_id: int, start_x: float, start_y: float,
                 sar_map: SARMap, blackboard: Blackboard,
                 api_key: str = ""):
        self.id = drone_id
        self.role = DRONE_ROLES.get(drone_id, "Scout")
        self.color = ROLE_COLORS[self.role]
        self.speed = ROLE_SPEEDS[self.role]
        self.state = DroneState(x=start_x, y=start_y)
        self.map = sar_map
        self.bb = blackboard
        self.api_key = api_key

        self._step_count = 0
        self._rng = random.Random(drone_id * 137 + 42)
        self._lock = threading.Lock()
        self._llm_thread: Optional[threading.Thread] = None
        self._pending_llm = False
        self._trail: list = []          # last N positions for visualization

        # Assign initial sector from blackboard
        zone = blackboard.get_zone(drone_id)
        if zone:
            cx = (zone.x_min + zone.x_max) / 2
            cy = (zone.y_min + zone.y_max) / 2
            self.state.target_x = cx
            self.state.target_y = cy
            self.state.current_task = f"Moving to sector ({cx:.0f},{cy:.0f})"

    # ─── Main step ──────────────────────────────────────────────────────────

    def step(self):
        """Called every simulation tick."""
        with self._lock:
            self._step_count += 1

            # Move toward target
            self._move()

            # Check for victims
            victim = self.map.check_victim_discovery(
                self.state.x, self.state.y, self.id, radius=1.8
            )
            if victim:
                self.state.victims_found += 1
                self.bb.report_victim(victim.to_dict())
                self.bb.post_message(
                    self.id,
                    f"🚨 VICTIM FOUND at ({victim.x:.1f},{victim.y:.1f}) "
                    f"priority={victim.priority}",
                    message_type="alert"
                )
                # Immediately pick new target after find
                self._pick_patrol_target()

            # Drain battery
            self.state.battery = max(0, self.state.battery - 0.03)
            self.state.in_danger_zone = self.map.is_dangerous(
                self.state.x, self.state.y
            )

            # Update blackboard
            self.bb.update_drone(
                self.id, self.state.x, self.state.y,
                self.state.battery, self.role,
                self.state.current_task, self.state.victims_found
            )

            # Keep trail
            self._trail.append((self.state.x, self.state.y))
            if len(self._trail) > 40:
                self._trail.pop(0)

            # Trigger LLM decision async
            if self._step_count % self.LLM_CALL_INTERVAL == 0 and not self._pending_llm:
                self._trigger_llm_decision()

    # ─── Movement ───────────────────────────────────────────────────────────

    def _move(self):
        if self.state.target_x is None or self.state.target_y is None:
            self._pick_patrol_target()
            return

        dx = self.state.target_x - self.state.x
        dy = self.state.target_y - self.state.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.3:
            # Reached target → pick next
            self._pick_patrol_target()
            return

        step = self.speed
        nx = self.state.x + (dx / dist) * step
        ny = self.state.y + (dy / dist) * step

        if self.map.is_passable(nx, ny):
            self.state.x = max(0.5, min(self.map.width - 0.5, nx))
            self.state.y = max(0.5, min(self.map.height - 0.5, ny))
        else:
            # Obstacle avoidance: try perpendicular
            angle = math.atan2(dy, dx) + (math.pi / 2) * self._rng.choice([-1, 1])
            ax = self.state.x + math.cos(angle) * step
            ay = self.state.y + math.sin(angle) * step
            if self.map.is_passable(ax, ay):
                self.state.x = max(0.5, min(self.map.width - 0.5, ax))
                self.state.y = max(0.5, min(self.map.height - 0.5, ay))
            else:
                self._pick_patrol_target()

    def _pick_patrol_target(self):
        """Pick next patrol waypoint within assigned zone."""
        zone = self.bb.get_zone(self.id)
        if zone and not zone.completed:
            x = self._rng.uniform(zone.x_min + 0.5, zone.x_max - 0.5)
            y = self._rng.uniform(zone.y_min + 0.5, zone.y_max - 0.5)
        else:
            x = self._rng.uniform(0.5, self.map.width - 0.5)
            y = self._rng.uniform(0.5, self.map.height - 0.5)

        self.state.target_x = x
        self.state.target_y = y
        self.state.current_task = f"Patrolling → ({x:.1f},{y:.1f})"

    # ─── LLM Decision ───────────────────────────────────────────────────────

    def _trigger_llm_decision(self):
        self._pending_llm = True
        context = self._build_context()
        t = threading.Thread(
            target=self._call_llm, args=(context,), daemon=True
        )
        self._llm_thread = t
        t.start()

    def _build_context(self) -> Dict[str, Any]:
        """Snapshot of state for LLM prompt."""
        statuses = self.bb.get_all_drone_statuses()
        other_drones = [
            {"id": s.drone_id, "role": s.role,
             "x": round(s.x, 1), "y": round(s.y, 1),
             "battery": round(s.battery, 0),
             "task": s.current_task}
            for s in statuses if s.drone_id != self.id
        ]
        recent_msgs = [
            {"from": m.sender_id, "msg": m.content, "type": m.message_type}
            for m in self.bb.get_recent_messages(5)
        ]
        zone = self.bb.get_zone(self.id)
        zone_info = None
        if zone:
            zone_info = {
                "x_min": zone.x_min, "y_min": zone.y_min,
                "x_max": zone.x_max, "y_max": zone.y_max,
                "completed": zone.completed,
            }
        return {
            "drone_id": self.id,
            "role": self.role,
            "position": {"x": round(self.state.x, 1), "y": round(self.state.y, 1)},
            "battery": round(self.state.battery, 1),
            "victims_found_by_me": self.state.victims_found,
            "in_danger_zone": self.state.in_danger_zone,
            "current_task": self.state.current_task,
            "my_zone": zone_info,
            "other_drones": other_drones,
            "total_found_victims": len(self.bb.get_found_victims()),
            "map_coverage_pct": round(self.bb.coverage_percent(), 1),
            "recent_messages": recent_msgs,
            "global_strategy": self.bb.global_strategy,
            "map_size": {"w": self.map.map_width, "h": self.map.map_height} if hasattr(self.map, 'map_width') else {"w": self.map.width, "h": self.map.height},
        }

    def _call_llm(self, context: Dict):
        """Synchronous LLM call (runs in thread)."""
        try:
            import urllib.request
            import json as json_mod

            prompt = f"""You are Drone-{self.id} ({self.role}) in a Search and Rescue swarm simulation.
Your job: decide your next waypoint and action based on the current situation.

Current state:
{json_mod.dumps(context, indent=2)}

Map bounds: x=[0,{context['map_size']['w']}], y=[0,{context['map_size']['h']}]

Based on this, respond with a JSON object only (no markdown, no explanation outside JSON):
{{
  "reasoning": "1-2 sentence tactical reasoning",
  "action": "move_to | stay | request_help | expand_zone",
  "target_x": <float within map bounds>,
  "target_y": <float within map bounds>,
  "message": "brief radio message to swarm (max 15 words)"
}}

Rules:
- Stay within map bounds
- If battery < 20%, move toward center (recharge zone)
- If in danger zone, move away immediately
- Coordinate with other drones to avoid overlap
- Prioritize unexplored areas
- If victim found nearby, circle the area"""

            payload = json_mod.dumps({
                "model": "claude-sonnet-4-6",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json_mod.loads(resp.read())
                text = data["content"][0]["text"].strip()
                # Strip markdown fences if present
                text = text.replace("```json", "").replace("```", "").strip()
                decision = json_mod.loads(text)

            with self._lock:
                # Apply decision
                tx = float(decision.get("target_x", self.state.x))
                ty = float(decision.get("target_y", self.state.y))
                # Clamp to map
                tx = max(0.5, min(self.map.width - 0.5, tx))
                ty = max(0.5, min(self.map.height - 0.5, ty))
                self.state.target_x = tx
                self.state.target_y = ty
                action = decision.get("action", "move_to")
                self.state.current_task = f"[LLM] {action} → ({tx:.1f},{ty:.1f})"

                # Post message
                msg = decision.get("message", "")
                if msg:
                    self.bb.post_message(self.id, f"[D{self.id}] {msg}", "info")

                # Log decision
                self.bb.log_llm_decision(
                    self.id,
                    decision.get("reasoning", ""),
                    action
                )

        except Exception as e:
            # Fallback: pick patrol target (no LLM)
            with self._lock:
                self._pick_patrol_target()
                self.bb.log_llm_decision(
                    self.id,
                    f"LLM unavailable ({type(e).__name__}), using heuristic",
                    "patrol_fallback"
                )
        finally:
            self._pending_llm = False

    # ─── Properties for rendering ────────────────────────────────────────────

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.state.x, self.state.y)

    @property
    def battery(self) -> float:
        return self.state.battery

    @property
    def trail(self) -> list:
        return list(self._trail)
