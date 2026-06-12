"""
SAR Environment: Map with victims, obstacles, and terrain zones.
"""
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class CellType(Enum):
    CLEAR = 0
    OBSTACLE = 1
    DANGER_ZONE = 2   # fire, flood, etc.
    SEARCH_ZONE = 3


@dataclass
class Victim:
    x: float
    y: float
    id: int
    found: bool = False
    found_by: Optional[int] = None  # drone id
    priority: str = "medium"  # low / medium / high

    def to_dict(self):
        return {
            "id": self.id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "found": self.found,
            "found_by": self.found_by,
            "priority": self.priority,
        }


@dataclass
class Obstacle:
    x: int
    y: int
    width: int
    height: int
    type: str = "building"  # building / debris / water


class SARMap:
    """
    Search-and-rescue map. Grid-based with float-position agents.
    Grid cell size = 1 unit.
    """

    def __init__(self, width: int = 40, height: int = 30, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        self.grid = np.zeros((height, width), dtype=int)
        self.victims: List[Victim] = []
        self.obstacles: List[Obstacle] = []
        self.danger_zones: List[Tuple[int, int, int, int]] = []  # x,y,w,h
        self._generate(seed)

    def _generate(self, seed: int):
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        # Place rectangular obstacles (buildings / debris)
        obstacle_configs = [
            (5, 3, 4, 5, "building"),
            (15, 2, 6, 4, "building"),
            (28, 5, 5, 7, "building"),
            (3, 14, 3, 6, "debris"),
            (12, 10, 4, 4, "debris"),
            (22, 12, 7, 5, "building"),
            (32, 15, 4, 8, "building"),
            (8, 20, 5, 4, "debris"),
            (20, 22, 6, 4, "building"),
            (33, 24, 4, 4, "debris"),
        ]
        for x, y, w, h, t in obstacle_configs:
            obs = Obstacle(x, y, w, h, t)
            self.obstacles.append(obs)
            for row in range(y, min(y + h, self.height)):
                for col in range(x, min(x + w, self.width)):
                    self.grid[row][col] = CellType.OBSTACLE.value

        # Danger zones (fire / flood)
        danger_configs = [
            (18, 18, 4, 3),
            (6, 8, 3, 3),
        ]
        for x, y, w, h in danger_configs:
            self.danger_zones.append((x, y, w, h))
            for row in range(y, min(y + h, self.height)):
                for col in range(x, min(x + w, self.width)):
                    if self.grid[row][col] == CellType.CLEAR.value:
                        self.grid[row][col] = CellType.DANGER_ZONE.value

        # Place victims in clear cells
        priorities = ["high", "high", "medium", "medium", "medium", "low", "low", "low"]
        victim_id = 0
        attempts = 0
        placed = 0
        target = 8
        while placed < target and attempts < 500:
            x = rng.uniform(1, self.width - 1)
            y = rng.uniform(1, self.height - 1)
            gx, gy = int(x), int(y)
            if (0 <= gx < self.width and 0 <= gy < self.height and
                    self.grid[gy][gx] == CellType.CLEAR.value):
                v = Victim(x, y, victim_id, priority=priorities[placed])
                self.victims.append(v)
                victim_id += 1
                placed += 1
            attempts += 1

    def is_passable(self, x: float, y: float) -> bool:
        gx, gy = int(x), int(y)
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return False
        cell = self.grid[gy][gx]
        return cell in (CellType.CLEAR.value, CellType.SEARCH_ZONE.value,
                        CellType.DANGER_ZONE.value)

    def is_dangerous(self, x: float, y: float) -> bool:
        gx, gy = int(x), int(y)
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return False
        return self.grid[gy][gx] == CellType.DANGER_ZONE.value

    def check_victim_discovery(self, x: float, y: float,
                                drone_id: int, radius: float = 1.5) -> Optional[Victim]:
        for v in self.victims:
            if not v.found:
                dist = ((v.x - x) ** 2 + (v.y - y) ** 2) ** 0.5
                if dist <= radius:
                    v.found = True
                    v.found_by = drone_id
                    return v
        return None

    @property
    def total_victims(self) -> int:
        return len(self.victims)

    @property
    def found_victims(self) -> int:
        return sum(1 for v in self.victims if v.found)
