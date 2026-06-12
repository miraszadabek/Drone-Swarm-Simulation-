import os

MAP_WIDTH = 44
MAP_HEIGHT = 33
SEED = 42

BATTERY_DRAIN_RATE = 0.025
LLM_CALL_INTERVAL_STEPS = 40
VICTIM_DETECTION_RADIUS = 1.8

DRONE_ROLES = {
    0: "Scout",
    1: "Rescuer",
    2: "Relay",
    3: "Scout",
    4: "Rescuer"
}

ROLE_SPEEDS = {
    "Scout": 0.22,
    "Rescuer": 0.13,
    "Relay": 0.09
}

ROLE_COLORS_HEX = {
    "Scout": "#3ab4ff",
    "Rescuer": "#ffa030",
    "Relay": "#80dd50"
}

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "mission_logs.db")