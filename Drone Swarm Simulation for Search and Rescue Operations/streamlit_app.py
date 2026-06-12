"""
SAR Drone Swarm Simulator — Streamlit UI
=========================================
Run:  streamlit run streamlit_app.py
"""
import time
import math
import random
import threading
import json
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAR Drone Swarm — AI Multi-Agent",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark tactical theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] { background: #0a0c12; color: #c8d8f0; }
[data-testid="stSidebar"] { background: #0d1018; border-right: 1px solid #1e3a5a; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e3a5a;
    border-radius: 8px;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: #6b8aaa !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #c8d8f0 !important; font-size: 22px !important; font-weight: 600 !important; }

/* Sidebar widgets */
[data-testid="stSidebar"] label { color: #8aa8c8 !important; }

/* Headers */
h1, h2, h3 { color: #50b4ff !important; }

/* Dividers */
hr { border-color: #1e3a5a !important; }

/* Code blocks */
code { background: #111827 !important; color: #80ccff !important; }

/* Buttons */
.stButton > button {
    background: #111827;
    border: 1px solid #2a5a8a;
    color: #80ccff;
    border-radius: 6px;
    font-weight: 500;
}
.stButton > button:hover { background: #1a2a40; border-color: #50b4ff; }

/* Log boxes */
.log-box {
    background: #080c14;
    border: 1px solid #1e3a5a;
    border-radius: 6px;
    padding: 10px 12px;
    font-family: monospace;
    font-size: 11px;
    max-height: 180px;
    overflow-y: auto;
    color: #90aacc;
}
.alert-msg { color: #ff6b6b !important; }
.info-msg  { color: #80ccaa !important; }
.llm-msg   { color: #a080ff !important; }

/* Drone status cards */
.drone-card {
    background: #0d1825;
    border: 1px solid #1e3a5a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: monospace;
    font-size: 12px;
}
.drone-scout   { border-left: 3px solid #3ab4ff; }
.drone-rescuer { border-left: 3px solid #ffa030; }
.drone-relay   { border-left: 3px solid #80dd50; }

/* Progress bars custom color */
.stProgress > div > div > div { background: #3ab4ff !important; }

/* Victim badge */
.victim-found { color: #50ff8c; font-weight: bold; }
.victim-lost  { color: #ffd060; }

/* Canvas area */
.map-container {
    background: #080c14;
    border: 1px solid #1e3a5a;
    border-radius: 8px;
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CORE  (self-contained, no pygame dependency)
# ─────────────────────────────────────────────────────────────────────────────

class CellType(Enum):
    CLEAR   = 0
    OBSTACLE= 1
    DANGER  = 2


@dataclass
class Victim:
    x: float; y: float; id: int
    found: bool = False; found_by: Optional[int] = None
    priority: str = "medium"


@dataclass
class DroneState:
    x: float; y: float
    battery: float = 100.0
    victims_found: int = 0
    current_task: str = "initializing"
    target_x: float = 0.0
    target_y: float = 0.0
    in_danger: bool = False


ROLE_NAMES  = {0:"Scout", 1:"Rescuer", 2:"Relay", 3:"Scout", 4:"Rescuer"}
ROLE_SPEEDS = {"Scout": 0.22, "Rescuer": 0.13, "Relay": 0.09}
ROLE_COLORS_HEX = {"Scout": "#3ab4ff", "Rescuer": "#ffa030", "Relay": "#80dd50"}
VICTIM_PRIORITIES = ["high","high","medium","medium","medium","low","low","low"]

OBSTACLE_CONFIGS = [
    (5,3,4,5,"building"),(15,2,6,4,"building"),(28,5,5,7,"building"),
    (3,14,3,6,"debris"),(12,10,4,4,"debris"),(22,12,7,5,"building"),
    (32,15,4,8,"building"),(8,20,5,4,"debris"),(20,22,6,4,"building"),
    (33,24,4,4,"debris"),
]
DANGER_CONFIGS = [(18,18,4,3),(6,8,3,3)]

MAP_W, MAP_H = 44, 33


def make_grid():
    g = np.zeros((MAP_H, MAP_W), dtype=int)
    for x,y,w,h,_ in OBSTACLE_CONFIGS:
        for r in range(y, min(y+h, MAP_H)):
            for c in range(x, min(x+w, MAP_W)):
                g[r][c] = CellType.OBSTACLE.value
    for x,y,w,h in DANGER_CONFIGS:
        for r in range(y, min(y+h, MAP_H)):
            for c in range(x, min(x+w, MAP_W)):
                if g[r][c] == 0: g[r][c] = CellType.DANGER.value
    return g


def make_victims(seed: int) -> List[Victim]:
    rng = random.Random(seed)
    grid = make_grid()
    victims, placed, att = [], 0, 0
    while placed < 8 and att < 500:
        vx = rng.uniform(1, MAP_W-1); vy = rng.uniform(1, MAP_H-1)
        if grid[int(vy)][int(vx)] == 0:
            victims.append(Victim(vx, vy, placed, priority=VICTIM_PRIORITIES[placed]))
            placed += 1
        att += 1
    return victims


def make_sectors(n: int) -> List[Dict]:
    cols = 2 if n <= 4 else 3; rows = math.ceil(n/cols)
    sw = MAP_W/cols; sh = MAP_H/rows
    secs = []
    for i in range(n):
        c = i % cols; r = i // cols
        secs.append(dict(xMin=c*sw, yMin=r*sh, xMax=(c+1)*sw, yMax=(r+1)*sh, done=False))
    return secs[:n]


class SwarmSimulation:
    """Full simulation state, designed to live in st.session_state."""

    def __init__(self, n_drones: int = 4, seed: int = 42, use_llm: bool = True,
                 anthropic_key: str = ""):
        self.n_drones   = n_drones
        self.seed       = seed
        self.use_llm    = use_llm
        self.api_key    = anthropic_key

        self.grid       = make_grid()
        self.victims    = make_victims(seed)
        self.sectors    = make_sectors(n_drones)

        self.step       = 0
        self.start_time = time.time()
        self.running    = False

        self.explored: set          = set()
        self.found_victims: list    = []
        self.messages: list         = []
        self.llm_decisions: list    = []
        self.total_llm_calls: int   = 0

        self.global_strategy = "Sector sweep. Prioritize uncharted areas. Report contacts immediately."

        # Drone states
        starts = [(1.5 + (MAP_W-3)*i/max(n_drones-1,1), MAP_H-2.0) for i in range(n_drones)]
        self.drones: List[DroneState] = []
        self.roles:  List[str]        = []
        self.speeds: List[float]      = []
        self.rngs:   List[random.Random] = []
        self.trails: List[list]       = []
        self.pending_llm: List[bool]  = []
        self.llm_cooldown: List[int]  = []
        self.step_count: List[int]    = []

        for i in range(n_drones):
            sx, sy = starts[i]
            role = ROLE_NAMES.get(i, "Scout")
            z = self.sectors[i]
            cx, cy = (z['xMin']+z['xMax'])/2, (z['yMin']+z['yMax'])/2
            ds = DroneState(x=sx, y=sy, target_x=cx, target_y=cy,
                            current_task=f"Moving to sector ({cx:.0f},{cy:.0f})")
            self.drones.append(ds)
            self.roles.append(role)
            self.speeds.append(ROLE_SPEEDS[role])
            self.rngs.append(random.Random(i*137+seed))
            self.trails.append([])
            self.pending_llm.append(False)
            self.llm_cooldown.append(0)
            self.step_count.append(0)

        self._post_message(-1, "🚀 Mission started. All drones deploy!", "info")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _post_message(self, sender, content, mtype="info"):
        self.messages.append({"from": sender, "content": content, "type": mtype,
                               "t": time.time()})
        if len(self.messages) > 60: self.messages.pop(0)

    def _log_decision(self, did, reasoning, action):
        self.llm_decisions.append({"id": did, "reasoning": reasoning,
                                   "action": action, "t": time.time()})
        self.total_llm_calls += 1
        if len(self.llm_decisions) > 50: self.llm_decisions.pop(0)

    def _is_passable(self, x, y) -> bool:
        gx, gy = int(x), int(y)
        if not (0 <= gx < MAP_W and 0 <= gy < MAP_H): return False
        return self.grid[gy][gx] != CellType.OBSTACLE.value

    def _pick_target(self, i: int):
        z = self.sectors[i]
        rng = self.rngs[i]
        if not z.get('done'):
            tx = rng.uniform(z['xMin']+0.5, z['xMax']-0.5)
            ty = rng.uniform(z['yMin']+0.5, z['yMax']-0.5)
        else:
            tx = rng.uniform(0.5, MAP_W-0.5)
            ty = rng.uniform(0.5, MAP_H-0.5)
        d = self.drones[i]
        d.target_x, d.target_y = tx, ty
        d.current_task = f"Patrolling → ({tx:.1f},{ty:.1f})"

    # ── Step all drones ────────────────────────────────────────────────────

    def tick(self):
        for i in range(self.n_drones):
            self._step_drone(i)
        self.step += 1

    def _step_drone(self, i: int):
        d = self.drones[i]
        rng = self.rngs[i]
        self.step_count[i] += 1
        if self.llm_cooldown[i] > 0: self.llm_cooldown[i] -= 1

        # Move toward target
        dx, dy = d.target_x - d.x, d.target_y - d.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.3:
            self._pick_target(i)
        else:
            spd = self.speeds[i]
            nx = d.x + (dx/dist)*spd; ny = d.y + (dy/dist)*spd
            if self._is_passable(nx, ny):
                d.x = max(0.5, min(MAP_W-0.5, nx))
                d.y = max(0.5, min(MAP_H-0.5, ny))
            else:
                angle = math.atan2(dy, dx) + (math.pi/2)*rng.choice([-1,1])
                ax = d.x + math.cos(angle)*spd; ay = d.y + math.sin(angle)*spd
                if self._is_passable(ax, ay):
                    d.x = max(0.5, min(MAP_W-0.5, ax))
                    d.y = max(0.5, min(MAP_H-0.5, ay))
                else:
                    self._pick_target(i)

        # Explore
        self.explored.add((int(d.x), int(d.y)))

        # Danger zone
        gx, gy = int(d.x), int(d.y)
        d.in_danger = (0<=gx<MAP_W and 0<=gy<MAP_H and
                       self.grid[gy][gx] == CellType.DANGER.value)
        if d.in_danger:
            d.target_x = MAP_W/2 + (rng.random()-0.5)*4
            d.target_y = MAP_H/2 + (rng.random()-0.5)*4

        # Victim detection
        for v in self.victims:
            if not v.found:
                dist_v = math.sqrt((v.x-d.x)**2 + (v.y-d.y)**2)
                if dist_v < 1.8:
                    v.found = True; v.found_by = i
                    d.victims_found += 1
                    self.found_victims.append(vars(v))
                    self._post_message(i,
                        f"🚨 VICTIM #{v.id} at ({v.x:.1f},{v.y:.1f}) [{v.priority.upper()}]",
                        "alert")
                    self._pick_target(i)

        # Battery drain
        d.battery = max(0, d.battery - 0.025)

        # Trail
        self.trails[i].append((d.x, d.y))
        if len(self.trails[i]) > 40: self.trails[i].pop(0)

        # LLM / heuristic decision every 40 steps
        if self.step_count[i] % 40 == 0 and self.llm_cooldown[i] == 0:
            self.llm_cooldown[i] = 60
            if self.use_llm and self.api_key:
                t = threading.Thread(target=self._llm_decision, args=(i,), daemon=True)
                t.start()
            else:
                self._heuristic_decision(i)

    def _heuristic_decision(self, i: int):
        rng = self.rngs[i]
        d = self.drones[i]
        actions = ["move_to","expand_zone","move_to","move_to","request_help"]
        reasons = [
            f"Battery {d.battery:.0f}% — continue sector sweep",
            f"{self.coverage():.0f}% covered — expanding to unexplored quadrant",
            "Coordinating with nearby drones to avoid overlap",
            "High-priority zone — intensifying search pattern",
            "Low coverage — requesting swarm reallocation",
        ]
        idx = int(rng.random() * len(reasons))
        action, reasoning = actions[idx], reasons[idx]
        self._pick_target(i)
        d.current_task = f"[AI] {action} → ({d.target_x:.1f},{d.target_y:.1f})"
        self._log_decision(i, reasoning, action)
        msgs = [
            f"D{i}: sweeping sector, no contacts",
            f"D{i}: adjusting patrol route",
            f"D{i}: moving to new waypoint",
            f"D{i}: battery nominal, continuing",
        ]
        self._post_message(i, msgs[int(rng.random()*len(msgs))], "info")

    def _llm_decision(self, i: int):
        """Real LLM call via Anthropic API."""
        try:
            import urllib.request, json as json_mod
            d = self.drones[i]
            context = {
                "drone_id": i, "role": self.roles[i],
                "position": {"x": round(d.x,1), "y": round(d.y,1)},
                "battery": round(d.battery,1),
                "victims_found_by_me": d.victims_found,
                "in_danger_zone": d.in_danger,
                "current_task": d.current_task,
                "my_zone": self.sectors[i],
                "other_drones": [
                    {"id": j, "role": self.roles[j],
                     "x": round(self.drones[j].x,1), "y": round(self.drones[j].y,1),
                     "battery": round(self.drones[j].battery,0)}
                    for j in range(self.n_drones) if j != i
                ],
                "total_found_victims": len(self.found_victims),
                "map_coverage_pct": round(self.coverage(),1),
                "global_strategy": self.global_strategy,
                "map_size": {"w": MAP_W, "h": MAP_H},
            }
            prompt = f"""You are Drone-{i} ({self.roles[i]}) in a Search and Rescue swarm.
State: {json_mod.dumps(context, indent=2)}
Map bounds: x=[0,{MAP_W}], y=[0,{MAP_H}]

Respond ONLY with JSON (no markdown):
{{
  "reasoning": "1-2 sentence tactical reasoning",
  "action": "move_to | expand_zone | request_help",
  "target_x": <float>,
  "target_y": <float>,
  "message": "radio message (max 12 words)"
}}
Rules: stay in bounds, if battery<20 move to center, avoid danger zones."""

            payload = json_mod.dumps({
                "model": "claude-sonnet-4-6",
                "max_tokens": 250,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                }, method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json_mod.loads(resp.read())
                text = data["content"][0]["text"].strip()
                text = text.replace("```json","").replace("```","").strip()
                dec = json_mod.loads(text)

            d.target_x = max(0.5, min(MAP_W-0.5, float(dec.get("target_x", d.x))))
            d.target_y = max(0.5, min(MAP_H-0.5, float(dec.get("target_y", d.y))))
            action = dec.get("action","move_to")
            d.current_task = f"[LLM] {action} → ({d.target_x:.1f},{d.target_y:.1f})"
            self._log_decision(i, dec.get("reasoning",""), action)
            msg = dec.get("message","")
            if msg: self._post_message(i, f"D{i}: {msg}", "info")

        except Exception as e:
            self._heuristic_decision(i)

    # ── Stats ───────────────────────────────────────────────────────────────

    def coverage(self) -> float:
        return min(100.0, len(self.explored)/(MAP_W*MAP_H)*100)

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def is_complete(self) -> bool:
        return len(self.found_victims) >= len(self.victims)


# ─────────────────────────────────────────────────────────────────────────────
# MAP RENDERING  (SVG for Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_svg_map(sim: SwarmSimulation, cell_px: int = 11) -> str:
    W = MAP_W * cell_px
    H = MAP_H * cell_px
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" style="display:block;border-radius:6px;background:#080c14;">']

    # Grid cells
    for ry in range(MAP_H):
        for rx in range(MAP_W):
            cell = sim.grid[ry][rx]
            explored = (rx, ry) in sim.explored
            if cell == CellType.OBSTACLE.value:
                fill = "#2a2535"
            elif cell == CellType.DANGER.value:
                fill = "#4a1510"
            elif explored:
                fill = "#0f2035"
            else:
                fill = "#0d1520"
            px, py = rx*cell_px, ry*cell_px
            parts.append(f'<rect x="{px}" y="{py}" width="{cell_px-0.5}" height="{cell_px-0.5}" fill="{fill}"/>')

    # Zone overlays
    zone_colors = ["#3ab4ff","#ffa030","#80dd50","#ff70d0","#c0a0ff"]
    for i, z in enumerate(sim.sectors):
        col = zone_colors[i % len(zone_colors)]
        zx, zy = z['xMin']*cell_px, z['yMin']*cell_px
        zw, zh = (z['xMax']-z['xMin'])*cell_px, (z['yMax']-z['yMin'])*cell_px
        parts.append(f'<rect x="{zx:.1f}" y="{zy:.1f}" width="{zw:.1f}" height="{zh:.1f}" fill="{col}" fill-opacity="0.08" stroke="{col}" stroke-width="0.5" stroke-opacity="0.3"/>')

    # Danger zone label
    for (dx, dy, dw, dh) in DANGER_CONFIGS:
        cx = (dx + dw/2)*cell_px; cy = (dy + dh/2)*cell_px
        parts.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" font-size="6" fill="#ff6633" opacity="0.8">DANGER</text>')

    # Victims
    for v in sim.victims:
        px, py = v.x*cell_px, v.y*cell_px
        if v.found:
            parts.append(f'<line x1="{px-4}" y1="{py}" x2="{px+4}" y2="{py}" stroke="#50ff8c" stroke-width="1.5"/>')
            parts.append(f'<line x1="{px}" y1="{py-4}" x2="{px}" y2="{py+4}" stroke="#50ff8c" stroke-width="1.5"/>')
            parts.append(f'<circle cx="{px}" cy="{py}" r="3.5" fill="none" stroke="#50ff8c" stroke-width="1"/>')
        else:
            s = 4
            parts.append(f'<polygon points="{px},{py-s} {px+s},{py} {px},{py+s} {px-s},{py}" fill="#ffd060" stroke="#ffaa00" stroke-width="0.5"/>')

    # Trails
    for i, trail in enumerate(sim.trails):
        col = ROLE_COLORS_HEX[sim.roles[i]]
        for j in range(1, len(trail)):
            alpha = j / len(trail) * 0.5
            x1, y1 = trail[j-1][0]*cell_px, trail[j-1][1]*cell_px
            x2, y2 = trail[j][0]*cell_px,   trail[j][1]*cell_px
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="0.8" opacity="{alpha:.2f}"/>')

    # Drones
    for i, d in enumerate(sim.drones):
        px, py = d.x*cell_px, d.y*cell_px
        col = ROLE_COLORS_HEX[sim.roles[i]]
        # Glow
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{col}" fill-opacity="0.12"/>')
        # Danger ring
        if d.in_danger:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="none" stroke="#ff4433" stroke-width="1.2" opacity="0.8"/>')
        # Body
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{col}" stroke="#ffffff" stroke-width="0.6" stroke-opacity="0.6"/>')
        # Rotors
        for angle_deg in [45, 135, 225, 315]:
            rad = math.radians(angle_deg)
            ax, ay = px + math.cos(rad)*4.5, py + math.sin(rad)*4.5
            parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="#aaaacc" stroke-width="0.5"/>')
            parts.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="1.2" fill="#ccccee" fill-opacity="0.7"/>')
        # Battery bar
        bat_col = "#50ff8c" if d.battery > 30 else "#ff5050"
        bar_w = 9 * d.battery / 100
        parts.append(f'<rect x="{px-4.5:.1f}" y="{py+6:.1f}" width="9" height="2" fill="#1a2a40" rx="1"/>')
        parts.append(f'<rect x="{px-4.5:.1f}" y="{py+6:.1f}" width="{bar_w:.1f}" height="2" fill="{bat_col}" rx="1"/>')
        # LLM indicator dots
        if sim.pending_llm[i]:
            for di in range(3):
                dot_x = px - 2 + di*2
                parts.append(f'<circle cx="{dot_x:.1f}" cy="{py-9:.1f}" r="1" fill="#3ab4ff"/>')
        # ID label
        parts.append(f'<text x="{px:.1f}" y="{py-11:.1f}" text-anchor="middle" font-size="7" fill="#c0d8f0" font-family="monospace">D{i}</text>')

    parts.append('</svg>')
    return ''.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    if "sim" not in st.session_state:
        st.session_state.sim = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "speed" not in st.session_state:
        st.session_state.speed = 3
    if "ticks_done" not in st.session_state:
        st.session_state.ticks_done = 0


def sidebar_controls():
    st.sidebar.markdown("## ⚙️ Configuration")
    st.sidebar.markdown("---")

    n_drones = st.sidebar.slider("🚁 Number of drones", 2, 5, 4)
    seed = st.sidebar.slider("🗺️ Map seed", 1, 99, 42)
    speed = st.sidebar.slider("⚡ Sim speed (ticks/frame)", 1, 8, 3)
    cell_px = st.sidebar.slider("🔍 Map cell size (px)", 8, 16, 11)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 LLM Mode")
    use_llm = st.sidebar.toggle("Enable real Claude API calls", value=False)
    api_key = ""
    if use_llm:
        api_key = st.sidebar.text_input("Anthropic API Key", type="password",
                                         placeholder="sk-ant-...")
        if not api_key:
            st.sidebar.warning("Enter API key to enable LLM mode")
            use_llm = False
        else:
            st.sidebar.success("LLM mode active")

    st.sidebar.markdown("---")

    col1, col2 = st.sidebar.columns(2)
    start_btn = col1.button("▶ Start", use_container_width=True)
    reset_btn = col2.button("↺ Reset", use_container_width=True)
    pause_btn = st.sidebar.button(
        "⏸ Pause" if st.session_state.running else "▶ Resume",
        use_container_width=True,
        disabled=(st.session_state.sim is None)
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Legend")
    st.sidebar.markdown("""
<div style="font-size:12px;font-family:monospace;">
<span style="color:#ffd060">◆</span> Victim (unfound)<br>
<span style="color:#50ff8c">✚</span> Victim (found)<br>
<span style="color:#3ab4ff">●</span> Scout drone<br>
<span style="color:#ffa030">●</span> Rescuer drone<br>
<span style="color:#80dd50">●</span> Relay drone<br>
<span style="color:#2a2535">█</span> Obstacle<br>
<span style="color:#ff4433">▪</span> Danger zone<br>
<span style="color:#0f2035">·</span> Explored area
</div>
""", unsafe_allow_html=True)

    return n_drones, seed, speed, cell_px, use_llm, api_key, start_btn, reset_btn, pause_btn


def main():
    init_session()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
  <span style="font-size:32px;">🚁</span>
  <div>
    <h1 style="margin:0;font-size:24px;color:#50b4ff;">SAR Drone Swarm Simulator</h1>
    <p style="margin:0;color:#6b8aaa;font-size:13px;">AI Multi-Agent Search & Rescue — LLM-Coordinated Autonomous Drones</p>
  </div>
</div>
<hr style="margin:8px 0 16px 0;"/>
""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    n_drones, seed, speed, cell_px, use_llm, api_key, start_btn, reset_btn, pause_btn = sidebar_controls()

    # ── Button logic ──────────────────────────────────────────────────────
    if start_btn or (st.session_state.sim is None and start_btn):
        st.session_state.sim = SwarmSimulation(n_drones, seed, use_llm, api_key)
        st.session_state.running = True
        st.session_state.ticks_done = 0

    if reset_btn:
        st.session_state.sim = SwarmSimulation(n_drones, seed, use_llm, api_key)
        st.session_state.running = True
        st.session_state.ticks_done = 0

    if pause_btn and st.session_state.sim is not None:
        st.session_state.running = not st.session_state.running

    sim: Optional[SwarmSimulation] = st.session_state.sim

    # ── No sim yet ────────────────────────────────────────────────────────
    if sim is None:
        st.info("👈 Configure and click **▶ Start** to launch the simulation.")
        st.markdown("""
### How it works
1. **Map** — procedurally generated grid with obstacles, debris, and danger zones
2. **Drones** — each has a role (Scout, Rescuer, Relay) and an assigned sector
3. **Blackboard** — shared memory for swarm coordination (found victims, messages)
4. **LLM Agent** — every 40 steps each drone calls Claude to decide its next waypoint
5. **Heuristic fallback** — works without API key (simulated decisions)
        """)
        return

    # ── Tick simulation ───────────────────────────────────────────────────
    if st.session_state.running and not sim.is_complete():
        for _ in range(speed):
            sim.tick()
        st.session_state.ticks_done += speed

    # ── Layout ────────────────────────────────────────────────────────────
    map_col, hud_col = st.columns([3, 1], gap="medium")

    with map_col:
        svg = render_svg_map(sim, cell_px=cell_px)
        st.markdown(f'<div class="map-container">{svg}</div>', unsafe_allow_html=True)

        if sim.is_complete():
            st.success("🎯 **MISSION COMPLETE** — All victims found!")
            st.session_state.running = False

    with hud_col:
        # ── Mission stats ─────────────────────────────────────────────────
        st.markdown("### 📊 Mission")
        c1, c2 = st.columns(2)
        c1.metric("Victims", f"{len(sim.found_victims)}/{len(sim.victims)}")
        c2.metric("Coverage", f"{sim.coverage():.1f}%")
        c1.metric("Elapsed", f"{sim.elapsed():.0f}s")
        c2.metric("LLM calls", sim.total_llm_calls)
        st.metric("Step", sim.step)

        st.progress(sim.coverage() / 100, text=f"Map coverage: {sim.coverage():.1f}%")
        victims_pct = len(sim.found_victims) / max(len(sim.victims), 1)
        st.progress(victims_pct, text=f"Victims: {len(sim.found_victims)}/{len(sim.victims)}")

        st.markdown("---")
        # ── Drone status ──────────────────────────────────────────────────
        st.markdown("### 🚁 Drones")
        for i, d in enumerate(sim.drones):
            role = sim.roles[i]
            col_hex = ROLE_COLORS_HEX[role]
            bat_color = "#50ff8c" if d.battery > 30 else "#ff5050"
            danger_html = '<span style="color:#ff6b6b;">⚠ DANGER</span>' if d.in_danger else '<span style="color:#50aa70;">✓ OK</span>'
            llm_dot = "🔵" if sim.pending_llm[i] else ""
            task_short = d.current_task[:38] + "…" if len(d.current_task) > 38 else d.current_task
            st.markdown(f"""
<div class="drone-card drone-{role.lower()}">
  <b style="color:{col_hex};">D{i} — {role}</b> {llm_dot}
  &nbsp;<span style="color:{bat_color};font-size:11px;">🔋{d.battery:.0f}%</span>
  &nbsp;{danger_html}<br>
  <span style="color:#6b8aaa;font-size:10px;">{task_short}</span><br>
  <span style="font-size:10px;">Found: {d.victims_found} | ({d.x:.1f},{d.y:.1f})</span>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        # ── LLM decisions ─────────────────────────────────────────────────
        st.markdown("### 🤖 AI Decisions")
        recent_dec = sim.llm_decisions[-4:][::-1]
        if recent_dec:
            dec_html = ""
            for dec in recent_dec:
                col_hex = ROLE_COLORS_HEX.get(sim.roles[dec['id']], "#80ccff")
                reason = dec['reasoning'][:55]+"…" if len(dec['reasoning'])>55 else dec['reasoning']
                dec_html += f'<div style="margin:3px 0;"><span style="color:{col_hex};">D{dec["id"]}</span> <b>{dec["action"]}</b><br><span style="color:#6b8aaa;font-size:10px;">{reason}</span></div>'
            st.markdown(f'<div class="log-box">{dec_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="log-box" style="color:#6b8aaa;">Waiting for AI decisions...</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        # ── Radio log ─────────────────────────────────────────────────────
        st.markdown("### 📡 Radio Log")
        recent_msgs = sim.messages[-8:][::-1]
        msg_html = ""
        for m in recent_msgs:
            css = "alert-msg" if m['type']=="alert" else "info-msg"
            msg_html += f'<div class="msg-item {css}">{m["content"]}</div>'
        st.markdown(f'<div class="log-box">{msg_html}</div>', unsafe_allow_html=True)

    # ── Victim table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Victim Registry")
    vcols = st.columns(min(8, len(sim.victims)))
    for idx, v in enumerate(sim.victims):
        with vcols[idx % len(vcols)]:
            if v.found:
                st.markdown(f"""
<div style="background:#0d2a1a;border:1px solid #1a6a3a;border-radius:6px;padding:6px;text-align:center;font-size:11px;font-family:monospace;">
<span class="victim-found">✔ #{v.id}</span><br>
{v.priority.upper()}<br>
D{v.found_by}→rescue
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div style="background:#1a1508;border:1px solid #5a4a10;border-radius:6px;padding:6px;text-align:center;font-size:11px;font-family:monospace;">
<span class="victim-lost">◆ #{v.id}</span><br>
{v.priority.upper()}<br>
searching…
</div>""", unsafe_allow_html=True)

    # ── Auto-rerun ────────────────────────────────────────────────────────
    if st.session_state.running and not sim.is_complete():
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
