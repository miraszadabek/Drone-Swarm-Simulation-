"""
Pygame Visualizer for SAR Drone Swarm Simulation.
Dark tactical aesthetic: dark map, glowing drones, HUD overlay.
"""
import math
import pygame
import pygame.gfxdraw
from typing import List, Optional, Tuple

from environment.map import SARMap, CellType
from agents.drone_agent import LLMDroneAgent, ROLE_COLORS
from agents.blackboard import Blackboard


# ─── Color palette ────────────────────────────────────────────────────────────
BG          = (10, 12, 18)
CELL_CLEAR  = (22, 28, 38)
CELL_OBS    = (45, 42, 52)
CELL_DANGER = (80, 25, 20)
CELL_EXPLO  = (18, 35, 50)   # explored cells slightly brighter

VICTIM_UN   = (255, 230, 80)    # undiscovered (yellow pulse)
VICTIM_FOUND= (80, 255, 140)    # found (green)

HUD_BG      = (12, 15, 22, 210)  # RGBA
HUD_TEXT    = (190, 220, 255)
HUD_ACCENT  = (80, 200, 255)
HUD_ALERT   = (255, 100, 80)
HUD_OK      = (80, 255, 140)

GRID_LINE   = (30, 38, 52)
TRAIL_ALPHA = 120

FONT_MONO   = None   # set in init


class SARVisualizer:
    CELL_SIZE = 22           # pixels per grid cell
    HUD_WIDTH = 340          # right-side HUD panel

    def __init__(self, sar_map: SARMap, drones: List[LLMDroneAgent],
                 blackboard: Blackboard):
        self.map = sar_map
        self.drones = drones
        self.bb = blackboard

        self.map_px_w = sar_map.width * self.CELL_SIZE
        self.map_px_h = sar_map.height * self.CELL_SIZE
        self.win_w = self.map_px_w + self.HUD_WIDTH
        self.win_h = self.map_px_h

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("SAR Drone Swarm — AI Multi-Agent Simulation")

        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.font_md  = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_lg  = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_xl  = pygame.font.SysFont("monospace", 22, bold=True)

        # Pre-render map base (static tiles)
        self._map_surface = pygame.Surface((self.map_px_w, self.map_px_h))
        self._render_map_base()

        # Explored overlay (updated each frame)
        self._explored_overlay = pygame.Surface(
            (self.map_px_w, self.map_px_h), pygame.SRCALPHA
        )

        self._pulse_t = 0   # animation timer

    # ─── Map base ──────────────────────────────────────────────────────────

    def _render_map_base(self):
        self._map_surface.fill(BG)
        cs = self.CELL_SIZE
        for row in range(self.map.height):
            for col in range(self.map.width):
                cell = self.map.grid[row][col]
                rect = pygame.Rect(col * cs, row * cs, cs - 1, cs - 1)
                if cell == CellType.CLEAR.value:
                    color = CELL_CLEAR
                elif cell == CellType.OBSTACLE.value:
                    color = CELL_OBS
                elif cell == CellType.DANGER_ZONE.value:
                    color = CELL_DANGER
                else:
                    color = CELL_CLEAR
                pygame.draw.rect(self._map_surface, color, rect)

        # Obstacle texture lines
        for obs in self.map.obstacles:
            rx = obs.x * cs
            ry = obs.y * cs
            rw = obs.width * cs
            rh = obs.height * cs
            label_color = (70, 65, 80) if obs.type == "building" else (65, 55, 45)
            # Hatch lines
            for i in range(0, rw + rh, 8):
                x1 = rx + min(i, rw)
                y1 = ry + max(0, i - rw)
                x2 = rx + max(0, i - rh)
                y2 = ry + min(i, rh)
                pygame.draw.line(self._map_surface, label_color, (x1, y1), (x2, y2), 1)
            pygame.draw.rect(self._map_surface, (55, 52, 65),
                             pygame.Rect(rx, ry, rw, rh), 1)

        # Danger zone overlays
        for (dx, dy, dw, dh) in self.map.danger_zones:
            surf = pygame.Surface((dw * cs, dh * cs), pygame.SRCALPHA)
            surf.fill((200, 40, 20, 60))
            self._map_surface.blit(surf, (dx * cs, dy * cs))
            # Warning stripes
            for i in range(0, (dw + dh) * cs, 10):
                x1 = dx * cs + min(i, dw * cs)
                y1 = dy * cs + max(0, i - dw * cs)
                x2 = dx * cs + max(0, i - dh * cs)
                y2 = dy * cs + min(i, dh * cs)
                pygame.draw.line(self._map_surface, (220, 60, 30, 120),
                                 (x1, y1), (x2, y2), 1)

    # ─── Main render ───────────────────────────────────────────────────────

    def render(self, step: int):
        self._pulse_t += 0.08
        cs = self.CELL_SIZE

        # Blit static map
        self.screen.blit(self._map_surface, (0, 0))

        # Explored overlay
        self._render_explored()
        self.screen.blit(self._explored_overlay, (0, 0))

        # Zone boundaries
        self._render_zones()

        # Victim markers
        self._render_victims()

        # Drone trails
        for drone in self.drones:
            self._render_trail(drone)

        # Drones
        for drone in self.drones:
            self._render_drone(drone)

        # HUD
        self._render_hud(step)

        pygame.display.flip()

    def _render_explored(self):
        self._explored_overlay.fill((0, 0, 0, 0))
        cs = self.CELL_SIZE
        for (gx, gy) in self.bb.explored_cells:
            if (0 <= gx < self.map.width and 0 <= gy < self.map.height and
                    self.map.grid[gy][gx] == CellType.CLEAR.value):
                rect = pygame.Rect(gx * cs, gy * cs, cs - 1, cs - 1)
                pygame.draw.rect(self._explored_overlay, (40, 100, 160, 35), rect)

    def _render_zones(self):
        cs = self.CELL_SIZE
        zone_colors = [
            (100, 200, 255, 40),
            (255, 180, 50, 40),
            (180, 255, 100, 40),
            (255, 100, 200, 40),
            (200, 140, 255, 40),
        ]
        for drone in self.drones:
            zone = self.bb.get_zone(drone.id)
            if not zone:
                continue
            color = zone_colors[drone.id % len(zone_colors)]
            zx = int(zone.x_min * cs)
            zy = int(zone.y_min * cs)
            zw = int((zone.x_max - zone.x_min) * cs)
            zh = int((zone.y_max - zone.y_min) * cs)
            surf = pygame.Surface((zw, zh), pygame.SRCALPHA)
            surf.fill(color)
            self.screen.blit(surf, (zx, zy))
            border_col = color[:3]
            pygame.draw.rect(self.screen, border_col, pygame.Rect(zx, zy, zw, zh), 1)

    def _render_victims(self):
        cs = self.CELL_SIZE
        pulse = abs(math.sin(self._pulse_t)) * 0.5 + 0.5

        for v in self.map.victims:
            px = int(v.x * cs)
            py = int(v.y * cs)
            if v.found:
                color = VICTIM_FOUND
                radius = 5
                # Cross marker
                pygame.draw.line(self.screen, color, (px - 6, py), (px + 6, py), 2)
                pygame.draw.line(self.screen, color, (px, py - 6), (px, py + 6), 2)
                pygame.draw.circle(self.screen, color, (px, py), radius, 1)
            else:
                # Pulsing diamond
                alpha_c = int(180 + 75 * pulse)
                color = (VICTIM_UN[0], VICTIM_UN[1], VICTIM_UN[2])
                size = int(5 + 3 * pulse)
                pts = [(px, py - size), (px + size, py),
                       (px, py + size), (px - size, py)]
                pygame.draw.polygon(self.screen, color, pts)
                pygame.draw.polygon(self.screen, (255, 255, 200), pts, 1)

    def _render_trail(self, drone: LLMDroneAgent):
        trail = drone.trail
        cs = self.CELL_SIZE
        if len(trail) < 2:
            return
        base_color = drone.color
        for i in range(1, len(trail)):
            alpha = int(TRAIL_ALPHA * i / len(trail))
            px1 = int(trail[i - 1][0] * cs)
            py1 = int(trail[i - 1][1] * cs)
            px2 = int(trail[i][0] * cs)
            py2 = int(trail[i][1] * cs)
            # Fade by drawing a thin line (approximated with surface)
            faded = tuple(int(c * alpha / TRAIL_ALPHA) for c in base_color)
            pygame.draw.line(self.screen, faded, (px1, py1), (px2, py2), 1)

    def _render_drone(self, drone: LLMDroneAgent):
        cs = self.CELL_SIZE
        px = int(drone.state.x * cs)
        py = int(drone.state.y * cs)
        color = drone.color
        pulse = abs(math.sin(self._pulse_t + drone.id * 1.2))

        # Glow ring
        glow_r = int(10 + 4 * pulse)
        glow_surf = pygame.Surface((glow_r * 2 + 4, glow_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 40), (glow_r + 2, glow_r + 2), glow_r)
        self.screen.blit(glow_surf, (px - glow_r - 2, py - glow_r - 2))

        # Drone body (hexagon approximation via circle)
        pygame.draw.circle(self.screen, color, (px, py), 7)
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 7, 1)

        # Rotor arms
        arm_len = 6
        for angle in [45, 135, 225, 315]:
            rad = math.radians(angle)
            ax = int(px + arm_len * math.cos(rad))
            ay = int(py + arm_len * math.sin(rad))
            pygame.draw.line(self.screen, (180, 180, 200), (px, py), (ax, ay), 1)
            pygame.draw.circle(self.screen, (220, 220, 255), (ax, ay), 2)

        # Battery indicator (arc below drone)
        bat_color = HUD_OK if drone.battery > 30 else HUD_ALERT
        bat_w = int(14 * drone.battery / 100)
        pygame.draw.rect(self.screen, (40, 40, 50),
                         pygame.Rect(px - 7, py + 9, 14, 3))
        pygame.draw.rect(self.screen, bat_color,
                         pygame.Rect(px - 7, py + 9, bat_w, 3))

        # Danger zone indicator
        if drone.state.in_danger_zone:
            pygame.draw.circle(self.screen, HUD_ALERT, (px, py), 11, 2)

        # LLM active indicator
        if drone._pending_llm:
            t_off = int(self._pulse_t * 3) % 6
            for dot_i in range(3):
                dot_x = px - 5 + dot_i * 5
                dot_y = py - 13
                alpha = 255 if dot_i == t_off % 3 else 80
                pygame.draw.circle(self.screen,
                                   (80, 200, 255),
                                   (dot_x, dot_y), 2)

        # ID label
        label = self.font_sm.render(f"D{drone.id}", True, (220, 230, 255))
        self.screen.blit(label, (px - 7, py - 20))

    # ─── HUD ───────────────────────────────────────────────────────────────

    def _render_hud(self, step: int):
        hx = self.map_px_w
        hw = self.HUD_WIDTH
        hh = self.win_h

        # HUD background
        hud_surf = pygame.Surface((hw, hh), pygame.SRCALPHA)
        hud_surf.fill(HUD_BG)
        self.screen.blit(hud_surf, (hx, 0))
        pygame.draw.line(self.screen, HUD_ACCENT, (hx, 0), (hx, hh), 2)

        x0 = hx + 12
        y = 14

        def text(txt, font, color=HUD_TEXT, x=x0):
            surf = font.render(txt, True, color)
            self.screen.blit(surf, (x, y))
            return surf.get_height() + 3

        def divider():
            nonlocal y
            pygame.draw.line(self.screen, (40, 60, 90),
                             (hx + 8, y + 2), (hx + hw - 8, y + 2), 1)
            y += 8

        # Title
        y += text("◈ SAR SWARM COMMAND", self.font_lg, HUD_ACCENT)
        y += text("Multi-Agent LLM Simulation", self.font_sm, (120, 150, 190))
        divider()

        # Stats
        summ = self.bb.summary()
        y += text("MISSION STATUS", self.font_md, HUD_ACCENT)
        y += 4

        total_v = self.map.total_victims
        found_v = summ["found_victims"]
        cov = summ["coverage_pct"]
        elapsed = summ["elapsed_sec"]
        llm_calls = summ["total_llm_calls"]

        v_color = HUD_OK if found_v == total_v else HUD_TEXT
        y += text(f"  Victims : {found_v}/{total_v}", self.font_md, v_color)
        y += text(f"  Coverage: {cov:.1f}%", self.font_md)
        y += text(f"  Elapsed : {int(elapsed)}s", self.font_md)
        y += text(f"  LLM calls: {llm_calls}", self.font_md, HUD_ACCENT)
        y += text(f"  Sim step : {step}", self.font_sm)
        divider()

        # Drone statuses
        y += text("DRONE STATUS", self.font_md, HUD_ACCENT)
        y += 4
        for drone in self.drones:
            s = drone.state
            bat_c = HUD_OK if s.battery > 30 else HUD_ALERT
            role_c = ROLE_COLORS.get(drone.role, HUD_TEXT)
            drone_line = f"D{drone.id} [{drone.role:7s}] ♥{s.battery:.0f}%"
            y += text(drone_line, self.font_sm, role_c)
            task_short = s.current_task[:35] + "…" if len(s.current_task) > 35 else s.current_task
            y += text(f"  {task_short}", self.font_sm, (140, 160, 190))
            y += text(f"  Found: {s.victims_found} | {'⚠ DANGER' if s.in_danger_zone else 'OK'}",
                      self.font_sm, HUD_ALERT if s.in_danger_zone else (100, 160, 100))
            y += 2
        divider()

        # Recent LLM decisions
        y += text("AI DECISIONS", self.font_md, HUD_ACCENT)
        y += 4
        decisions = self.bb.get_recent_decisions(3)
        for d in reversed(decisions):
            drone_id = d["drone_id"]
            color = ROLE_COLORS.get(
                self.drones[drone_id].role if drone_id < len(self.drones) else "Scout",
                HUD_TEXT
            )
            y += text(f"D{drone_id}: {d['action']}", self.font_sm, color)
            reason = d["reasoning"][:55] + "…" if len(d["reasoning"]) > 55 else d["reasoning"]
            y += text(f"  {reason}", self.font_sm, (130, 155, 185))
            y += 2
        divider()

        # Message log
        y += text("RADIO LOG", self.font_md, HUD_ACCENT)
        y += 4
        msgs = self.bb.get_recent_messages(6)
        for msg in reversed(msgs):
            col = HUD_ALERT if msg.message_type == "alert" else (140, 170, 210)
            line = msg.content[:42] + "…" if len(msg.content) > 42 else msg.content
            y += text(line, self.font_sm, col)

        # Legend (bottom)
        legend_y = hh - 110
        pygame.draw.line(self.screen, (40, 60, 90),
                         (hx + 8, legend_y), (hx + hw - 8, legend_y), 1)
        self.screen.blit(self.font_md.render("LEGEND", True, HUD_ACCENT), (x0, legend_y + 6))
        items = [
            ("◆ Victim (unfound)", VICTIM_UN),
            ("✚ Victim (found)", VICTIM_FOUND),
            ("■ Obstacle", (70, 65, 80)),
            ("▪ Danger zone", CELL_DANGER),
            ("· Explored cell", (60, 120, 180)),
        ]
        ly = legend_y + 24
        for label, col in items:
            surf = self.font_sm.render(label, True, col)
            self.screen.blit(surf, (x0, ly))
            ly += 15

    # ─── Controls ──────────────────────────────────────────────────────────

    def handle_events(self) -> bool:
        """Returns False if user wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
        return True
