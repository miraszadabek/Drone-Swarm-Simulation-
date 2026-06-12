"""
SAR Drone Swarm Simulation — Main Entry Point
=============================================
Run:  python main.py
      python main.py --no-llm      (heuristic mode, no API calls)
      python main.py --drones 5    (set swarm size)
      python main.py --seed 7      (different map)
"""
import sys
import time
import argparse
import os

import pygame


def parse_args():
    p = argparse.ArgumentParser(description="SAR Drone Swarm Simulator")
    p.add_argument("--drones", type=int, default=4,
                   help="Number of drones in swarm (2-5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Map generation seed")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable LLM calls (pure heuristic mode)")
    p.add_argument("--fps", type=int, default=30,
                   help="Target FPS")
    return p.parse_args()


def build_start_positions(n: int, map_w: int, map_h: int):
    """Spread drones along bottom edge of map."""
    positions = []
    for i in range(n):
        x = 1.5 + (map_w - 3) * i / max(n - 1, 1)
        y = map_h - 2.0
        positions.append((x, y))
    return positions


def main():
    args = parse_args()
    n_drones = max(2, min(5, args.drones))

    # ── Imports ────────────────────────────────────────────────────────────
    from environment.map import SARMap
    from agents.blackboard import Blackboard
    from agents.drone_agent import LLMDroneAgent
    from visualization.renderer import SARVisualizer

    # ── Setup ──────────────────────────────────────────────────────────────
    print(f"[SAR] Initializing map (seed={args.seed})...")
    sar_map = SARMap(width=40, height=30, seed=args.seed)
    print(f"[SAR] Map: {sar_map.width}×{sar_map.height} | "
          f"Victims: {sar_map.total_victims} | "
          f"Obstacles: {len(sar_map.obstacles)}")

    blackboard = Blackboard(sar_map.width, sar_map.height)

    # Assign sectors
    sectors = blackboard.get_unassigned_area(n_drones)
    for i, sec in enumerate(sectors):
        blackboard.assign_zone(i, sec["x_min"], sec["y_min"],
                               sec["x_max"], sec["y_max"])

    # Create drones
    starts = build_start_positions(n_drones, sar_map.width, sar_map.height)
    drones = []
    for i in range(n_drones):
        sx, sy = starts[i]
        drone = LLMDroneAgent(
            drone_id=i,
            start_x=sx,
            start_y=sy,
            sar_map=sar_map,
            blackboard=blackboard,
        )
        if args.no_llm:
            # Patch: disable LLM calls
            drone.LLM_CALL_INTERVAL = 99999
        drones.append(drone)
        print(f"[SAR] Drone {i} ({drone.role}) starting at ({sx:.1f},{sy:.1f})")

    blackboard.post_message(-1, "Mission started. All drones deploy!", "info")
    blackboard.global_strategy = (
        "Sector sweep pattern. Prioritize high-density areas. "
        "Report all victims immediately."
    )

    # ── Renderer ───────────────────────────────────────────────────────────
    renderer = SARVisualizer(sar_map, drones, blackboard)
    clock = pygame.time.Clock()

    print(f"[SAR] Simulation running. Press Q or ESC to quit.")
    if not args.no_llm:
        print("[SAR] LLM mode: drones will call Claude API every "
              f"{drones[0].LLM_CALL_INTERVAL} steps.")
    else:
        print("[SAR] Heuristic mode (--no-llm): no API calls.")

    step = 0
    running = True

    while running:
        running = renderer.handle_events()

        # Step all drones
        for drone in drones:
            drone.step()

        # Render
        renderer.render(step)
        clock.tick(args.fps)
        step += 1

        # Auto-end if all victims found
        if sar_map.found_victims == sar_map.total_victims and step > 60:
            blackboard.post_message(-1,
                "🎯 ALL VICTIMS FOUND — Mission complete!", "alert")
            renderer.render(step)
            pygame.time.wait(3000)
            running = False

    # ── Summary ────────────────────────────────────────────────────────────
    summ = blackboard.summary()
    print("\n" + "="*50)
    print("MISSION SUMMARY")
    print("="*50)
    print(f"  Victims found  : {summ['found_victims']}/{sar_map.total_victims}")
    print(f"  Map coverage   : {summ['coverage_pct']:.1f}%")
    print(f"  Elapsed time   : {summ['elapsed_sec']:.1f}s")
    print(f"  LLM API calls  : {summ['total_llm_calls']}")
    print(f"  Messages sent  : {summ['messages_sent']}")
    print("="*50)

    pygame.quit()


if __name__ == "__main__":
    main()
