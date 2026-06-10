import tmnf_evo
from fitness import FitnessTracker, Vec3
from config import (
    ALLOW_BRAKE,
    CHECKPOINT_BONUS,
    COMPLETION_TIME_BONUS,
    MAX_STEP_MS,
    SPEED_BONUS_FACTOR,
    STUCK_STEPS_LIMIT,
    WAYPOINTS_FILE,
)
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tminterface.client import Client
from tminterface.interface import TMInterface

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------

def build_inputs(state, target: Optional[Vec3]) -> List[float]:
    """Pack simulation state into the 9-element network input vector."""
    spd = state.display_speed / 100.0
    vx = state.velocity[0] / 100.0
    vy = state.velocity[1] / 100.0
    vz = state.velocity[2] / 100.0
    yaw = state.yaw_pitch_roll[0]

    dx = dy = dz = 0.0
    if target is not None:
        dx = (target[0] - state.position[0]) / 100.0
        dy = (target[1] - state.position[1]) / 100.0
        dz = (target[2] - state.position[2]) / 100.0

    return [spd, vx, vy, vz, math.sin(yaw), math.cos(yaw), dx, dy, dz]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EvoClient(Client):
    """Evaluates every agent in the population sequentially."""

    def __init__(self, population: tmnf_evo.Population) -> None:
        super().__init__()
        self.population = population
        self.waypoints: List = _load_waypoints()

        self.fitnesses: List[float] = []
        self.current_agent: int = 0
        self._tracker: Optional[FitnessTracker] = None

        # Captured once at the first valid run step; reused for all agent rewinds.
        self._initial_state = None
        self._last_pos: Vec3 = (0.0, 0.0, 0.0)
        self._is_rewinding: bool = False
        self._stuck_steps: int = 0
        self._agent_start_time: int = 0
        self._current_time: int = 0

    # ------------------------------------------------------------------
    # TMInterface callbacks
    # ------------------------------------------------------------------

    def on_registered(self, iface: TMInterface) -> None:
        print(f"[evo] connected — generation {self.population.generation}, "
              f"{self.population.size} agents")
        iface.give_up()

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time < 0:
            return

        self._current_time = _time

        # Capture start state only at t=0-10ms: car is stationary at spawn.
        # Capturing later introduces variance — the same weights give different
        # trajectories depending on where the car happens to be at capture time.
        if self._initial_state is None:
            if _time <= 10:
                self._initial_state = iface.get_simulation_state()
            elif _time > 3000:
                # give_up() failed to restart in time.
                if not getattr(self, "_warned_mid_race", False):
                    print(f"[warn] Could not auto-restart (t={_time}ms). "
                          "Press Delete in-game.")
                    self._warned_mid_race = True
            # Always return until state is locked — don't start evaluating yet.
            return

        if self._tracker is None:
            self._tracker = FitnessTracker(
                self.waypoints,
                CHECKPOINT_BONUS,
                max_step_ms=MAX_STEP_MS,
                completion_time_bonus=COMPLETION_TIME_BONUS,
                speed_bonus_factor=SPEED_BONUS_FACTOR,
            )
            self._stuck_steps = 0
            self._agent_start_time = _time

        if self.current_agent >= self.population.size:
            return

        state = iface.get_simulation_state()
        self._last_pos = (state.position[0],
                          state.position[1], state.position[2])
        self._tracker.update(
            self._last_pos, speed_kmh=float(state.display_speed))

        # All waypoints validated by proximity — finish immediately.
        if self._tracker.is_complete:
            elapsed = _time - self._agent_start_time
            self._tracker.on_finish(elapsed)
            self._finish_agent(iface)
            return

        # Kill stuck agents after grace period.
        if _time - self._agent_start_time > 500:
            if state.display_speed < 5:
                self._stuck_steps += 1
                if self._stuck_steps >= STUCK_STEPS_LIMIT:
                    self._finish_agent(iface)
                    return
            else:
                self._stuck_steps = 0

        if _time >= self._agent_start_time + MAX_STEP_MS:
            self._finish_agent(iface)
            return

        inputs = build_inputs(state, self._tracker.current_target)
        actions = self.population.get_actions(self.current_agent, inputs)

        steer = int(actions[0] * 65536)
        gas = actions[1] > 0.0
        brake = (actions[2] > 0.0) and ALLOW_BRAKE
        iface.set_input_state(accelerate=gas, brake=brake, steer=steer)

    def on_checkpoint_count_changed(
        self, iface: TMInterface, current: int, target: int
    ) -> None:
        if self._is_rewinding:
            return
        if current == target:
            # The game says the race is done. Record finish time and keep sim alive.
            if self._tracker is not None and not self._tracker.is_complete:
                elapsed = self._current_time - self._agent_start_time
                self._tracker.on_finish(elapsed)
            iface.prevent_simulation_finish()
            self._finish_agent(iface)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finish_agent(self, iface: TMInterface) -> None:
        f = self._tracker.fitness if self._tracker else 0.0
        self.fitnesses.append(f)
        cps = self._tracker.n_validated if self._tracker else 0
        print(
            f"  agent {self.current_agent:02d}/{self.population.size - 1:02d}"
            f"  wps={cps}"
            f"  fitness={f:8.1f}"
        )
        self.current_agent += 1
        self._tracker = None

        if self.current_agent >= self.population.size:
            iface.close()
            return

        self._is_rewinding = True
        iface.rewind_to_state(self._initial_state)
        self._is_rewinding = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_waypoints() -> List:
    """Load waypoints file. Supports both new format and legacy checkpoint format."""
    if not WAYPOINTS_FILE.exists():
        print(
            f"[warn] {WAYPOINTS_FILE} not found.\n"
            "       Run `python scripts/scout.py` first."
        )
        return []
    with open(WAYPOINTS_FILE) as fh:
        data = json.load(fh)

    # New format: {"waypoints": [{"pos": [...], "radius": float}, ...]}
    if "waypoints" in data:
        return data["waypoints"]

    # Legacy format: {"checkpoints": [[x, y, z], ...]}
    if "checkpoints" in data:
        print("[info] Legacy checkpoint format detected — converting. "
              "Re-run scout.py for proper waypoints.")
        return [{"pos": cp, "radius": 12.0} for cp in data["checkpoints"]]

    return []
