import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tminterface.client import Client
from tminterface.interface import TMInterface

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINT_BONUS, CHECKPOINT_FILE, MAX_STEP_MS
from fitness import FitnessTracker, Vec3

import tmnf_evo


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------

def build_inputs(state, next_cp: Optional[Vec3]) -> List[float]:
    """Pack simulation state into the 9-element network input vector."""
    spd = state.display_speed / 100.0  # km/h, ~0-2 for typical TMNF speeds
    vx  = state.velocity[0] / 100.0
    vy  = state.velocity[1] / 100.0
    vz  = state.velocity[2] / 100.0
    yaw = state.yaw_pitch_roll[0]

    dx = dy = dz = 0.0
    if next_cp is not None:
        dx = (next_cp[0] - state.position[0]) / 100.0
        dy = (next_cp[1] - state.position[1]) / 100.0
        dz = (next_cp[2] - state.position[2]) / 100.0

    return [spd, vx, vy, vz, math.sin(yaw), math.cos(yaw), dx, dy, dz]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EvoClient(Client):
    """Evaluates every agent in the population sequentially on A01."""

    def __init__(self, population: tmnf_evo.Population) -> None:
        super().__init__()
        self.population = population
        self.checkpoint_positions: List[Vec3] = _load_checkpoints()

        self.fitnesses: List[float] = []
        self.current_agent: int = 0
        self._tracker: Optional[FitnessTracker] = None

        # Captured once at the first run step; reused for all agent rewinds.
        self._initial_state = None

        # Cached last known position — used in on_checkpoint_count_changed
        # so we never call get_simulation_state() during a rewind callback.
        self._last_pos: Vec3 = (0.0, 0.0, 0.0)

        # Suppresses the synthetic on_checkpoint_count_changed fired by
        # rewind_to_state before the server has responded.
        self._is_rewinding: bool = False

    # ------------------------------------------------------------------
    # TMInterface callbacks
    # ------------------------------------------------------------------

    def on_registered(self, iface: TMInterface) -> None:
        print(f"[evo] connected — generation {self.population.generation}, "
              f"{self.population.size} agents")
        # give_up() restarts the race so _initial_state is captured at t≈0.
        iface.give_up()

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time < 0:
            return

        # Capture the start state only in the first second of a fresh race.
        # If we're mid-race (connected late), warn and wait for restart.
        if self._initial_state is None:
            if _time > 1000:
                if not getattr(self, "_warned_mid_race", False):
                    print(f"[warn] Connected mid-race (t={_time}ms). "
                          "Press Delete in-game to restart the race.")
                    self._warned_mid_race = True
                return
            self._initial_state = iface.get_simulation_state()

        # Create a fresh tracker for this agent on its first step.
        if self._tracker is None:
            self._tracker = FitnessTracker(self.checkpoint_positions, CHECKPOINT_BONUS)

        if self.current_agent >= self.population.size:
            return

        state = iface.get_simulation_state()
        self._last_pos = (state.position[0], state.position[1], state.position[2])
        self._tracker.update(self._last_pos)

        if _time >= MAX_STEP_MS:
            self._finish_agent(iface)
            return

        inputs = build_inputs(state, self._tracker.nearest_unvalidated)
        actions = self.population.get_actions(self.current_agent, inputs)

        steer = int(actions[0] * 65536)
        gas   = actions[1] > 0.0
        brake = actions[2] > 0.0
        iface.set_input_state(accelerate=gas, brake=brake, steer=steer)

    def on_checkpoint_count_changed(
        self, iface: TMInterface, current: int, target: int
    ) -> None:
        # Ignore the synthetic callback fired inside rewind_to_state —
        # shared memory is mid-write at that point, unsafe to read from.
        if self._is_rewinding:
            return

        if self._tracker is not None:
            self._tracker.on_checkpoint(self._last_pos)

        if current == target:
            # Keep the simulation running so rewind_to_state works cleanly.
            iface.prevent_simulation_finish()
            self._finish_agent(iface)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finish_agent(self, iface: TMInterface) -> None:
        f = self._tracker.fitness if self._tracker else 0.0
        self.fitnesses.append(f)
        print(
            f"  agent {self.current_agent:02d}/{self.population.size - 1:02d}"
            f"  cps={self._tracker.n_validated if self._tracker else 0}"
            f"  fitness={f:8.1f}"
        )
        self.current_agent += 1
        self._tracker = None  # on_run_step will create a fresh one

        if self.current_agent >= self.population.size:
            iface.close()
            return

        self._is_rewinding = True
        iface.rewind_to_state(self._initial_state)
        self._is_rewinding = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoints() -> List[Vec3]:
    if not CHECKPOINT_FILE.exists():
        print(
            f"[warn] {CHECKPOINT_FILE} not found.\n"
            "       Run `python scripts/scout.py` first."
        )
        return []
    with open(CHECKPOINT_FILE) as fh:
        data = json.load(fh)
    return [tuple(p) for p in data["checkpoints"]]
