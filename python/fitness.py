import math
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]
Waypoint = Dict  # {"pos": [x, y, z], "radius": float, "note": str (optional)}


def _dist_2d(a: Vec3, b) -> float:
    """Horizontal (x, z) distance — prevents rewarding underground proximity."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)


def _dist_3d(a: Vec3, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class FitnessTracker:
    """Sequential waypoint fitness for a single agent run.

    Waypoints must be validated in order. Validation uses 3D proximity
    (the car must physically enter the waypoint sphere). The fitness gradient
    uses 2D horizontal distance to avoid rewarding underground positions.

    fitness (incomplete) = n_done * bonus - min_2d_dist_to_current_target
    fitness (complete)   = n_done * bonus + (1 - elapsed/max_ms) * completion_bonus
    """

    def __init__(
        self,
        waypoints: List[Waypoint],
        bonus: float,
        max_step_ms: int = 300_000,
        completion_time_bonus: float = 0.0,
        speed_bonus_factor: float = 0.0,
    ) -> None:
        self._waypoints = waypoints
        self._bonus = bonus
        self._max_step_ms = max_step_ms
        self._completion_bonus = completion_time_bonus
        self._speed_factor = speed_bonus_factor
        self._current: int = 0
        self._min_dist: float = math.inf
        self._elapsed_ms: Optional[int] = None
        self._speed_sum: float = 0.0
        self._n_steps: int = 0

    # ------------------------------------------------------------------
    # Called every simulation step.
    # ------------------------------------------------------------------

    def update(self, pos: Vec3, speed_kmh: float = 0.0) -> None:
        self._speed_sum += speed_kmh
        self._n_steps += 1

        if self._current >= len(self._waypoints):
            return

        wp = self._waypoints[self._current]
        target = wp["pos"]
        radius = wp.get("radius", 15.0)

        # 2D gradient: guide toward target horizontally.
        d2d = _dist_2d(pos, target)
        if d2d < self._min_dist:
            self._min_dist = d2d

        # 3D validation: car must enter the waypoint sphere.
        if _dist_3d(pos, target) < radius:
            self._current += 1
            self._min_dist = math.inf  # fresh tracking for next target

    # ------------------------------------------------------------------
    # Called when the game signals all checkpoints completed.
    # ------------------------------------------------------------------

    def on_finish(self, elapsed_ms: int) -> None:
        self._current = len(self._waypoints)
        self._elapsed_ms = elapsed_ms

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_validated(self) -> int:
        return self._current

    @property
    def is_complete(self) -> bool:
        return self._current >= len(self._waypoints)

    @property
    def current_target(self) -> Optional[Vec3]:
        """Position of the next waypoint to reach, or None if done."""
        if self._current < len(self._waypoints):
            return tuple(self._waypoints[self._current]["pos"])
        return None

    @property
    def fitness(self) -> float:
        n = self._current
        avg_spd = self._speed_sum / self._n_steps if self._n_steps > 0 else 0.0
        base = float(n) * self._bonus + avg_spd * self._speed_factor

        if self._elapsed_ms is not None:
            remaining = max(0.0, 1.0 - self._elapsed_ms / self._max_step_ms)
            return base + remaining * self._completion_bonus

        if self._min_dist == math.inf:
            return base
        return base - self._min_dist
