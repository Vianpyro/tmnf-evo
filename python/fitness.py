import math
from typing import List, Optional, Tuple

Vec3 = Tuple[float, float, float]


def dist3(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class FitnessTracker:
    """Computes fitness for a single agent run.

    fitness = n_validated * bonus - min_dist_to_nearest_unvalidated

    Checkpoints are treated as an unordered set: any unvalidated checkpoint
    can be targeted, which allows shortcuts to emerge naturally.  When a
    checkpoint is triggered, we identify it by proximity (the car must be
    adjacent) and mark it as validated.

    min_dist tracks the closest approach ever made to *any* unvalidated
    checkpoint this run, so fitness is monotonically non-decreasing.
    It resets when a new nearest checkpoint is elected after a validation.
    """

    def __init__(self, checkpoint_positions: List[Vec3], bonus: float) -> None:
        self._checkpoints = checkpoint_positions
        self._bonus = bonus
        self._validated: set[int] = set()
        self._min_dist: float = math.inf
        self._nearest_idx: int = -1   # cached after each update()

    # ------------------------------------------------------------------
    # Called every simulation step.
    # ------------------------------------------------------------------

    def update(self, pos: Vec3) -> None:
        """Find nearest unvalidated checkpoint and update min_dist."""
        best_dist = math.inf
        best_idx = -1
        for i, cp in enumerate(self._checkpoints):
            if i in self._validated:
                continue
            d = dist3(pos, cp)
            if d < best_dist:
                best_dist = d
                best_idx = i

        self._nearest_idx = best_idx
        if best_dist < self._min_dist:
            self._min_dist = best_dist

    # ------------------------------------------------------------------
    # Called when TMInterface fires on_checkpoint_count_changed.
    # pos: car position at the moment of validation.
    # ------------------------------------------------------------------

    def on_checkpoint(self, pos: Vec3) -> None:
        """Mark the nearest unvalidated checkpoint as validated."""
        if not self._unvalidated_indices():
            return

        nearest = min(
            self._unvalidated_indices(),
            key=lambda i: dist3(pos, self._checkpoints[i]),
        )
        self._validated.add(nearest)
        # Reset min_dist so we start fresh toward the new nearest target.
        self._min_dist = math.inf
        self._nearest_idx = -1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fitness(self) -> float:
        n = len(self._validated)
        if self._min_dist == math.inf:
            return float(n) * self._bonus
        return n * self._bonus - self._min_dist

    @property
    def n_validated(self) -> int:
        return len(self._validated)

    @property
    def nearest_unvalidated(self) -> Optional[Vec3]:
        """Position of the currently nearest unvalidated checkpoint, or None."""
        if self._nearest_idx >= 0:
            return self._checkpoints[self._nearest_idx]
        return None

    def _unvalidated_indices(self) -> List[int]:
        return [i for i in range(len(self._checkpoints)) if i not in self._validated]
