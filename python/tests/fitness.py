import math
import pytest
from fitness import FitnessTracker

# Three checkpoints on the X axis, 100 units apart.
CPS = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
BONUS = 1000.0


def tracker() -> FitnessTracker:
    return FitnessTracker(CPS, BONUS)


# ── update() ──────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_nearest_checkpoint_selected(self):
        t = tracker()
        # (30, 0, 0) is 30 units from cp0 and 70 from cp1.
        t.update((30.0, 0.0, 0.0))
        assert t._min_dist == pytest.approx(30.0)

    def test_min_dist_is_monotonically_non_increasing(self):
        t = tracker()
        t.update((30.0, 0.0, 0.0))
        t.update((60.0, 0.0, 0.0))  # now farther from cp0, 40 from cp1
        # min_dist should stay at 30, not grow to 40
        assert t._min_dist == pytest.approx(30.0)

    def test_nearest_unvalidated_points_to_closest(self):
        t = tracker()
        # Closer to cp1 (100, 0, 0) than cp0 (0, 0, 0)
        t.update((90.0, 0.0, 0.0))
        assert t.nearest_unvalidated == (100.0, 0.0, 0.0)

    def test_nearest_unvalidated_none_when_no_checkpoints(self):
        t = FitnessTracker([], BONUS)
        t.update((0.0, 0.0, 0.0))
        assert t.nearest_unvalidated is None


# ── on_checkpoint() ───────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_nearest_checkpoint_validated(self):
        t = tracker()
        t.on_checkpoint((5.0, 0.0, 0.0))  # near cp0
        assert 0 in t._validated

    def test_non_sequential_validation(self):
        # Car skips to cp2 — checkpoint 2 should be validated, not 0.
        t = tracker()
        t.on_checkpoint((198.0, 0.0, 0.0))
        assert 2 in t._validated
        assert 0 not in t._validated

    def test_min_dist_resets_after_validation(self):
        t = tracker()
        t.update((5.0, 0.0, 0.0))
        assert t._min_dist < math.inf
        t.on_checkpoint((1.0, 0.0, 0.0))
        assert t._min_dist == math.inf

    def test_already_validated_not_double_counted(self):
        t = tracker()
        t.on_checkpoint((1.0, 0.0, 0.0))   # validates cp0
        t.on_checkpoint((2.0, 0.0, 0.0))   # car still near cp0 — should validate cp1 now
        assert 1 in t._validated
        assert t.n_validated == 2

    def test_no_crash_when_all_validated(self):
        t = tracker()
        t.on_checkpoint((1.0, 0.0, 0.0))
        t.on_checkpoint((101.0, 0.0, 0.0))
        t.on_checkpoint((201.0, 0.0, 0.0))
        # Spurious extra call must not raise.
        t.on_checkpoint((0.0, 0.0, 0.0))
        assert t.n_validated == 3


# ── fitness property ───────────────────────────────────────────────────────────

class TestFitness:
    def test_zero_before_first_update(self):
        # min_dist is inf → fallback returns n_validated * bonus = 0
        assert tracker().fitness == 0.0

    def test_closer_position_yields_higher_fitness(self):
        t_far = tracker()
        t_far.update((50.0, 0.0, 0.0))

        t_near = tracker()
        t_near.update((10.0, 0.0, 0.0))

        assert t_near.fitness > t_far.fitness

    def test_checkpoint_bonus_increases_fitness(self):
        t = tracker()
        t.update((5.0, 0.0, 0.0))
        before = t.fitness
        t.on_checkpoint((1.0, 0.0, 0.0))
        t.update((5.0, 0.0, 0.0))  # still near start, now targeting cp1
        assert t.fitness > before

    def test_fitness_formula(self):
        t = tracker()
        t.update((40.0, 0.0, 0.0))   # 40 units from cp0
        assert t.fitness == pytest.approx(0 * BONUS - 40.0)

    def test_fitness_non_decreasing_on_approach(self):
        t = tracker()
        prev = -math.inf
        for x in [80.0, 60.0, 40.0, 20.0, 5.0]:
            t.update((x, 0.0, 0.0))
            assert t.fitness >= prev
            prev = t.fitness
