import math
import pytest
from fitness import FitnessTracker

# Three waypoints in a line, 100 units apart, radius 15.
WPS = [
    {"pos": [0.0, 0.0, 0.0], "radius": 15.0},
    {"pos": [100.0, 0.0, 0.0], "radius": 15.0},
    {"pos": [200.0, 0.0, 0.0], "radius": 15.0},
]
BONUS = 1000.0


def t() -> FitnessTracker:
    return FitnessTracker(WPS, BONUS)


# ── update / proximity validation ─────────────────────────────────────────────

class TestUpdate:
    def test_tracks_2d_min_dist(self):
        tr = t()
        tr.update((50.0, 0.0, 0.0))
        assert tr._min_dist == pytest.approx(50.0)

    def test_min_dist_non_increasing(self):
        tr = t()
        tr.update((30.0, 0.0, 0.0))
        tr.update((50.0, 0.0, 0.0))
        assert tr._min_dist == pytest.approx(30.0)

    def test_validates_within_radius(self):
        tr = t()
        tr.update((5.0, 0.0, 0.0))   # 3D dist = 5 < radius 15
        assert tr.n_validated == 1

    def test_no_validate_outside_radius(self):
        tr = t()
        tr.update((20.0, 0.0, 0.0))  # 3D dist = 20 > radius 15
        assert tr.n_validated == 0

    def test_sequential_order_enforced(self):
        tr = t()
        # Pass near wp1 without having passed wp0 first — should not validate.
        tr.update((100.0, 0.0, 0.0))
        assert tr.n_validated == 0  # wp0 not yet validated

    def test_advances_target_after_validation(self):
        tr = t()
        tr.update((5.0, 0.0, 0.0))
        assert tr.current_target == (100.0, 0.0, 0.0)

    def test_min_dist_resets_after_validation(self):
        tr = t()
        tr.update((5.0, 0.0, 0.0))
        assert tr._min_dist == math.inf

    def test_underground_does_not_validate(self):
        # Directly below wp0 (y=-100): 3D dist ≈ 100 >> radius 15.
        tr = t()
        tr.update((0.0, -100.0, 0.0))
        assert tr.n_validated == 0

    def test_no_crash_after_all_validated(self):
        tr = t()
        for pos in [(5, 0, 0), (100, 0, 0), (200, 0, 0)]:
            tr.update(pos)
        tr.update((999, 0, 0))  # must not raise


# ── fitness formula ────────────────────────────────────────────────────────────

class TestFitness:
    def test_zero_before_first_update(self):
        assert t().fitness == 0.0

    def test_closer_position_better_fitness(self):
        t_far = t(); t_far.update((50.0, 0.0, 0.0))
        t_near = t(); t_near.update((10.0, 0.0, 0.0))
        assert t_near.fitness > t_far.fitness

    def test_fitness_formula(self):
        tr = t()
        tr.update((40.0, 0.0, 0.0))
        assert tr.fitness == pytest.approx(0 * BONUS - 40.0)

    def test_bonus_after_validation(self):
        tr = t()
        tr.update((5.0, 0.0, 0.0))    # validate wp0 (dist3d=5 < 15)
        tr.update((120.0, 0.0, 0.0))  # 20 from wp1 — outside radius, not validated
        assert tr.fitness == pytest.approx(1 * BONUS - 20.0)

    def test_completion_time_bonus(self):
        tr = FitnessTracker(WPS, BONUS, max_step_ms=1000, completion_time_bonus=500.0)
        tr.on_finish(elapsed_ms=500)  # half of max time → 50% bonus
        assert tr.fitness == pytest.approx(3 * BONUS + 250.0)

    def test_completion_at_limit_no_time_bonus(self):
        tr = FitnessTracker(WPS, BONUS, max_step_ms=1000, completion_time_bonus=500.0)
        tr.on_finish(elapsed_ms=1000)
        assert tr.fitness == pytest.approx(3 * BONUS + 0.0)

    def test_fitness_non_decreasing_on_approach(self):
        tr = t()
        prev = -math.inf
        for x in [80.0, 60.0, 40.0, 20.0, 5.0]:
            tr.update((x, 0.0, 0.0))
            assert tr.fitness >= prev
            prev = tr.fitness


# ── current_target ─────────────────────────────────────────────────────────────

class TestCurrentTarget:
    def test_initial_target(self):
        assert t().current_target == (0.0, 0.0, 0.0)

    def test_advances_after_validation(self):
        tr = t()
        tr.update((5.0, 0.0, 0.0))
        assert tr.current_target == (100.0, 0.0, 0.0)

    def test_none_when_complete(self):
        tr = t()
        tr.on_finish(0)
        assert tr.current_target is None
