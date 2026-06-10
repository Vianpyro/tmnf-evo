import os
from pathlib import Path

# -- Network ----------------------------------------------------------
# Inputs (9):
#   speed, vx, vy, vz, sin(yaw), cos(yaw), dx, dy, dz
#   (dx/dy/dz = normalised vector to next unvalidated checkpoint)
# Outputs (3): steer, gas, brake  — all tanh in [-1, 1]
TOPOLOGY = [9, 8, 3]

# -- Genetic algorithm ------------------------------------------------
POP_SIZE = 60
MUTATION_RATE = 0.10   # probability of perturbing each weight
MUTATION_SIGMA = 0.20   # std dev of gaussian perturbation
ELITE_COUNT = 4     # top ~13% survive unchanged each generation
# Only the top PARENT_POOL_FRAC of the population can be parents.
# 0.25 = top 25% (aggressive, fast convergence, less diversity)
# 0.50 = top 50% (moderate, good starting point)
PARENT_POOL_FRAC = 0.5

# -- Evaluation -------------------------------------------------------
# After the 500ms grace period, kill the agent if speed stays below 2 km/h
# for this many consecutive steps (10ms each). 30 steps = 300ms sim time.
STUCK_STEPS_LIMIT = 30

# Time budget per agent per generation (milliseconds of simulation time).
# At set speed 10 : 300s sim = 30s real per agent = ~15 min/gen.
# At set speed 50 : 300s sim =  6s real per agent = ~3  min/gen.
MAX_STEP_MS = 300_000

# Bonus awarded for finishing the map, scaled by how quickly.
# finish at t=0          → full bonus (COMPLETION_TIME_BONUS)
# finish at t=MAX_STEP_MS → 0
# Tune relative to CHECKPOINT_BONUS * n_checkpoints on your map.
COMPLETION_TIME_BONUS = 2_000

# Fitness: cleared_checkpoints * BONUS - min_distance_to_next_checkpoint
CHECKPOINT_BONUS = 1_000.0

# -- Action space -----------------------------------------------------
# Disabling brake simplifies early training: the network can't stop the
# car, reducing backward-drift behaviour on slopes. Re-enable once agents
# consistently reach checkpoints.
ALLOW_BRAKE = False

# -- Speed reward -----------------------------------------------------
# Adds avg_speed_kmh * factor to fitness. Encourages forward momentum
# in addition to waypoint progress.
# Scale: 100 km/h avg × 0.5 = +50 pts vs CHECKPOINT_BONUS of 1000.
SPEED_BONUS_FACTOR = 0.5
DATA_DIR = Path(__file__).parent.parent / "data"
WAYPOINTS_FILE = DATA_DIR / "waypoints_a01.json"
# Default 3D proximity radius used by scout.py for game checkpoints.
# Virtual waypoints can use a larger value (e.g., 25.0) to be more forgiving.
WAYPOINT_RADIUS = 12.0

# -- TMInterface ------------------------------------------------------
# Overridden via environment variable in Docker / DevContainer.
# Docker Desktop resolves "host.docker.internal" automatically.
# On Linux Docker, extra_hosts in docker-compose.yml provides the mapping.
# -- TMInterface ------------------------------------------------------
# Server name matches the TMInterface instance (title bar shows "TMInterface0").
# Multiple instances would be TMInterface1, TMInterface2, etc.
TMI_SERVER = os.getenv("TMI_SERVER", "TMInterface0")
