import os
from pathlib import Path

# -- Network ----------------------------------------------------------
# Inputs (9):
#   speed, vx, vy, vz, sin(yaw), cos(yaw), dx, dy, dz
#   (dx/dy/dz = normalised vector to next unvalidated checkpoint)
# Outputs (3): steer, gas, brake  — all tanh in [-1, 1]
TOPOLOGY = [9, 8, 3]

# -- Genetic algorithm ------------------------------------------------
POP_SIZE       = 30
MUTATION_RATE  = 0.10   # probability of perturbing each weight
MUTATION_SIGMA = 0.20   # std dev of gaussian perturbation
ELITE_COUNT    = 2

# -- Evaluation -------------------------------------------------------
# Time budget per agent per generation (milliseconds).
MAX_STEP_MS = 30_000

# Fitness: cleared_checkpoints * BONUS - min_distance_to_next_checkpoint
CHECKPOINT_BONUS = 1_000.0

# -- Paths ------------------------------------------------------------
DATA_DIR        = Path(__file__).parent.parent / "data"
CHECKPOINT_FILE = DATA_DIR / "checkpoints_a01.json"

# -- TMInterface ------------------------------------------------------
# Change to "host.docker.internal" when running from inside the DevContainer.
TMI_HOST = "localhost"
TMI_PORT = 8477

# -- TMInterface ------------------------------------------------------
# Server name matches the TMInterface instance (title bar shows "TMInterface0").
# Multiple instances would be TMInterface1, TMInterface2, etc.
TMI_SERVER = os.getenv("TMI_SERVER", "TMInterface0")
