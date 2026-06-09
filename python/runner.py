"""
Main training loop.

Usage (from the python/ directory):
    python runner.py

Prerequisites:
  1. TMInterface running with A01 loaded and the car at the start line.
  2. `python scripts/scout.py` has been run to generate data/checkpoints_a01.json.
  3. `maturin develop` has built the tmnf_evo extension.
"""
import sys
from pathlib import Path

from tminterface.client import run_client

sys.path.insert(0, str(Path(__file__).parent))
from client import EvoClient
from config import (
    ELITE_COUNT,
    MUTATION_RATE,
    MUTATION_SIGMA,
    POP_SIZE,
    TMI_SERVER,
    TOPOLOGY,
)

import tmnf_evo

GENERATIONS = 100


def main() -> None:
    population = tmnf_evo.Population(
        size=POP_SIZE,
        topology=TOPOLOGY,
        mutation_rate=MUTATION_RATE,
        mutation_sigma=MUTATION_SIGMA,
        elite_count=ELITE_COUNT,
    )

    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen} / {GENERATIONS - 1} ===")
        client = EvoClient(population)

        # run_client blocks until the client calls iface.close(),
        # which happens once all agents in the generation are evaluated.
        run_client(client, TMI_SERVER)

        if not client.fitnesses:
            print("[error] No fitnesses recorded — did the client connect?")
            break

        best = max(client.fitnesses)
        avg  = sum(client.fitnesses) / len(client.fitnesses)
        print(f"[gen {gen:03d}] best={best:8.1f}  avg={avg:8.1f}")

        population.evolve(client.fitnesses)


if __name__ == "__main__":
    main()
