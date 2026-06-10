"""
Main training loop.

Usage:
  python runner.py              # fresh run
  python runner.py --resume     # resume from data/latest.json
"""
import tmnf_evo
from config import (
    DATA_DIR,
    ELITE_COUNT,
    MUTATION_RATE,
    MUTATION_SIGMA,
    PARENT_POOL_FRAC,
    POP_SIZE,
    TMI_SERVER,
    TOPOLOGY,
)
from client import EvoClient
import checkpoint
import argparse
import sys
from pathlib import Path

from tminterface.client import run_client

sys.path.insert(0, str(Path(__file__).parent))


GENERATIONS = 100
LATEST_PATH = DATA_DIR / "latest.json"
PERIODIC_EVERY = 10  # save a dated snapshot every N generations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="resume from data/latest.json")
    args = parser.parse_args()

    population = tmnf_evo.Population(
        size=POP_SIZE,
        topology=TOPOLOGY,
        mutation_rate=MUTATION_RATE,
        mutation_sigma=MUTATION_SIGMA,
        elite_count=ELITE_COUNT,
        parent_pool_frac=PARENT_POOL_FRAC,
    )

    start_gen = 0
    if args.resume:
        if not LATEST_PATH.exists():
            print("[warn] --resume: data/latest.json not found, starting fresh.")
        else:
            start_gen = checkpoint.load(population, LATEST_PATH) + 1
            print(f"[info] resuming from generation {start_gen}")

    for gen in range(start_gen, start_gen + GENERATIONS):
        print(f"\n=== Generation {gen} ===")
        client = EvoClient(population)
        run_client(client, TMI_SERVER)

        if not client.fitnesses:
            print("[error] No fitnesses recorded — did the client connect?")
            break

        best = max(client.fitnesses)
        avg = sum(client.fitnesses) / len(client.fitnesses)
        atb = population.all_time_best_fitness
        print(
            f"[gen {gen:03d}] best={best:8.1f}  avg={avg:8.1f}  all_time={atb:8.1f}")

        # Save before evolve — weights correspond to these fitnesses.
        checkpoint.save(population, client.fitnesses, LATEST_PATH)
        if gen % PERIODIC_EVERY == 0:
            checkpoint.save(population, client.fitnesses,
                            DATA_DIR / f"checkpoint_gen{gen:03d}.json")

        population.evolve(client.fitnesses)


if __name__ == "__main__":
    main()
