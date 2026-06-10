"""
Save and load population checkpoints.

Format:
  {
    "generation": 42,
    "topology": [9, 8, 3],
    "individuals": [                         # sorted best → worst
      {"fitness": -192.0, "weights": [...]}
    ]
  }
"""
import json
from pathlib import Path
from typing import Optional

import tmnf_evo


def save(population: tmnf_evo.Population, fitnesses: list[float], path: Path) -> None:
    """Save the current population before evolve() is called.

    Weights and fitnesses must be captured before evolve() since evolve resets
    fitness values to 0 in the next generation.
    """
    weights = population.get_all_weights()
    individuals = sorted(
        [{"fitness": float(f), "weights": w}
         for w, f in zip(weights, fitnesses)],
        key=lambda x: x["fitness"],
        reverse=True,
    )
    data = {
        "generation": population.generation,
        "topology": list(population.topology),
        "all_time_best": {
            "fitness": float(population.all_time_best_fitness),
            "weights": population.get_all_time_best_weights(),
        },
        "individuals": individuals,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh)

    best = individuals[0]["fitness"] if individuals else float("-inf")
    print(
        f"[ckpt] gen {population.generation:03d} → {path.name}  best={best:.1f}")


def load(population: tmnf_evo.Population, path: Path) -> int:
    """Load weights into population. Returns the saved generation number."""
    with open(path) as fh:
        data = json.load(fh)

    saved_topo = data.get("topology", [])
    current_topo = list(population.topology)
    if saved_topo and saved_topo != current_topo:
        raise ValueError(
            f"Topology mismatch — checkpoint: {saved_topo}, current: {current_topo}"
        )

    individuals = data["individuals"]
    n = min(len(individuals), population.size)
    for i in range(n):
        population.set_weights_at(i, individuals[i]["weights"])

    if "all_time_best" in data:
        atb = data["all_time_best"]
        population.set_all_time_best(atb["weights"], atb["fitness"])

    gen = data["generation"]
    print(f"[ckpt] loaded gen {gen:03d} from {path}  ({n} individuals)"
          f"  all_time_best={data.get('all_time_best', {}).get('fitness', '?'):.1f}"
          if "all_time_best" in data else
          f"[ckpt] loaded gen {gen:03d} from {path}  ({n} individuals)")
    return gen
