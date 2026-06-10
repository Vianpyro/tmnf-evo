"""
Scout run — records waypoints along the track.

Drives the map with full throttle and saves the car position at each
game checkpoint as a waypoint entry (type=checkpoint). You can then
manually add virtual waypoints in the JSON to guide the car through
corners or U-turns that the straight-line heuristic can't handle.

Usage (from project root):
    python scripts/scout.py

Output: data/waypoints_MAPNAME.json
"""
import json
import sys
from pathlib import Path

from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from config import DATA_DIR, TMI_SERVER, WAYPOINT_RADIUS, WAYPOINTS_FILE


class ScoutClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self._waypoints = []

    def on_registered(self, iface: TMInterface) -> None:
        iface.give_up()
        print("Scout connected — recording checkpoint positions.")

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time >= 0:
            iface.set_input_state(accelerate=True, brake=False, steer=0)

    def on_checkpoint_count_changed(
        self, iface: TMInterface, current: int, target: int
    ) -> None:
        state = iface.get_simulation_state()
        pos = [state.position[0], state.position[1], state.position[2]]
        self._waypoints.append({
            "pos": pos,
            "radius": WAYPOINT_RADIUS,
            "note": f"checkpoint {current}/{target}",
        })
        print(f"  checkpoint {current}/{target}  pos={[round(v, 1) for v in pos]}")

        if current == target:
            self._save()
            iface.close()

    def _save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {"waypoints": self._waypoints}
        with open(WAYPOINTS_FILE, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nSaved {len(self._waypoints)} waypoint(s) → {WAYPOINTS_FILE}")
        print("\nTo add virtual waypoints for U-turns or complex routes,")
        print("manually insert entries before the relevant checkpoints:")
        print('  {"pos": [x, y, z], "radius": 25.0, "note": "virtual - go forward"}')


if __name__ == "__main__":
    run_client(ScoutClient(), TMI_SERVER)
