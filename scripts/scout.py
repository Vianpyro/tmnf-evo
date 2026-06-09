"""
Scout run — drives A01 with full throttle and records the car's position
at each checkpoint.  Run this once before training.

Usage (from the project root):
    python scripts/scout.py
"""
import json
import sys
from pathlib import Path

from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from config import CHECKPOINT_FILE, DATA_DIR, TMI_SERVER


class ScoutClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self._positions = []

    def on_registered(self, iface: TMInterface) -> None:
        iface.execute_command("set countdown 0")
        print("Scout connected — drive A01 to record checkpoint positions.")

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        # Full throttle, no steering — enough to reach every checkpoint on A01.
        if _time >= 0:
            iface.set_input_state(accelerate=True, brake=False, steer=0)

    def on_checkpoint_count_changed(
        self, iface: TMInterface, current: int, target: int
    ) -> None:
        state = iface.get_simulation_state()
        pos = [state.position[0], state.position[1], state.position[2]]
        self._positions.append(pos)
        print(f"  checkpoint {current}/{target}  pos={pos}")

        if current == target:
            self._save()
            iface.close()

    def _save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, "w") as fh:
            json.dump({"checkpoints": self._positions}, fh, indent=2)
        print(f"\nSaved {len(self._positions)} checkpoint(s) → {CHECKPOINT_FILE}")


if __name__ == "__main__":
    run_client(ScoutClient(), TMI_SERVER)
