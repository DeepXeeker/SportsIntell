from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ABLATIONS = {
    "motion_models": [
        "configs/ablations/no_motion.yaml",
        "configs/ablations/kalman_tracker.yaml",
        "configs/ablations/vanilla_transformer.yaml",
        "configs/ablations/sportsintell_full.yaml",
    ],
    "tfen": [
        "configs/ablations/without_tfen.yaml",
        "configs/ablations/sportsintell_full.yaml",
    ],
    "dmal": [
        "configs/ablations/without_dmal.yaml",
        "configs/ablations/sportsintell_full.yaml",
    ],
    "trajectory_length": [
        "configs/ablations/p3.yaml",
        "configs/ablations/p6.yaml",
        "configs/ablations/p12.yaml",
        "configs/ablations/p15.yaml",
        "configs/ablations/p18.yaml",
    ],
    "beta": [
        "configs/ablations/beta_01.yaml",
        "configs/ablations/beta_02.yaml",
        "configs/ablations/beta_03.yaml",
        "configs/ablations/beta_04.yaml",
        "configs/ablations/beta_05.yaml",
        "configs/ablations/beta_06.yaml",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(ABLATIONS.keys()))
    args = parser.parse_args()

    configs = ABLATIONS[args.group]
    for cfg in configs:
        print(f"Running {cfg}")
        subprocess.run(["python", "scripts/train.py", "--config", cfg], check=True)


if __name__ == "__main__":
    main()
