from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize prediction files into a TrackEval-friendly folder layout.")
    parser.add_argument("--predictions", required=True, help="Directory containing MOT-format prediction txt files")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--tracker_name", default="SportsIntell")
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    save_dir = Path(args.save_dir) / args.tracker_name / "data"
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for txt in pred_dir.glob("*.txt"):
        (save_dir / txt.name).write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
        count += 1
    print(f"Copied {count} prediction files to {save_dir}")


if __name__ == "__main__":
    main()
