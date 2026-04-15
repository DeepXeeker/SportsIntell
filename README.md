# SportsIntell

Implementation of **SportsIntell: Computer Vision-Based Spatio-Temporal Motion Estimation Framework for Non-Linear Multi-Player Tracking in Dynamic Sports Environments**.

## What is implemented

- Historical trajectory embedding of 8-D track states to 32-D hidden features.
- **TFEN**: four residual dilated causal temporal blocks with dropout.
- **Temporal transformer**: six encoder layers, eight attention heads, hidden size 32.
- **Motion head** predicting `(dx, dy, dw, dh)` for the next frame.
- **DMAL** (Directional Motion Alignment Loss) computed from center and four corners.
- Inference-time tracker using IoU cost + Hungarian assignment.
- Baseline motion models for ablation:
  - no motion
  - constant-velocity / Kalman filter
  - vanilla transformer
  - full SportsIntell
- Dataset readers for MOT-style layouts used by SportsMOT and SoccerNet-Tracking.
- TrackEval export helper and minimal internal MOT metric helpers.
- Config-driven ablation runner.

## Repository layout

```text
sportsintell/
  data/           dataset readers and collation
  losses/         L1 and DMAL losses
  metrics/        simple metric helpers and TrackEval export
  models/         SportsIntell modules and ablation baselines
  tracker/        IoU/Hungarian association, Kalman filter, online tracker
  engine/         train/eval loops
  utils/          configuration, logging, geometry, checkpoints
configs/
  sportsmot.yaml
  soccernet.yaml
  ablations/
scripts/
  train.py
  evaluate.py
  infer.py
  run_ablations.py
  export_trackeval.py
docs/
  paper_analysis.md
  reproducibility_notes.md
tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Expected data layout

### SportsMOT

The official SportsMOT repository distributes data in MOTChallenge-style folders. Point the config `data.root` to a directory like:

```text
SportsMOT/
  train/
    video_001/
      img1/
      gt/gt.txt
      seqinfo.ini
  val/
  test/
```

### SoccerNet-Tracking

SoccerNet tracking annotations also use MOT-like CSV files. A normalized local layout is:

```text
SoccerNetTracking/
  train/
    game_x_clip_y/
      img1/
      gt/gt.txt
      seqinfo.ini
  val/
  test/
```

The repository contains conversion helpers so you can normalize the official download to this format.

## Quick start

### Train full SportsIntell

```bash
python scripts/train.py --config configs/sportsmot.yaml
```

### Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --config configs/sportsmot.yaml \
  --checkpoint outputs/sportsmot_full/best.pt
```

### Online tracking inference from detections

```bash
python scripts/infer.py \
  --config configs/sportsmot.yaml \
  --checkpoint outputs/sportsmot_full/best.pt \
  --sequence /path/to/sequence \
  --detections /path/to/det.txt \
  --save outputs/demo_tracks.txt
```

### Reproduce paper ablations

```bash
python scripts/run_ablations.py --group motion_models
python scripts/run_ablations.py --group tfen
python scripts/run_ablations.py --group dmal
python scripts/run_ablations.py --group trajectory_length
python scripts/run_ablations.py --group beta
```

## Notes on reproduction

The manuscript contains internal inconsistencies:
- the historical trajectory length is stated as **10** frames in the experimental setup but the ablation text reports the best result at **12** frames;
- the DMAL loss weight `beta` is stated as **0.3** in one section but **0.4** in the ablation text.

This repo defaults to:
- `history_len = 12`
- `beta = 0.4`

Both are easy to change in config files.

## Outputs

Training creates:
- `outputs/<run_name>/best.pt`
- `outputs/<run_name>/last.pt`
- `outputs/<run_name>/metrics.json`
- `outputs/<run_name>/predictions/` (optional)
- `outputs/<run_name>/trackeval/` (if exported)


