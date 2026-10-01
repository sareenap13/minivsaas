# minivsaas

A small demo that mimics the "Intelligent Analytics" features of a commercial
Video-Surveillance-as-a-Service (VSaaS) product, built on top of YOLOv8.
It also has a sports mode that tracks players in game footage and measures how
far each one runs. See [Sports mode](#sports-mode).

## Features

- Object detection and classification (YOLOv8)
- Persistent multi-object tracking (YOLOv8's built-in ByteTrack)
- Direction-of-travel estimation from centroid trails
- Restricted-zone loitering detection (dwell-time threshold + alert cooldown)
- Event logging with snapshots (CSV log, mimics an event feed)

`updatedmain.py` extends `main.py` with:

- Per-class live and cumulative unique counts (people, cars, etc.)
- 8-way compass directions (N / NE / E / SE / S / SW / W / NW)
- A decaying density heat map overlaid on the frame
- An on-frame dashboard (FPS, zone occupancy, peak occupancy, active
  loiterers, per-class counts) and a JSON session summary on exit

## Requirements

- Python 3.9+
- [ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8)
- opencv-python
- numpy

```bash
pip install ultralytics opencv-python numpy
```

The YOLOv8 nano weights (`yolov8n.pt`) are downloaded automatically by
`ultralytics` on first run if not already present locally.

## Usage

Both scripts expect a video file in the working directory (default:
`sample_video.mp4`) and default settings can be edited at the top of each
file (confidence threshold, loiter threshold, restricted zone coordinates,
etc.).

### main.py

```bash
python main.py                       # uses the default footage
python main.py --source other.mp4    # or a different video file
```

### updatedmain.py

```bash
python updatedmain.py
python updatedmain.py --source other.mp4
python updatedmain.py --save-video           # also writes annotated_output.mp4
python updatedmain.py --headless             # no display window (servers/testing)
python updatedmain.py --headless --max-frames 500
python updatedmain.py --sports --source game.mp4 --save-video # sports mode
```

Keyboard controls while `updatedmain.py` is running:

| Key | Action |
| --- | --- |
| `q` | Quit |
| `h` | Toggle heat map |
| `d` | Toggle dashboard |
| `z` | Toggle restricted-zone outline |
| `b` | Toggle detection boxes |
| `s` | Save a snapshot immediately |

## Output

- `events.csv` — append-only log of loitering alerts
- `snapshots/` — JPEG snapshots captured on each alert
- `session_summary.json` — written by `updatedmain.py` on exit, summarizing
  the session (duration, frame count, average FPS, unique object counts,
  peak zone occupancy, loitering alerts)

These are runtime-generated artifacts and are excluded from version control
via `.gitignore`.

## Sports mode

<!-- heat map GIF goes here -->

I adapted the tracking pipeline to game footage. With `--sports`, the script tracks only people, turns off the restricted zone and loitering alerts, and measures how far each player moves.

### What it adds

- Distance covered per player, shown on each box and in a live top-5 leaderboard
- A heat map that builds over the whole clip and shows where play actually happened
- Per-player distances in `session_summary.json`, sorted from most to least

### How distance is measured

Each frame, the script measures how far a player's center moved since the last frame and adds it to their total. Steps under 2 pixels are skipped because detection boxes wobble slightly even when someone stands still. Steps over 120 pixels are skipped because a jump that large almost always means the tracker swapped IDs between two players. Both thresholds are set at the top of `updatedmain.py`.

### Limitations

- Distances are in pixels, not meters. A player far from the camera covers fewer pixels for the same run.
- The camera needs to stay still. If it pans, every player appears to move.
- A player who leaves the frame and comes back can get a new ID, which splits their distance across two entries.

### Next steps

- Convert pixels to meters by mapping the field's corners to real coordinates (a homography).
- Identify players by jersey number so IDs stay stable across the whole game.