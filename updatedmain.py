"""
Mini VSaaS Analytics Module
----------------------------
An enhanced copy of main.py. Everything the original did still works, plus:
    1. Per-class live counts + cumulative UNIQUE counts (no. of people,
       no. of cars, etc.) driven off the tracker IDs.
    2. Compass directions (N / NE / E / SE / S / SW / W / NW) instead of
       up/down/left/right.
    3. A density heat map overlaid on the frame. Time spent in
       one spot builds heat; loitering counts extra.
    4. A dashboard drawn on the frame (FPS, zone
       occupancy, peak occupancy, active loiterers, per-class uniques) and a
       JSON session summary written on exit.

Keyboard while running:
    q  quit          h  toggle heat map        d  toggle dashboard
    z  toggle zone   b  toggle boxes           s  save a snapshot now

Run:
    python trial2.py (uses the footage set in this folder)
    python trial2.py --source other.mp4 (for a different file)
"""

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict, Counter
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "yolov8n.pt"
conf_threshold = 0.4 # ignore low-confidence detections
loiter_threshold = 5 # how long someone must stay in the zone to trigger an alert
alert_cooldown = 10 # don't re-alert the same id too often
trail_length = 30 # how many past positions to keep per object
source = "sample_video.mp4"
restricted_zone = [(260, 180), (700, 180), (700, 400), (260, 320)]
log_file = "events.csv"
snapshot_dir = "snapshots"
summary_file = "session_summary.json"
relevant_classes = {"person", "car", "truck", "backpack", "handbag", "suitcase"}
# sports mode: distance tracking
min_counted_step_pixels = 2.0     # smaller moves are box jitter, not running
max_plausible_step_pixels = 120.0 # bigger jumps are tracker ID swaps, not running
distance_leaderboard_size = 5     # how many players to list on the dashboard

# heat map configuration
heat_decay = 0.985 # per-frame fade (closer to 1.0 = longer memory)
heat_radius = 22 # size of the blob stamped per object (pixels)
heat_weight = 1.0 # heat added per object per frame
heat_loiter_bonus = 4.0 # extra heat for objects sitting in the zone
heat_alpha = 0.5 # overlay opacity
heat_show_min = 0.08 # only tint pixels above this (normalised) heat

"""
Creates snapshots folder. Only creates events.csv if it doesn't already exist.
"""
def setup_logging():
    os.makedirs(snapshot_dir, exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "event_type", "unique_id", "class", "direction", "snapshot_path"]
            )

"""
Called when something important happens ie. someone loiters too long.
"""
def log_event(event_type, unique_id, class_name, direction, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snap_path = os.path.join(snapshot_dir, f"{event_type}_{unique_id}_{timestamp}.jpg")
    cv2.imwrite(snap_path, frame)
    with open(log_file, "a", newline = "") as f:
        csv.writer(f).writerow(
            [timestamp, event_type, unique_id, class_name, direction, snap_path]
        )
    print(f"[ALERT] {event_type.upper()} | id = {unique_id} | {class_name} | {direction}")

"""
Determines whether an object's center point is within a restricted area.
"""
def point_in_zone(cx, cy, polygon):
    poly = np.array(polygon, dtype = np.int32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


# 8-way compass. Image y grows DOWNWARD, so we flip dy to make 'up' = North.
_COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

"""
Estimates the direction an object is moving.
"""
def get_direction(trail):
    if len(trail) < 2:
        return ""
    dx = trail[-1][0] - trail[0][0]
    dy = trail[-1][1] - trail[0][1]
    if abs(dx) < 5 and abs(dy) < 5:
        return "stationary"
    angle = math.degrees(math.atan2(-dy, dx)) # north-up convention
    idx = int((angle + 22.5) // 45) % 8
    return _COMPASS[idx]


"""
Sports mode: adds the distance a player moved since the previous frame to
their running total. Skips jitter and tracker ID-swap jumps.
"""
def accumulate_player_distance(distance_pixels_by_player_id, last_centroid_by_player_id,
                               player_id, current_centroid):
    previous_centroid = last_centroid_by_player_id.get(player_id)
    last_centroid_by_player_id[player_id] = current_centroid
    if previous_centroid is None:
        distance_pixels_by_player_id.setdefault(player_id, 0.0)
        return
    step_length_pixels = math.dist(previous_centroid, current_centroid)
    if min_counted_step_pixels <= step_length_pixels <= max_plausible_step_pixels:
        distance_pixels_by_player_id[player_id] = (
            distance_pixels_by_player_id.get(player_id, 0.0) + step_length_pixels
        )

"""
Add weight to a square patch of the heat accumulator around (cx, cy).
"""
def stamp_heat(heat, cx, cy, weight, radius=heat_radius):
    h, w = heat.shape
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x1 > x0 and y1 > y0:
        heat[y0:y1, x0:x1] += weight

"""
Render the accumulated heat map as a colored overlay on the frame.
"""
def render_heat(frame, heat):
    peak = float(heat.max())
    if peak <= 1e-5:
        return frame
    norm = np.clip(heat / peak, 0.0, 1.0)
    blurred = cv2.GaussianBlur(norm, (0, 0), sigmaX = 9)
    heat_u8 = (np.clip(blurred / max(blurred.max(), 1e-5), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    mask = (heat_u8 > int(heat_show_min * 255))[:, :, None]
    blended = cv2.addWeighted(frame, 1.0, colored, heat_alpha, 0)
    return np.where(mask, blended, frame)

"""
Draw the analytics dashboard and statistics on the video frame.
"""
def draw_dashboard(frame, stats):
    if stats.get("sports_mode"):
        lines = [
            f"FPS: {stats['fps']:.1f}",
            f"Players in frame: {stats['in_frame']}    Tracked total: {stats['total_unique']}",
            "-- distance covered (px) --",
        ]
        for player_id, distance_pixels in stats["distance_leaders"]:
            lines.append(f"  ID{player_id:<6} {distance_pixels:,.0f}")
    else:
        lines = [
            f"FPS: {stats['fps']:.1f}",
            f"In frame: {stats['in_frame']}    Tracked total: {stats['total_unique']}",
            f"Zone now: {stats['zone_now']}    Peak: {stats['zone_peak']}",
            f"Active loiterers: {stats['loiterers']}",
            "-- unique by class --",
        ]
        for cname, n in stats["unique_by_class"]:
            lines.append(f"  {cname:<9} {n}")
        if stats["in_frame_by_class"]:
            lines.append("-- in frame now --")
            for cname, n in stats["in_frame_by_class"]:
                lines.append(f"  {cname:<9} {n}")
    for cname, n in stats["unique_by_class"]:
        lines.append(f"  {cname:<9} {n}")
    if stats["in_frame_by_class"]:
        lines.append("-- in frame now --")
        for cname, n in stats["in_frame_by_class"]:
            lines.append(f"  {cname:<9} {n}")

    pad, lh = 10, 20
    w = 300 if stats.get("sports_mode") else 260
    h = pad * 2 + lh * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + w, 8 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = 8 + pad + 14
    for i, text in enumerate(lines):
        color = (0, 255, 255) if text.startswith("--") else (255, 255, 255)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += lh
    return frame

"""
Write the final session analytics and statistics to a JSON file.
"""
def write_summary(stats, id_to_class, unique_by_class, elapsed):
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(stats.get("source")),
        "duration_seconds": round(elapsed, 1),
        "frames_processed": stats["frames"],
        "avg_fps": round(stats["frames"] / elapsed, 2) if elapsed > 0 else 0,
        "total_unique_objects": len(id_to_class),
        "unique_by_class": {k: len(v) for k, v in unique_by_class.items()},
        "peak_zone_occupancy": stats["zone_peak"],
        "loitering_alerts": stats["loiter_alerts"],
    }
    if stats.get("sports_mode"):
        summary["mode"] = "sports"
        summary["distance_pixels_by_player"] = {
            f"ID{player_id}": round(distance_pixels, 1)
            for player_id, distance_pixels in sorted(
                stats["distance_pixels_by_player_id"].items(), key=lambda kv: -kv[1]
            )
        }
        summary["distance_note"] = (
            "Pixel distance from each player's centroid path. Not meters, and a "
            "player who leaves and re-enters frame may get a new ID."
        )
        summary.pop("peak_zone_occupancy")
        summary.pop("loitering_alerts")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent = 2)
    print(f"\n[SUMMARY] written to {summary_file}")
    print(json.dumps(summary, indent = 2))

"""
Tracks how long an object has been inside the restricted zone. Returns
(is_loitering, alert_triggered) and saves a highlighted snapshot when an alert fires.
"""
def update_loitering_status_and_alert(unique_id, inside, class_name, direction,
                                      frame, box_corners, state):
    if not inside:
        state["zone_entry_time"].pop(unique_id, None)
        return False, False

    state["zone_entry_time"].setdefault(unique_id, time.time())

    dwell = time.time() - state["zone_entry_time"][unique_id]

    if dwell <= loiter_threshold:
        return False, False

    alert_triggered = False

    if (time.time() - state["last_alert_time"][unique_id] > alert_cooldown):
        x1, y1, x2, y2 = box_corners
        snapshot = frame.copy()
        cv2.rectangle(snapshot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(snapshot, f"ID {unique_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        log_event(
            "loitering",
            unique_id,
            class_name,
            direction,
            snapshot,
        )

        state["last_alert_time"][unique_id] = time.time()
        alert_triggered = True

    return True, alert_triggered

def main(source, save_video = False, headless = False, max_frames = 0, sports_mode = False):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Could not find footage file: '{source}'")
        print("Update `source` at the top of this script, or pass --source path/to/video.mp4")
        return

    setup_logging()
    model = YOLO(model_path)

    track_trails = defaultdict(list)      # id -> [(cx, cy), ...]
    zone_entry_time = {}                   # id -> time first seen inside zone
    last_alert_time = defaultdict(float)   # id -> last alert time
    id_to_class = {}                       # id -> class name (last seen)
    unique_by_class = defaultdict(set)     # class -> {ids ever seen}
    distance_pixels_by_player_id = {}      # sports mode: id -> total pixels covered
    last_centroid_by_player_id = {}        # sports mode: id -> (cx, cy) last frame

    heat = None
    writer = None

    show_heat, show_dash, show_zone, show_boxes = True, True, not sports_mode, True

    # rolling stats
    frames = 0
    zone_peak = 0
    loiter_alerts = 0
    t_start = time.time()
    fps = 0.0
    fps_t = time.time()
    fps_frames = 0

    results_stream = model.track(
        source = source, persist = True, conf = conf_threshold,
        stream = True, verbose = False,
        classes = [0] if sports_mode else None
    )

    for result in results_stream:
        frame = result.orig_img
        h, w = frame.shape[:2]
        if heat is None:
            heat = np.zeros((h, w), dtype = np.float32)
        heat *= heat_decay

        frames += 1
        fps_frames += 1
        if time.time() - fps_t >= 0.5:
            fps = fps_frames / (time.time() - fps_t)
            fps_t = time.time()
            fps_frames = 0

        in_frame_counter = Counter()
        zone_now = 0
        loiterers = 0

        boxes = result.boxes
        if boxes.id is not None:
            for box, unique_id, class_id, conf in zip(
                boxes.xyxy, boxes.id, boxes.cls, boxes.conf
            ):
                class_name = model.names[int(class_id)]
                if class_name not in relevant_classes:
                    continue

                x1, y1, x2, y2 = box.tolist()
                unique_id = int(unique_id)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # counting bookkeeping
                id_to_class[unique_id] = class_name
                unique_by_class[class_name].add(unique_id)
                in_frame_counter[class_name] += 1

                trail = track_trails[unique_id]
                trail.append((cx, cy))
                if len(trail) > trail_length:
                    trail.pop(0)
                direction = get_direction(trail)
                if sports_mode:
                    accumulate_player_distance(distance_pixels_by_player_id,
                                               last_centroid_by_player_id,
                                               unique_id, (cx, cy))

                inside = (not sports_mode and point_in_zone(cx, cy, restricted_zone))

                if inside:
                    zone_now += 1

                is_loitering, alert_triggered = update_loitering_status_and_alert(unique_id, inside,
                                            class_name, direction, frame, (x1, y1, x2, y2),
                    {
                        "zone_entry_time": zone_entry_time,
                        "last_alert_time": last_alert_time,
                    },
                )

                if is_loitering:
                    loiterers += 1

                if alert_triggered:
                    loiter_alerts += 1

                # feed the heat map (extra heat when loitering)
                stamp_heat(heat, cx, cy,
                           heat_weight + (heat_loiter_bonus if is_loitering else 0.0))

                # per-object box + label
                if show_boxes:
                    color = (0, 165, 255) if inside else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    if sports_mode:
                        label = (f"ID{unique_id} {direction} "
                                 f"{distance_pixels_by_player_id.get(unique_id, 0.0):,.0f}px")
                    else:
                        label = f"ID{unique_id} {class_name} {direction}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        zone_peak = max(zone_peak, zone_now)

        # ---- heat overlay goes down first, annotations on top ----
        if show_heat:
            frame = render_heat(frame, heat)

        if show_zone:
            cv2.polylines(frame, [np.array(restricted_zone, np.int32)], True, (0, 0, 225), 2)
            cv2.putText(frame, "RESTRICTED ZONE", restricted_zone[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)

        if show_dash:
            stats = {
                "fps": fps,
                "in_frame": sum(in_frame_counter.values()),
                "total_unique": len(id_to_class),
                "zone_now": zone_now,
                "zone_peak": zone_peak,
                "loiterers": loiterers,
                "unique_by_class": sorted(
                    ((c, len(ids)) for c, ids in unique_by_class.items()),
                    key=lambda kv: -kv[1],
                ),
                "in_frame_by_class": in_frame_counter.most_common(),
                                "sports_mode": sports_mode,
                                "distance_leaders": sorted(
                                distance_pixels_by_player_id.items(), key=lambda kv: -kv[1]
                )[:distance_leaderboard_size],
            }
            frame = draw_dashboard(frame, stats)

        if save_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter("annotated_output.mp4", fourcc, 25, (w, h))
            writer.write(frame)

        # headless mode: no window, just process (optionally capped) frames
        if headless:
            if max_frames and frames >= max_frames:
                break
            continue

        cv2.imshow("Mini VSaaS Analytics - trial2", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            show_heat = not show_heat
        elif key == ord("d"):
            show_dash = not show_dash
        elif key == ord("z"):
            show_zone = not show_zone
        elif key == ord("b"):
            show_boxes = not show_boxes
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(snapshot_dir, f"manual_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[SNAPSHOT] {path}")

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    write_summary(
        {"source": source, "frames": frames, "zone_peak": zone_peak,
         "loiter_alerts": loiter_alerts, "sports_mode": sports_mode,
         "distance_pixels_by_player_id": distance_pixels_by_player_id},
        id_to_class, unique_by_class, elapsed,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini VSaaS analytics demo (trial2)")
    parser.add_argument("--source", default=source,
                        help=f"Path to a video file, or a webcam index like 0 (default: {source})")
    parser.add_argument("--save-video", action="store_true",
                        help="Also write annotated_output.mp4")
    parser.add_argument("--headless", action="store_true",
                        help="Run without a display window (for servers / testing)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = whole video)")
    parser.add_argument("--sports", action="store_true",
                        help="Sports mode: track people only")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    main(source, save_video = args.save_video,
         headless = args.headless, max_frames = args.max_frames,
         sports_mode = args.sports)