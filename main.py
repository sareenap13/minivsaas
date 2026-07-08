"""
Mini VSaaS Analytics Module
----------------------------
A small demo that is similar to "Intelligent Analytics" features of a commercial
video-surveillance-as-a-service product:
    - Object detection and classification (YOLOv8)
    - Persistent tracking (YOLOv8 built-in ByteTrack)
    - Direction of travel (centroid trail analysis)
    - Restricted-zone loitering detection (dwell-time threshold)
    - Real-time event logging + snapshots (CSV log, mimics an event feed)

This version is built to run again against pre-recorded footage. Webcam
input is also supported by the code, but isn't part of the current workflow.
Footage is the active path for now.

Run:
    python main.py (uses the footage set in this folder)
    pythin main.py --source other.mp4 (for a different file)
"""

import argparse
import csv
import os
import time
from collections import defaultdict
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
# security relevant
relevant_classes = {"person", "car", "truck", "backpack", "handbag", "suitcase"}

"""
Creates snapshots folder. Only creates events.csv if it doesn't already exist.
"""
def setup_logging():
    os.makedirs(snapshot_dir, exist_ok = True)
    if not os.path.exists(log_file):
        with open(log_file, "w", newline = "") as f:
            csv.writer(f).writerow(
                ["timestamp", "event_type", "unique_id", "class", "snapshot_path"]
            )

"""
Called when something important happens ie. someone loiters too long.
"""
def log_event(event_type, unique_id, class_name, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snap_path = os.path.join(snapshot_dir, f"{event_type}_{unique_id}_{timestamp}.jpg")
    cv2.imwrite(snap_path, frame)
    with open(log_file, "a", newline = "") as f:
        csv.writer(f).writerow([timestamp, event_type, unique_id, class_name, snap_path])
    print(f"[ALERT] {event_type.upper()} | unique_id = {unique_id} | class = {class_name}")

"""
Determines whether an object's center point is within a restricted area.
"""
def point_in_zone(cx, cy, polygon):
    poly = np.array(polygon, dtype = np.int32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

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
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"

def main(source):
    # check that the video exists
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Could not find footage file: '{source}")
        print("Update SOURCE at the top of this script, or pass --source path/to/your/video.mph")
        return

    # make sure event logging is ready
    setup_logging()
    model = YOLO(model_path)

    track_trails = defaultdict(list)     # unique_id -> list of (cx, cy)
    zone_entry_time = {}                 # unique_id -> time first seen inside zone
    last_alert_time = defaultdict(float) # unique_id -> last time an alert fired

    # start processing the video
    results_stream = model.track(
        source = source, persist = True, conf = conf_threshold, stream = True, verbose = False
    )

    # executes for every frame in the video
    for result in results_stream:
        frame = result.orig_img
        # draws restricted zone
        cv2.polylines(frame, [np.array(restricted_zone, np.int32)], True, (0, 0, 225), 2)
        cv2.putText(frame, "RESTRICTED ZONE", restricted_zone[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)

        # contains every detected object in the frame
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
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) # compute object's center

                trail = track_trails[unique_id]
                trail.append((cx, cy))

                # limits trail length
                if len(trail) > trail_length:
                    trail.pop(0)
                direction = get_direction(trail)

                # zone/loitering logic
                inside = point_in_zone(cx, cy, restricted_zone)
                if inside:
                    zone_entry_time.setdefault(unique_id, time.time())
                    dwell = time.time() - zone_entry_time[unique_id]
                    if (
                        dwell > loiter_threshold
                        and time.time() - last_alert_time[unique_id] > alert_cooldown
                    ):
                        snapshot = frame.copy()

                        cv2.rectangle(
                            snapshot,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 255),  # red box
                            3
                        )

                        cv2.putText(
                            snapshot,
                            f"ID {unique_id}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        log_event("loitering", unique_id, class_name, frame)
                        last_alert_time[unique_id] = time.time()
                else:
                    zone_entry_time.pop(unique_id, None)

                # draw box around object
                color = (0, 165, 255) if inside else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID{unique_id} {class_name} {direction}"
                cv2.putText(frame, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show processed film
        cv2.imshow("Mini VSaaS Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Mini VSaaS analytics demo")
    parser.add_argument(
        "--source", default = source,
        help =f"Path to a video file, or a webcam index like 0 (default: {source}, unused for now)"
    )
    args = parser.parse_args()
    # supports a webcam index for future use, even though it's not part of the current workflow
    source = int(args.source) if str(args.source).isdigit() else args.source
    main(source)