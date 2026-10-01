"""
Turns a stretch of annotated_output.mp4 into a small looping GIF for the README.

Run:
    python make_heatmap_gif.py
    python make_heatmap_gif.py --start-seconds 5 --duration-seconds 10
"""

import argparse
import cv2
from PIL import Image

default_input_video_path = "annotated_output.mp4"
default_output_gif_path = "sports_heatmap.gif"
gif_width_pixels = 480 # smaller = lighter file
gif_frames_per_second = 8 # fewer frames = lighter file


def make_heatmap_gif(input_video_path, output_gif_path, start_seconds, duration_seconds):
    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print(f"Could not open '{input_video_path}'")
        return

    source_frames_per_second = video_capture.get(cv2.CAP_PROP_FPS) or 25
    first_frame_index = int(start_seconds * source_frames_per_second)
    last_frame_index = int((start_seconds + duration_seconds) * source_frames_per_second)
    keep_every_nth_frame = max(1, round(source_frames_per_second / gif_frames_per_second))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)
    gif_frames = []
    current_frame_index = first_frame_index

    while current_frame_index < last_frame_index:
        frame_was_read, frame_bgr = video_capture.read()
        if not frame_was_read:
            break
        if (current_frame_index - first_frame_index) % keep_every_nth_frame == 0:
            frame_height, frame_width = frame_bgr.shape[:2]
            gif_height_pixels = int(frame_height * gif_width_pixels / frame_width)
            resized_frame_bgr = cv2.resize(frame_bgr, (gif_width_pixels, gif_height_pixels),
                                           interpolation=cv2.INTER_AREA)
            resized_frame_rgb = cv2.cvtColor(resized_frame_bgr, cv2.COLOR_BGR2RGB)
            gif_frames.append(Image.fromarray(resized_frame_rgb))
        current_frame_index += 1

    video_capture.release()

    if not gif_frames:
        print("No frames were read. Check --start-seconds against the video length.")
        return

    milliseconds_per_gif_frame = int(1000 * keep_every_nth_frame / source_frames_per_second)
    gif_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=milliseconds_per_gif_frame,
        loop=0, # loop forever
        optimize=True,
    )
    print(f"Saved {output_gif_path} ({len(gif_frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a README GIF from the annotated video")
    parser.add_argument("--input", default=default_input_video_path)
    parser.add_argument("--output", default=default_output_gif_path)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    args = parser.parse_args()
    make_heatmap_gif(args.input, args.output, args.start_seconds, args.duration_seconds)