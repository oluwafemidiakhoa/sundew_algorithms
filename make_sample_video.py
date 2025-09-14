# make_sample_video.py
import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=300, help="total frames to write")
ap.add_argument("--fps", type=int, default=30, help="frames per second")
ap.add_argument("--out", default="sample.mp4", help="output MP4 path")
args = ap.parse_args()

w, h = 640, 360
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

for i in range(args.frames):
    frame = np.full((h, w, 3), 180, np.uint8)
    x = int((i * 5) % (w - 120))
    y = 120 + int(40 * np.sin(i / 10))
    cv2.rectangle(frame, (x, y), (x + 120, y + 60), (40, 40, 40), -1)
    cv2.putText(frame, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2)
    out.write(frame)

out.release()
print(f"Wrote {args.out} with {args.frames} frames at {args.fps} fps")
