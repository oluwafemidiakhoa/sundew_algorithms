# video_gate.py
from __future__ import annotations

import argparse
import csv
import time
from typing import Dict, Optional

import cv2
import numpy as np

from sundew import SundewAlgorithm
from sundew.config_presets import get_preset


def _energy_float(algo: SundewAlgorithm) -> float:
    e = getattr(algo, "energy", 0.0)
    v = getattr(e, "value", None)
    try:
        return float(v if v is not None else e)
    except Exception:
        return 0.0


class EMA:
    def __init__(self, alpha: float, init: float = 0.0) -> None:
        self.a = float(alpha)
        self.y = float(init)

    def update(self, x: float) -> float:
        self.y = (1.0 - self.a) * self.y + self.a * float(x)
        return self.y


def frame_to_event(
    gray: np.ndarray,
    prev_gray: Optional[np.ndarray],
    ema_ctx: EMA,
) -> Dict[str, float]:
    """Convert a grayscale frame into Sundew feature dict."""
    # Motion-based anomaly in [0,1]
    if prev_gray is None:
        motion = 0.0
    else:
        motion = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
    anomaly = float(np.clip(motion * 3.0, 0.0, 1.0))

    # Context relevance as EMA of anomaly
    context = float(np.clip(ema_ctx.update(anomaly), 0.0, 1.0))

    # Urgency via average gradient magnitude (edges) in [0,1]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    urgency = float(np.clip(np.mean(grad) / 255.0 * 2.0, 0.0, 1.0))

    # Magnitude from frame stddev mapped to [0,100]
    std = float(np.std(gray))  # ~0..~64 typical
    magnitude = float(np.clip((std / 64.0) * 100.0, 0.0, 100.0))

    return {
        "magnitude": magnitude,
        "anomaly_score": anomaly,
        "context_relevance": context,
        "urgency": urgency,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Live video gating with Sundew")
    ap.add_argument("--source", required=True, help="0 for webcam, or a path to a video file")
    ap.add_argument("--preset", default="tuned_v2", help="Sundew config preset")
    ap.add_argument("--frames", type=int, default=600, help="Max frames to process (0 = unlimited)")
    ap.add_argument("--show", action="store_true", help="Show a live preview window")
    ap.add_argument("--save", type=str, default=None, help="Optional CSV log path")
    ap.add_argument("--width", type=int, default=None, help="Resize width for processing/preview")
    args = ap.parse_args()

    # Load algorithm
    cfg = get_preset(args.preset)
    algo = SundewAlgorithm(cfg)

    # Open source (webcam index or file path)
    src: object
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    # On Windows, forcing DirectShow helps some webcams open reliably
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    # CSV logging
    writer = None
    fhandle = None
    if args.save:
        fhandle = open(args.save, "w", newline="", encoding="utf-8")
        writer = csv.writer(fhandle)
        writer.writerow(
            [
                "frame",
                "activated",
                "magnitude",
                "anomaly",
                "context",
                "urgency",
                "threshold",
                "energy",
                "significance",
                "processing_time",
                "energy_consumed",
            ]
        )

    ema_ctx = EMA(alpha=0.05)
    prev_gray: Optional[np.ndarray] = None

    n = 0
    activations = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.width:
            h, w = frame.shape[:2]
            new_w = args.width
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ev = frame_to_event(gray, prev_gray, ema_ctx)
        prev_gray = gray

        res = algo.process(ev)
        activated = res is not None
        if activated:
            activations += 1

        # Overlay
        if args.show:
            overlay = frame.copy()
            color = (0, 200, 0) if activated else (0, 0, 200)
            label = "ACTIVATED" if activated else "idle"
            cv2.putText(
                overlay,
                f"thr={algo.threshold:.3f} | energy={_energy_float(algo):.1f} | {label}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            if activated:
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), color, 3)
            cv2.imshow("Sundew Gating", overlay)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        # Log
        if writer:
            energy = _energy_float(algo)
            if res is None:
                writer.writerow(
                    [
                        n,
                        0,
                        ev["magnitude"],
                        ev["anomaly_score"],
                        ev["context_relevance"],
                        ev["urgency"],
                        f"{algo.threshold:.6f}",
                        f"{energy:.6f}",
                        "",
                        "",
                        "",
                    ]
                )
            else:
                writer.writerow(
                    [
                        n,
                        1,
                        ev["magnitude"],
                        ev["anomaly_score"],
                        ev["context_relevance"],
                        ev["urgency"],
                        f"{algo.threshold:.6f}",
                        f"{energy:.6f}",
                        f"{res.significance:.6f}",
                        f"{res.processing_time:.6f}",
                        f"{res.energy_consumed:.6f}",
                    ]
                )

        n += 1
        if args.frames and n >= args.frames:
            break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    if fhandle:
        fhandle.close()

    dt = time.time() - t0
    rate = (activations / n) if n else 0.0
    print(f"Frames: {n} | Activations: {activations} | Activation rate: {rate:.3f} | {dt:.1f}s")


if __name__ == "__main__":
    main()
