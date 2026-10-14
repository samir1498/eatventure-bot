"""Timing baseline for Detector.detect (perf-neutrality gate for the split).

Runs one full-screen detect() per golden frame (no category filter =
worst case), 3 timed passes after warmup. Re-run after the refactor;
means must stay within noise.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

import cv2

from vision.detector import Detector


def main():
    images = sorted((ROOT / "tests" / "images").glob("*.png"))

    assert images, "no golden frames found"

    detector = Detector()

    # Warmup: fills scale/resize caches so timed passes are steady-state.
    detector.detect(cv2.imread(str(images[0])))

    print(f"{'frame':<28} {'ms/pass':>8}")

    total = 0.0

    for image_path in images:
        image = cv2.imread(str(image_path))

        start = time.perf_counter()

        for _ in range(3):
            detector.detect(image)

        elapsed_ms = (time.perf_counter() - start) / 3 * 1000

        total += elapsed_ms

        print(f"{image_path.name:<28} {elapsed_ms:>8.1f}")

    print(f"{'MEAN':<28} {total / len(images):>8.1f}")


if __name__ == "__main__":
    main()
