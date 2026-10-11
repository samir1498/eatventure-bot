"""
Detection regression: runs the detector on tests/images and compares with
tests/golden/expectations.json (--update rewrites; REVIEW the diff before committing).
Images with no golden entry become a WARNING, not a failure. Don't use "screen.png" as
a fixture: that's the name template_selector overwrites.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from core import log                        # noqa: E402
from vision.detector import Detector        # noqa: E402

IMAGES_DIR = ROOT / "tests" / "images"
GOLDEN_PATH = ROOT / "tests" / "golden" / "expectations.json"

# Tolerances: two-stage refinement may shift the
# point by a few pixels, and that is not a regression.
POSITION_TOLERANCE = 10
CONFIDENCE_TOLERANCE = 0.03


def detect_all(detector):

    results = {}

    for image_path in sorted(IMAGES_DIR.glob("*.png")):

        image = cv2.imread(str(image_path))

        if image is None:

            print(f"  ! não carregou: {image_path.name}")

            continue

        detections = detector.detect(image)

        results[image_path.name] = [
            {
                "category": d["category"],
                "x": d["x"],
                "y": d["y"],
                "confidence": round(d["confidence"], 4),
                "color_similarity": round(
                    d["color_similarity"], 4
                ),
            }
            for d in detections
        ]

    return results


def matches(expected, actual):

    return (
        expected["category"] == actual["category"]
        and abs(expected["x"] - actual["x"])
        <= POSITION_TOLERANCE
        and abs(expected["y"] - actual["y"])
        <= POSITION_TOLERANCE
    )


def describe(detection):

    return (
        f"{detection['category']} "
        f"@({detection['x']},{detection['y']}) "
        f"F={detection['confidence']:.3f} "
        f"C={detection['color_similarity']:.3f}"
    )


def compare(expected_all, actual_all):

    failures = []
    warnings = []

    for name in sorted(
        set(expected_all) | set(actual_all)
    ):

        expected = expected_all.get(name)
        actual = actual_all.get(name, [])

        if expected is None:

            warnings.append(
                f"{name}: imagem nova, sem esperado "
                f"({len(actual)} detecções). "
                f"Rode --update."
            )

            continue

        remaining = list(actual)

        for item in expected:

            found = None

            for candidate in remaining:

                if matches(item, candidate):

                    found = candidate

                    break

            if found is None:

                failures.append(
                    f"{name}: PERDEU {describe(item)}"
                )

                continue

            remaining.remove(found)

            drift = abs(
                found["confidence"] - item["confidence"]
            )

            if drift > CONFIDENCE_TOLERANCE:

                warnings.append(
                    f"{name}: {item['category']} confiança "
                    f"{item['confidence']:.3f} -> "
                    f"{found['confidence']:.3f}"
                )

        for extra in remaining:

            failures.append(
                f"{name}: EXTRA {describe(extra)}"
            )

    return failures, warnings


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--update",
        action="store_true",
        help="regrava o arquivo de esperados",
    )

    parser.add_argument(
        "--bench",
        action="store_true",
        help="mede o tempo por passada",
    )

    args = parser.parse_args()

    log.setup("WARNING")

    detector = Detector()

    print(
        f"Templates: {len(detector.templates)} | "
        f"escala grossa: {detector.coarse_scale}"
    )

    if args.bench:

        images = [
            cv2.imread(str(p))
            for p in sorted(IMAGES_DIR.glob("*.png"))
        ]

        for image in images:
            detector.detect(image)

        started = time.perf_counter()

        rounds = 3

        for _ in range(rounds):

            for image in images:

                detector.detect(image)

        elapsed = (
            (time.perf_counter() - started)
            / (rounds * len(images))
        )

        print(
            f"\nPassada completa: {elapsed * 1000:.0f} ms "
            f"({1 / elapsed:.1f} FPS)"
        )

        return 0

    actual = detect_all(detector)

    if args.update:

        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)

        GOLDEN_PATH.write_text(
            json.dumps(actual, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        total = sum(len(v) for v in actual.values())

        print(
            f"\nGravado {GOLDEN_PATH.name}: "
            f"{len(actual)} telas, {total} detecções."
        )

        print("CONFIRA o diff antes de comitar.")

        return 0

    if not GOLDEN_PATH.exists():

        print(
            f"\n{GOLDEN_PATH} não existe. "
            f"Rode com --update."
        )

        return 1

    expected = json.loads(
        GOLDEN_PATH.read_text(encoding="utf-8")
    )

    failures, warnings = compare(expected, actual)

    print()

    for name in sorted(actual):

        print(f"  {name}: {len(actual[name])} detecções")

    for warning in warnings:

        print(f"\n  AVISO  {warning}")

    if failures:

        print()

        for failure in failures:

            print(f"  FALHA  {failure}")

        print(f"\n{len(failures)} regressão(ões).")

        return 1

    print("\nOK — nenhuma regressão.")

    return 0


if __name__ == "__main__":

    sys.exit(main())
