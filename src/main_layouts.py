"""EatVenture AI — layout updater (python src/main_layouts.py).

Same window/vision as main.py; the only difference is what runs behind it:
no StateMachine or actions here, only capture + Detector with defaults
enabled in every category (categories=None), which triggers the
already existing override learning in vision/detector.py
(_queue_learn_candidate / _learn_worker) — at no cost to the bot's
main loop when it runs with `main.py --layout-only`.

The other process reloads the overrides on its own — see
Detector._maybe_reload_templates / TEMPLATES_WATCH_INTERVAL. No
communication between the two besides the disk.
"""

import argparse
import subprocess
import sys
import time

import cv2

from capture.screen import ScreenCapture
from core import devices, log
from core.config import (
    AI_WINDOW_NAME,
    AI_WINDOW_POSITION,
    DEVICE_SERIAL,
    LOG_LEVEL,
    SCRCPY_EXTRA_ARGS,
    SCRCPY_PATH,
    SHOW_AI_VISION,
    SHOW_SCRCPY,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from core.metrics import DurationMeter, RateMeter

from vision.detector import Detector

logger = log.get("main_layouts")

# Only used when SHOW_AI_VISION is off — without a window, it is the only
# way to see progress.
REPORT_INTERVAL = 10.0


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="EatVenture AI — atualizador de layouts",
    )

    parser.add_argument(
        "--device",
        metavar="SERIAL",
        help=(
            "serial do device (adb devices). Sem isto, usa "
            "DEVICE_SERIAL do config; sem os dois, pergunta "
            "quando houver mais de um conectado."
        ),
    )

    return parser.parse_args(argv)


def start_scrcpy(device_id):

    window_title = f"EatVenture layouts ({device_id})"

    logger.info("Starting %s...", window_title)

    return subprocess.Popen(
        [
            SCRCPY_PATH,

            "--serial",
            device_id,

            "--window-title",
            window_title,
        ]
        + list(SCRCPY_EXTRA_ARGS)
    )


def setup_window():

    cv2.namedWindow(AI_WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(
        AI_WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    cv2.moveWindow(AI_WINDOW_NAME, *AI_WINDOW_POSITION)


def run_loop(capture, detector):

    last_version = 0
    reported_at = 0.0

    detect_rate = RateMeter()
    detect_duration = DurationMeter()

    while True:

        # Short timeout so the OpenCV window stays responsive
        # (and ESC works) even if the stream stalls.
        frame, version, _timestamp = capture.get_frame(
            since_version=last_version,
            timeout=0.2,
        )

        if not capture.is_running():

            logger.warning("Stream ended.")

            break

        if frame is None:
            continue

        last_version = version

        started = time.monotonic()

        # categories=None: goes through ALL categories, not just the
        # game's current state's — that is what gives defaults broad coverage.
        detections = detector.detect(frame)

        detect_duration.add(time.monotonic() - started)
        detect_rate.tick()

        if not SHOW_AI_VISION:

            agora = time.monotonic()

            if agora - reported_at >= REPORT_INTERVAL:

                reported_at = agora

                cobertos, total = detector.template_coverage() or (0, 0)

                logger.info(
                    "current resolution coverage: %d/%d defaults with override",
                    cobertos,
                    total,
                )

            continue

        altura_frame, largura_frame = frame.shape[:2]

        escala_janela = min(
            WINDOW_WIDTH / largura_frame,
            WINDOW_HEIGHT / altura_frame,
            1.0,
        )

        if escala_janela < 1.0:

            ai_frame = cv2.resize(
                frame,
                (
                    int(round(largura_frame * escala_janela)),
                    int(round(altura_frame * escala_janela)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:

            ai_frame = frame.copy()

        ai_frame = detector.draw(
            ai_frame,
            detections,
            {
                "capture_fps": capture.get_fps(),
                "detect_fps": detect_rate.rate(),
                "detect_ms": detect_duration.average() * 1000,
                "detections": len(detections),

                # Coverage of the resolution folder over the default —
                # the number that says whether this process is still worth running.
                "coverage": detector.template_coverage(),
            },
            scale=escala_janela,
        )

        cv2.imshow(AI_WINDOW_NAME, ai_frame)

        # ESC only arrives here; this window receives keys.
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main(argv=None):

    log.setup(LOG_LEVEL)

    args = parse_args(argv)

    try:

        device_id = devices.resolver(args.device or DEVICE_SERIAL)

    except KeyboardInterrupt:

        logger.info("Cancelled.")

        return 1

    # The scrcpy mirror is optional: this process's capture does not go through it.
    scrcpy_process = (
        start_scrcpy(device_id)
        if SHOW_SCRCPY
        else None
    )

    capture = ScreenCapture(device_id)

    detector = Detector()

    logger.info(
        "Layout updater running for %s — quit with %s",
        device_id,
        "ESC or Ctrl+C" if SHOW_AI_VISION else "Ctrl+C",
    )

    try:

        capture.start()

        if SHOW_AI_VISION:
            setup_window()

        run_loop(capture, detector)

    except KeyboardInterrupt:

        logger.info("Interrupted by user.")

    finally:

        logger.info("Shutting down layout updater...")

        capture.stop()

        cv2.destroyAllWindows()

        if scrcpy_process:

            scrcpy_process.terminate()

            try:

                scrcpy_process.wait(timeout=2)

            except subprocess.TimeoutExpired:

                scrcpy_process.kill()

    return 0


if __name__ == "__main__":

    sys.exit(main())
