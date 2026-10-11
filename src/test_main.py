"""Visualizador do stream cru, sem detecção (python src/test_main.py) — para conferir a captura antes de culpar o detector."""

import argparse
import sys

import cv2

from capture.screen import ScreenCapture
from core import devices, log
from core.config import (
    DEVICE_SERIAL,
    LOG_LEVEL,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

WINDOW_NAME = "AI Raw"


def main(argv=None):

    log.setup(LOG_LEVEL)

    logger = log.get("raw")

    parser = argparse.ArgumentParser(
        description="Visualizador do stream cru",
    )

    parser.add_argument(
        "--device",
        metavar="SERIAL",
        help="serial do device (adb devices)",
    )

    args = parser.parse_args(argv)

    # Mesma escolha do main.py: sem o -s, com dois devices o adb recusa toda chamada.
    capture = ScreenCapture(
        devices.resolver(args.device or DEVICE_SERIAL)
    )

    try:

        capture.start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(
            WINDOW_NAME,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )

        last_version = 0

        while True:

            frame, version, _ = capture.get_frame(
                since_version=last_version,
                timeout=1.0,
            )

            if not capture.is_running():

                logger.warning("Stream ended.")

                break

            if frame is None:
                continue

            last_version = version

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:

        logger.info("Interrupted by user.")

    finally:

        capture.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    sys.exit(main())
