import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import selector_layout as layout                  # noqa: E402
from capture.screen import ScreenCapture          # noqa: E402
from core import devices, log                     # noqa: E402
from vision.detector import Detector               # noqa: E402
from core.config import (                         # noqa: E402
    DEVICE_SERIAL,
    LOG_LEVEL,
    SELECTOR_FALLBACK_HEIGHT,
    SELECTOR_FALLBACK_WIDTH,
    SELECTOR_HEIGHT_FRACTION,
    SELECTOR_MAX_HEIGHT,
    SELECTOR_MAX_WIDTH,
    SELECTOR_WIDTH_FRACTION,
)


WINDOW_NAME = "Detect Preview"

COR_DEFAULT = (0, 165, 255)
COR_OVERRIDE = (0, 255, 0)


class DetectPreview:

    def __init__(self, serial=None):
        # Same source as the bot: preview over screencap showed a
        # confidence the detector never sees on the real H.264 frame.
        self.capture = ScreenCapture(serial=serial)
        self.detector = Detector()
        self.image = None
        self.display = None
        self.scale = 1.0

    def capture_screen(self):

        print()
        print("[SCREEN] Capturando nova tela...")

        if not self.capture.is_running():
            self.capture.start()

        frame, _, _ = self.capture.get_frame()

        if frame is None:

            raise RuntimeError(
                "Não veio frame do stream."
            )

        self.image = frame.copy()

        height, width = self.image.shape[:2]

        print(f"[SCREEN] Resolução: {width} x {height}")

        self._detect_and_draw()

    def _janela(self):

        largura = SELECTOR_MAX_WIDTH
        altura = SELECTOR_MAX_HEIGHT

        if largura and altura:
            return largura, altura

        tela = layout.tela_disponivel(
            SELECTOR_HEIGHT_FRACTION,
            SELECTOR_WIDTH_FRACTION,
        )

        if tela is None:

            tela = (
                SELECTOR_FALLBACK_WIDTH,
                SELECTOR_FALLBACK_HEIGHT,
            )

        return (
            largura or tela[0],
            altura or tela[1],
        )

    def _label(self, detection):
        """Common category (box, close, ...) uses the category name; food uses the
        item name. Without an override at the current resolution, prefixes with 'default_'."""

        if detection["category"] == "food":
            # Base name: the default's resolution prefix is provenance,
            # not part of the item name.
            base = Detector._base_name(
                detection["name"]
            ).removesuffix(".png")
        else:
            base = detection["category"]

        if detection["origin"] == "override":
            return base

        return f"default_{base}"

    def _detect_and_draw(self):

        height, width = self.image.shape[:2]

        detections = self.detector.detect(self.image)

        max_width, max_height = self._janela()

        self.scale = layout.escala(
            width,
            height,
            max_width,
            max_height,
        )

        self.display = (
            self.image.copy()
            if self.scale == 1.0
            else cv2.resize(
                self.image,
                layout.tamanho_canvas(width, height, self.scale),
                interpolation=cv2.INTER_AREA,
            )
        )

        print(f"[DETECT] {len(detections)} detecção(ões)")

        for detection in detections:

            rotulo = self._label(detection)

            print(
                f"  {rotulo}  F:{detection['confidence']:.2f}  "
                f"({detection['x']},{detection['y']} "
                f"{detection['width']}x{detection['height']})"
            )

            self._draw(detection, rotulo)

    def _draw(self, detection, rotulo):

        x = int(detection["x"] * self.scale)
        y = int(detection["y"] * self.scale)
        w = int(detection["width"] * self.scale)
        h = int(detection["height"] * self.scale)

        cor = (
            COR_OVERRIDE
            if detection["origin"] == "override"
            else COR_DEFAULT
        )

        cv2.rectangle(
            self.display,
            (x, y),
            (x + w, y + h),
            cor,
            2,
        )

        posicao = (x, max(y - 8, 14))

        cv2.putText(
            self.display,
            rotulo,
            posicao,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            self.display,
            rotulo,
            posicao,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cor,
            1,
            cv2.LINE_AA,
        )

    def run(self):

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        print()
        print("===================================")
        print("        DETECT PREVIEW")
        print("===================================")
        print()
        print("Laranja = via default | Verde = via override da resolução")
        print()
        print("R/F5 = nova captura   ESC = sair")
        print()

        while True:

            cv2.imshow(WINDOW_NAME, self.display)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == 116 or key == ord("r"):
                self.capture_screen()

        cv2.destroyWindow(WINDOW_NAME)

        self.capture.stop()


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Mostra tudo que o detector acha na tela do device.",
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

    args = parser.parse_args(argv)

    log.setup(LOG_LEVEL)

    try:

        serial = devices.resolver(args.device or DEVICE_SERIAL)

    except KeyboardInterrupt:

        print("Cancelado.")

        return 1

    preview = DetectPreview(serial)

    preview.capture_screen()
    preview.run()

    return 0


if __name__ == "__main__":

    sys.exit(main())
