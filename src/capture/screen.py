"""ScreenCapture: scrcpy-server H.264 stream, versioning, FPS cap."""

import random
import socket
import subprocess
import threading
import time
import os

import av
import cv2

from core import log
from core.metrics import RateMeter
from core.config import (
    CAPTURE_CONNECT_TIMEOUT,
    CAPTURE_MAX_FPS,
    CAPTURE_START_TIMEOUT,
    SCRCPY_DEVICE_JAR,
    SCRCPY_PORT,
    SCRCPY_SERVER_PATH,
    SCRCPY_SERVER_VERSION,
    ADB_PATH,
)

logger = log.get("capture")


class ScreenCapture:

    SERVER_VERSION = SCRCPY_SERVER_VERSION
    SERVER_PATH = SCRCPY_SERVER_PATH

    # Path on the device, separate from what scrcpy.exe uses.
    DEVICE_JAR = SCRCPY_DEVICE_JAR

    PORT = SCRCPY_PORT

    def __init__(self, serial=None):

        self.serial = serial  # Device; None = let adb decide (requires 1 device)

        self.server_process = None
        self.socket = None
        self.codec = None

        self.scid = 0  # Server instance ID (scrcpy_<id> in scrcpy 4.1+)
        self.socket_name = "scrcpy"

        self.first_chunk = None  # First chunk of the stream (real H.264, not disposable)

        self.running = False
        self.capture_thread = None

        self.latest_frame = None
        self.latest_frame_time = 0.0

        self.version = 0  # Increments on each frame (lets callers detect novelty)

        # Frames per second the device is delivering.
        self.rate = RateMeter()

        # Minimum interval between published frames (FPS capping)
        self._min_interval = (
            1.0 / CAPTURE_MAX_FPS
            if CAPTURE_MAX_FPS
            else 0.0
        )
        self._published_at = 0.0
        self.dropped = 0  # Frames dropped by the cap

        self.condition = threading.Condition()

    def start(self):

        # A fresh scid per run: the server's abstract
        # socket is "scrcpy_<scid>", and our own id ensures
        # we do not collide with the mirror or leftovers from
        # previous runs.
        self.scid = random.randint(0, 0x7FFFFFFF)
        self.socket_name = f"scrcpy_{self.scid:08x}"

        logger.info("Starting scrcpy-server (socket @%s)...", self.socket_name)

        self._cleanup_forward()
        self._push_server()
        self._setup_forward()
        self._start_server()

        logger.info("Connecting to H.264 stream...")
        self.socket, self.first_chunk = self._connect(CAPTURE_CONNECT_TIMEOUT)

        if self.socket is None:

            raise RuntimeError(
                "\n".join(
                    [
                        "Could not read the stream in "
                        f"{CAPTURE_CONNECT_TIMEOUT:.0f}s "
                        f"(port {self.PORT}, socket "
                        f"@{self.socket_name}).",

                        "Likely causes:",

                        "  - the device screen is off "
                        "(the encoder captures the display)",

                        "  - SCRCPY_SERVER_VERSION does not match "
                        "the installed scrcpy-server",

                        "  - server did not start: run "
                        + " ".join(self._adb("shell", "cat"))
                        + " /proc/net/unix | grep scrcpy "
                        "and check whether the socket shows up",
                    ]
                )
            )

        logger.info("Stream open (%d initial bytes).", len(self.first_chunk))

        self.codec = av.CodecContext.create("h264", "r")

        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="capture",
            daemon=True,
        )
        self.capture_thread.start()
        logger.info("Capture thread started.")

        if not self._wait_first_frame(CAPTURE_START_TIMEOUT):

            raise RuntimeError(
                "\n".join(
                    [
                        f"No frames in "
                        f"{CAPTURE_START_TIMEOUT:.0f}s on "
                        f"port {self.PORT}.",

                        "Likely causes:",

                        "  - another process using the port "
                        "(the scrcpy mirror uses "
                        "27183-27199 by default)",

                        "  - the device screen is off",

                        "  - SCRCPY_SERVER_VERSION does not match "
                        "the installed scrcpy-server",
                    ]
                )
            )

    def _connect(self, timeout):
        """Connects and waits for the first valid chunk (0 bytes = socket does not exist yet)."""

        deadline = time.monotonic() + timeout

        tentativas = 0

        while time.monotonic() < deadline:

            tentativas += 1

            candidato = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            )

            candidato.settimeout(1.0)

            try:
                candidato.connect(("127.0.0.1", self.PORT))
            except OSError:
                self._fecha(candidato)
                time.sleep(0.15)
                continue

            # Distinction: 0 bytes = socket missing; timeout = waiting for valid data

            morto = False

            while time.monotonic() < deadline:

                try:

                    data = candidato.recv(65536)

                except socket.timeout:

                    continue

                except OSError:

                    morto = True

                    break

                if not data:

                    morto = True

                    break

                candidato.settimeout(2)

                logger.debug(
                    "Stream answered on attempt %d.",
                    tentativas,
                )

                return candidato, data

            if not morto:
                break

            self._fecha(candidato)

            time.sleep(0.15)

        logger.warning(
            "No stream after %d attempts.",
            tentativas,
        )

        return None, None

    @staticmethod
    def _fecha(sock):

        try:
            sock.close()

        except OSError:
            pass

    def _wait_first_frame(self, timeout):

        with self.condition:

            return self.condition.wait_for(
                lambda: (
                    self.latest_frame is not None
                    or not self.running
                ),
                timeout=timeout,
            ) and self.latest_frame is not None

    def _capture_loop(self):
        # first_chunk is real H.264, not disposable
        pendente = self.first_chunk

        self.first_chunk = None

        while self.running:

            if pendente is not None:

                data = pendente
                pendente = None

                self._decode(data)

                continue

            try:

                data = self.socket.recv(65536)

            except socket.timeout:

                continue

            except OSError as error:

                if self.running:

                    logger.warning(
                        "Socket dropped: %s",
                        error,
                    )

                break

            if not data:

                logger.warning(
                    "Stream closed on the other side (0 bytes). "
                    "Port %d busy with another process, or "
                    "server ended.",
                    self.PORT,
                )

                break

            if not self._decode(data):
                break

        # Wake up anyone waiting for a frame.
        with self.condition:

            self.running = False

            self.condition.notify_all()

    def _decode(self, data):
        """Returns False when the stream ended (EOF)."""

        try:

            for packet in self.codec.parse(data):

                for frame in self.codec.decode(packet):

                    self._publish(frame)

        except av.error.EOFError:

            logger.warning("H264 decoder got EOF.")

            return False

        except av.error.FFmpegError as error:

            logger.warning("H264 error: %s", error)

        return True

    def _publish(self, frame):
        # Capping BEFORE the expensive conversion (YUV->BGR). Decoding is not skipped (inter-frame)
        if self._min_interval:

            agora = time.monotonic()

            if (
                agora - self._published_at
                < self._min_interval
            ):

                self.dropped += 1

                return

            self._published_at = agora

        yuv = frame.to_ndarray()

        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        with self.condition:
            # Keep only the newest frame (dataset gets the original, detector normalizes)
            self.latest_frame = bgr
            self.latest_frame_time = time.monotonic()

            self.version += 1

            self.condition.notify_all()

        self.rate.tick()

    def get_frame(self, since_version=None, timeout=1.0):
        """Returns (frame, version, timestamp). With since_version, blocks until a new frame. Array is shared."""

        with self.condition:

            if since_version is not None:

                self.condition.wait_for(
                    lambda: (
                        self.version != since_version
                        or not self.running
                    ),
                    timeout=timeout,
                )

            if self.latest_frame is None:

                return None, self.version, 0.0

            return (
                self.latest_frame,
                self.version,
                self.latest_frame_time,
            )

    def is_running(self):
        return self.running

    def get_fps(self):
        return self.rate.rate()

    def _adb(self, *args):
        """Builds the adb command for the chosen device (without -s, adb refuses with multiple devices)."""

        comando = [ADB_PATH]

        if self.serial:

            comando += ["-s", self.serial]

        return comando + list(args)

    def _push_server(self):

            # Check that the scrcpy-server path is configured and exists
            if not self.SERVER_PATH:
                raise RuntimeError(
                    "scrcpy-server path not configured (empty SCRCPY_SERVER_PATH). "
                    "Install scrcpy on PATH or place scrcpy-server*.jar in tools/scrcpy and try again."
                )

            if not os.path.exists(self.SERVER_PATH):
                raise RuntimeError(
                    f"scrcpy-server not found at '{self.SERVER_PATH}'. Check SCRCPY_SERVER_PATH."
                )

            result = subprocess.run(
                self._adb(
                    "push",
                    self.SERVER_PATH,
                    self.DEVICE_JAR,
                ),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:

                raise RuntimeError(
                    "Error sending scrcpy-server:\n"
                    + (result.stderr or "")
                )

    def _setup_forward(self):

        subprocess.run(
            self._adb(
                "forward",
                f"tcp:{self.PORT}",
                f"localabstract:{self.socket_name}",
            ),
            check=True,
        )

    def _start_server(self):

        command = self._adb(
            "shell",

            f"CLASSPATH={self.DEVICE_JAR}",

            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            self.SERVER_VERSION,
            f"scid={self.scid:08x}",  # Required: abstract socket is scrcpy_<id>

            "tunnel_forward=true",

            "audio=false",

            "control=false",

            "cleanup=false",

            "raw_stream=true",

            "video_codec=h264",
        )

        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        logger.info("Shutting down capture...")

        with self.condition:
            self.running = False
            self.condition.notify_all()

        if self.socket:

            try:
                self.socket.close()

            except OSError:
                pass
            self.socket = None

        if self.capture_thread:
            self.capture_thread.join(timeout=2)
            self.capture_thread = None

        self._cleanup_forward()

        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None

        subprocess.run(
            self._adb(
                "shell",
                "rm",
                self.DEVICE_JAR,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        logger.info("Capture stopped.")

    def _cleanup_forward(self):

        subprocess.run(
            self._adb(
                "forward",
                "--remove",
                f"tcp:{self.PORT}",
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
