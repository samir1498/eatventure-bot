"""Worker thread: async detector with category filter, no letterbox, blocking wait."""

import threading
import time

from core import log
from core.config import VISION_INTERVAL
from core.metrics import DurationMeter, RateMeter

logger = log.get("vision")


class VisionWorker:

    def __init__(self, detector, interval=VISION_INTERVAL):

        self.detector = detector

        self.interval = interval

        self.running = False
        self.thread = None

        # Initialized to avoid AttributeError on the first _run loop
        self.latest_raw_frame = None
        self.latest_frame_time = 0.0

        self.detections = []
        self.detections_time = 0.0

        # Frame that produced the detections above. Not
        # latest_frame: that one was already replaced.
        self.detections_frame = None

        self.categories = None

        # New frame wakes the worker (replaces busy-wait)
        self.frame_lock = threading.Condition()

        self.detection_lock = threading.Lock()

        # Last thread failure, so the main loop and the
        # HUD can tell WHY there is no detection.
        self.last_error = None
        self._error_kinds = set()

        # Frames older than this timestamp are rejected (post-action settling)
        self.frame_floor = 0.0

        # Frames dropped for being too old to
        # decide on. Lets the HUD explain a frozen overlay.
        self.skipped_stale = 0

        # Waiting for the screen to settle after an action?
        self.waiting_settle = False

        # Actual cost of one pass, and how many per second.
        #
        # It's the number that tells whether the bot sees fast
        # enough — not the capture FPS.
        self.last_duration = 0.0

        self.duration = DurationMeter()

        self.rate = RateMeter()

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            name="vision-worker",
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread:

            self.thread.join(timeout=5)

            self.thread = None

    def set_frame(self, frame, timestamp=None):
        """Raw unprocessed frame. Detector rescales templates per resolution."""

        with self.frame_lock:

            self.latest_raw_frame = frame

            self.latest_frame_time = (
                time.monotonic() if timestamp is None else timestamp
            )

            self.frame_lock.notify()

    def set_frame_floor(self, timestamp):
        """Rejects frames older than this timestamp (post-action settling)."""

        with self.frame_lock:

            self.frame_floor = timestamp or 0.0

    def set_categories(self, categories):
        """Restricts search by category. Tuple preserves priority (order matters)."""

        with self.detection_lock:

            self.categories = (
                None
                if categories is None
                else tuple(categories)
            )

    def get_input(self):
        """Returns consistent (frame, detections, timestamp) for dataset."""

        with self.detection_lock:

            return (
                self.detections_frame,
                list(self.detections),
                self.detections_time,
            )

    def get_detections(self):
        """Returns (detections, age_in_seconds). Age is of the frame, not the query."""

        with self.detection_lock:

            detections = list(self.detections)

            frame_time = self.detections_time

        if not frame_time:
            return detections, 0.0

        return (
            detections,
            max(0.0, time.monotonic() - frame_time),
        )

    def get_fps(self):
        return self.rate.rate()

    def get_duration(self):
        return self.duration.average()

    def is_alive(self):
        """Is the vision thread alive? (silent worker death detection)."""

        return bool(
            self.running
            and self.thread is not None
            and self.thread.is_alive()
        )

    def _fail(self, error):
        """Logs each error type ONCE (avoids spam in a fast loop)."""

        self.last_error = f"{type(error).__name__}: {error}"

        if type(error).__name__ in self._error_kinds:
            return

        self._error_kinds.add(type(error).__name__)

        logger.exception(
            "Detector pass failed — %s. Thread "
            "continues; this message does not repeat.",
            self.last_error,
        )

    def _run(self):
        """Guarded worker: errors are caught to avoid silent death."""

        while self.running:

            try:

                self._pass()

            except Exception as error:

                self._fail(error)

                # Without this, an immediate error would become a
                # hot loop consuming a whole CPU.
                time.sleep(0.1)

    def _pass(self):
        """One pass: waits for a new frame, detects, publishes."""
        # Drops late frames (only the newest matters)
        with self.frame_lock:

            if self.latest_raw_frame is None:

                # Sleeps until a frame arrives. The timeout only lets
                # self.running be re-checked.
                self.frame_lock.wait(0.2)

            raw_frame = self.latest_raw_frame
            frame_time = self.latest_frame_time

            # Clears to signal it was consumed.
            self.latest_raw_frame = None

            floor = self.frame_floor

        if raw_frame is None:
            return

        # Skips too-old frames (shows the screen BEFORE the previous action's effect)
        if floor and frame_time and frame_time <= floor:

            self.skipped_stale += 1

            self.waiting_settle = True

            return

        self.waiting_settle = False

        with self.detection_lock:
            categories = (
                None if self.categories is None else self.categories
            )

        started = time.monotonic()

        detections = self.detector.detect(
            raw_frame,
            categories,
        )

        self.last_duration = time.monotonic() - started

        self.duration.add(self.last_duration)

        self.rate.tick()

        with self.detection_lock:

            self.detections = detections
            self.detections_time = frame_time

            self.detections_frame = raw_frame

        if self.interval:
            time.sleep(self.interval)
