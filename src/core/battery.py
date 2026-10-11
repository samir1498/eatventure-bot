"""Battery reads off the critical path: `dumpsys battery` costs ~56ms (>3x one detector pass), so it runs on its own thread and the loop only reads the latest in-memory value."""

import threading
import time

from core import log
from core.config import BATTERY_POLL_INTERVAL

logger = log.get("battery")


class BatteryMonitor:

    def __init__(self, android, interval=BATTERY_POLL_INTERVAL):

        self.android = android
        self.interval = interval

        self.running = False
        self.thread = None

        # None = not read yet, or adb never succeeded.
        self.level = None
        self.charging = False

        # Reading timestamp, so the HUD can flag stale values.
        self.updated = 0.0

        self.lock = threading.Lock()

        # Wakes the thread on stop, without waiting the whole interval.
        self.wake = threading.Event()

    def start(self):

        if self.running:
            return

        self.running = True

        self.wake.clear()

        self.thread = threading.Thread(
            target=self._run,
            name="battery",
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

        # Without this, stopping the bot would wait up to BATTERY_POLL_INTERVAL.
        self.wake.set()

        if self.thread:

            self.thread.join(timeout=2.0)

            self.thread = None

    def get(self):
        """(percentage, charging, age_in_seconds); percentage is None with no reading yet."""

        with self.lock:

            if self.level is None:
                return None, False, 0.0

            return (
                self.level,
                self.charging,
                time.monotonic() - self.updated,
            )

    def _run(self):

        while self.running:

            leitura = self.android.battery()

            if leitura is not None:

                nivel, carregando = leitura

                with self.lock:

                    anterior = self.level

                    self.level = nivel
                    self.charging = carregando
                    self.updated = time.monotonic()

                if anterior != nivel:

                    logger.info(
                        "Battery: %d%%%s",
                        nivel,
                        " (charging)" if carregando else "",
                    )

            # Interruptible wait.
            self.wake.wait(self.interval)
