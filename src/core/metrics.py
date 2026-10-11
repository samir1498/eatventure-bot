"""Rate and duration meters, shared between capture and detector to avoid double counting."""

import threading
import time
from collections import deque


class RateMeter:
    """Events per second over a sliding window — average-since-start would hide when the stream stalled."""

    def __init__(self, window=2.0):

        self.window = window

        self.timestamps = deque()

        self.lock = threading.Lock()

    def tick(self):

        now = time.monotonic()

        with self.lock:

            self.timestamps.append(now)

            self._trim(now)

    def rate(self):

        now = time.monotonic()

        with self.lock:

            self._trim(now)

            if len(self.timestamps) < 2:
                return 0.0

            elapsed = self.timestamps[-1] - self.timestamps[0]

            if elapsed <= 0:
                return 0.0

            # n events delimit n-1 intervals.
            return (len(self.timestamps) - 1) / elapsed

    def _trim(self, now):

        limit = now - self.window

        while self.timestamps and self.timestamps[0] < limit:

            self.timestamps.popleft()


class DurationMeter:
    """Exponential moving average of duration — cheap (one float), no history."""

    def __init__(self, smoothing=0.2):

        self.smoothing = smoothing

        self.value = 0.0

        self.lock = threading.Lock()

    def add(self, duration):

        with self.lock:

            if self.value <= 0.0:

                self.value = duration

            else:

                self.value = (
                    self.smoothing * duration
                    + (1.0 - self.smoothing) * self.value
                )

    def average(self):

        with self.lock:
            return self.value


def formata_duracao(segundos):
    """Glanceable duration: "42s", "3m07s", "1h04m". Whole seconds, never decimal — corner-of-the-eye reading."""

    if segundos is None:
        return "--"

    total = int(max(0, segundos))

    if total < 60:
        return f"{total}s"

    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"

    return f"{total // 3600}h{(total % 3600) // 60:02d}m"
