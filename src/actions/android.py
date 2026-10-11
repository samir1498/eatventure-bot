"""Lowest layer: adb commands on the device. Transient failure becomes WARNING and returns False, never crashes the bot."""

import re
import subprocess

from core import log, config

logger = log.get("android")
ADB = config.ADB_PATH


class AndroidActions:

    def __init__(self, serial=None):

        self.serial = serial

    def _base(self):

        command = [ADB]

        if self.serial:

            command += ["-s", self.serial]

        return command

    def _run(self, *args, timeout=10.0):
        command = self._base() + list(args)

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        except subprocess.TimeoutExpired:

            logger.warning(
                "adb hung (timeout %.1fs): %s",
                timeout,
                " ".join(args),
            )

            return None

        except OSError as error:

            logger.warning(
                "adb unavailable: %s",
                error,
            )

            return None

        if result.returncode != 0:

            logger.warning(
                "adb failed (%d): %s | %s",
                result.returncode,
                " ".join(args),
                (result.stderr or "").strip(),
            )

            return None

        return result.stdout

    def battery(self, timeout=5.0):
        """(percentage, charging) or None. Short timeout: it is HUD, better to go stale than block the thread."""

        output = self._run(
            "shell",
            "dumpsys",
            "battery",
            timeout=timeout,
        )

        if not output:
            return None

        campos = {}

        for linha in output.splitlines():

            if ":" not in linha:
                continue

            chave, _, valor = linha.partition(":")

            campos[chave.strip()] = valor.strip()

        try:

            level = int(campos["level"])

            # Scale is not always 100. Assuming so would turn
            # "level 17" into 17% on a scale-255 device.
            scale = int(campos.get("scale", 100)) or 100

        except (KeyError, ValueError):

            logger.warning("dumpsys battery without readable 'level'")

            return None

        # status 2 = charging, 5 = full. The power flags
        # cover being plugged in without counting as
        # 'charging' (full battery, or paused charging).
        carregando = campos.get("status") in ("2", "5") or any(
            campos.get(f"{fonte} powered") == "true"
            for fonte in ("AC", "USB", "Wireless", "Dock")
        )

        return round(level * 100.0 / scale), carregando

    def get_screen_size(self):
        """Real device resolution ('wm size'), used to convert frame coordinates into taps when the stream comes scaled down."""

        output = self._run("shell", "wm", "size")

        if not output:
            return None

        # Prefer "Override size" when present: it is the
        # resolution in use.
        match = None

        for line in output.splitlines():

            found = re.search(
                r"(\d+)x(\d+)",
                line,
            )

            if not found:
                continue

            match = found

            if "Override" in line:
                break

        if not match:
            return None

        return (
            int(match.group(1)),
            int(match.group(2)),
        )

    def click(self, x, y):

        return self._run(
            "shell",
            "input",
            "tap",
            str(int(x)),
            str(int(y)),
        ) is not None

    def tap_many(self, x, y, times, timeout=None):
        """N taps in ONE adb command: N separate calls each paid spawn+JVM, exceeding 1s with 5 taps."""

        vezes = max(1, int(times))

        if vezes == 1:
            return self.click(x, y)

        um = f"input tap {int(x)} {int(y)}"

        return self._run(
            "shell",
            "; ".join([um] * vezes),

            # adb must outlive the N taps.
            timeout=(
                timeout
                if timeout is not None
                else 5.0 + 0.5 * vezes
            ),
        ) is not None

    def swipe_many(self, x1, y1, x2, y2, times, duration=300):
        """N identical swipes in ONE adb command (same reason as tap_many)."""

        vezes = max(1, int(times))

        um = (
            f"input swipe {int(x1)} {int(y1)} "
            f"{int(x2)} {int(y2)} {int(duration)}"
        )

        return self._run(
            "shell",
            "; ".join([um] * vezes),

            timeout=(duration / 1000.0 + 2.0) * vezes + 5.0,
        ) is not None

    def press(self, x, y, duration=4.0):

        duration_ms = int(duration * 1000)

        logger.debug(
            "long press (%d, %d) -> %.1fs",
            x,
            y,
            duration,
        )

        return self._run(
            "shell",
            "input",
            "swipe",
            str(int(x)),
            str(int(y)),
            str(int(x)),
            str(int(y)),
            str(duration_ms),

            # adb must outlive the whole press.
            timeout=duration + 5.0,
        ) is not None

    def swipe(self, x1, y1, x2, y2, duration=300):

        return self._run(
            "shell",
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(int(duration)),

            timeout=duration / 1000.0 + 5.0,
        ) is not None

    def back(self):

        return self._run(
            "shell",
            "input",
            "keyevent",
            "4",
        ) is not None

    def focused_package(self, timeout=5.0):
        """Foreground package (mFocusedApp), or None if undetermined. None = adb failed: callers must not assume the game is closed."""

        output = self._run(
            "shell",
            "dumpsys",
            "activity",
            "activities",
            timeout=timeout,
        )

        if not output:
            return None

        match = re.search(
            r"mFocusedApp=\S+\s+u\d+\s+([A-Za-z0-9_.]+)/",
            output,
        )

        return match.group(1) if match else None

    def launch_package(self, package, timeout=10.0):
        """Brings the package to the foreground via monkey (1 event)."""

        return self._run(
            "shell",
            "monkey",
            "-p", package,
            "-c", "android.intent.category.LAUNCHER",
            "1",
            timeout=timeout,
        ) is not None

    def collapse_shade(self, timeout=5.0):
        """Collapses the notification shade if open (harmless if not)."""

        return self._run(
            "shell",
            "cmd",
            "statusbar",
            "collapse",
            timeout=timeout,
        ) is not None
