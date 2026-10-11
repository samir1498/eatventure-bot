"""Camada mais baixa: comandos adb no device. Falha transitória vira WARNING e devolve False, nunca derruba o bot."""

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
        """(porcentagem, carregando) ou None. Timeout curto: é HUD, melhor envelhecer que travar a thread."""

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

            # A escala nem sempre é 100. Assumir que é daria
            # "level 17" virando 17% num device de escala 255.
            scale = int(campos.get("scale", 100)) or 100

        except (KeyError, ValueError):

            logger.warning("dumpsys battery without readable 'level'")

            return None

        # status 2 = charging, 5 = full. As flags de energia
        # cobrem o caso de estar plugado sem contar como
        # 'charging' (bateria cheia, ou carga pausada).
        carregando = campos.get("status") in ("2", "5") or any(
            campos.get(f"{fonte} powered") == "true"
            for fonte in ("AC", "USB", "Wireless", "Dock")
        )

        return round(level * 100.0 / scale), carregando

    def get_screen_size(self):
        """Resolução real do device ('wm size'), usada para converter coordenada de frame em toque quando o stream vem reduzido."""

        output = self._run("shell", "wm", "size")

        if not output:
            return None

        # Prefere "Override size" quando existe: é a
        # resolução em uso.
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
        """N toques em UM comando adb: N chamadas separadas pagavam spawn+JVM cada, passando de 1s com 5 toques."""

        vezes = max(1, int(times))

        if vezes == 1:
            return self.click(x, y)

        um = f"input tap {int(x)} {int(y)}"

        return self._run(
            "shell",
            "; ".join([um] * vezes),

            # O adb tem de sobreviver aos N toques.
            timeout=(
                timeout
                if timeout is not None
                else 5.0 + 0.5 * vezes
            ),
        ) is not None

    def swipe_many(self, x1, y1, x2, y2, times, duration=300):
        """N swipes iguais em UM comando adb (mesmo motivo do tap_many)."""

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

            # O adb precisa sobreviver ao press inteiro.
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
        """Pacote em primeiro plano (mFocusedApp), ou None se indeterminado. None = adb falhou: quem chama não deve presumir jogo fechado."""

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
        """Traz o pacote para a frente via monkey (1 evento)."""

        return self._run(
            "shell",
            "monkey",
            "-p", package,
            "-c", "android.intent.category.LAUNCHER",
            "1",
            timeout=timeout,
        ) is not None

    def collapse_shade(self, timeout=5.0):
        """Recolhe a gaveta de notificações se estiver aberta (inoffensivo se não estiver)."""

        return self._run(
            "shell",
            "cmd",
            "statusbar",
            "collapse",
            timeout=timeout,
        ) is not None
