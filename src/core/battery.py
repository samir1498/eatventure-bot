"""Leitura da bateria fora do caminho crítico: `dumpsys battery` custa ~56ms (>3x uma passada do detector), então roda em thread própria e o loop só lê o último valor em memória."""

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

        # None = ainda não leu, ou o adb nunca deu certo.
        self.level = None
        self.charging = False

        # Timestamp da leitura, para o HUD marcar valor velho.
        self.updated = 0.0

        self.lock = threading.Lock()

        # Acorda a thread ao parar, sem esperar o intervalo inteiro.
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

        # Sem isto, encerrar o bot esperaria até BATTERY_POLL_INTERVAL.
        self.wake.set()

        if self.thread:

            self.thread.join(timeout=2.0)

            self.thread = None

    def get(self):
        """(porcentagem, carregando, idade_em_segundos); porcentagem é None sem leitura ainda."""

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

            # Espera interrompível.
            self.wake.wait(self.interval)
