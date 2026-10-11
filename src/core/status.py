"""
Fixed status block rewritten in place, without piling up lines.

Answers "what is happening RIGHT NOW" — the log tells the story
event by event, which forces scrolling the terminal to find the current line.
Panel and log fight over the same terminal, so main lowers the console log
to WARNING while the panel is on (turn off
STATUS_PANEL to go back to line-by-line logging).
"""

import os
import platform
import shutil
import sys
import time

from core.config import (
    BATTERY_POLL_INTERVAL,
    DETECTOR_DEBUG_MISSES,
)
from core.metrics import formata_duracao


def formata_bateria(leitura):
    """
    The BatteryMonitor reading (level, charging, age) as a short word.

    Flags STALE readings instead of hiding them — a frozen number looks
    current, and that is how an hour passes without noticing adb hung.
    """

    if not leitura:
        return "battery ?"

    nivel, carregando, idade = leitura

    if nivel is None:
        return "battery ?"

    texto = f"battery {nivel}%"

    if carregando:
        return texto + " charging"

    if idade and idade > BATTERY_POLL_INTERVAL * 2:
        texto += f" ({idade:.0f}s ago)"

    return texto


# Fixed on purpose: rewriting must know how many lines to move up, and
# counting what was printed skews the layout. The near-miss
# line is the only optional one, decided ONCE at construction — if it
# came and went, it would shift the block and `\033[nA` would move up wrong.
LINHAS_BASE = 5


class StatusPanel:
    """In-place rewritten status block; `update(...)` decides on its own when to redraw."""

    def __init__(
        self,
        device,
        interval=0.5,
        stream=None,
        misses_line=None,
    ):

        self.device = device or "?"

        # "No detection" only makes sense with DETECTOR_DEBUG_MISSES on
        # (without it the detector does not even compute the best match).
        self.misses_line = (
            DETECTOR_DEBUG_MISSES
            if misses_line is None
            else misses_line
        )

        self.linhas = LINHAS_BASE + (
            1 if self.misses_line else 0
        )

        self.interval = interval

        self.stream = stream or sys.stdout

        self.desenhado = False
        self.ultimo = 0.0

        # Last drawn key: no change, no redraw.
        # Each redraw competes with log lines landing in the
        # middle of the block and breaking in-place rewrite — the
        # scrollback becomes a graveyard of old blocks. The
        # 30s heartbeat keeps the times ("running", "ago")
        # reasonably fresh even when idle.
        self._chave = None

        self.iniciado = time.monotonic()

        # Without ANSI (file, pipe, CI log) the cursor sequence
        # would become garbage amid the text; then the panel prints once and stays quiet.
        self.ansi = self._suporta_ansi()

    def _suporta_ansi(self):

        if not hasattr(self.stream, "isatty"):
            return False

        try:
            if not self.stream.isatty():
                return False

        except Exception:
            return False

        if platform.system() != "Windows":
            return True

        # The legacy Windows console only interprets ANSI after
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING; without it, the panel
        # would show up as "←[2K" on screen.
        try:

            import ctypes

            kernel32 = ctypes.windll.kernel32

            # -11 = STD_OUTPUT_HANDLE, 0x0004 = VT processing
            handle = kernel32.GetStdHandle(-11)

            modo = ctypes.c_uint32()

            if not kernel32.GetConsoleMode(
                handle,
                ctypes.byref(modo),
            ):
                return False

            return bool(
                kernel32.SetConsoleMode(
                    handle,
                    modo.value | 0x0004,
                )
            )

        except Exception:
            return False

    def _largura(self):
        """Usable terminal width — a longer line WRAPS in two and misaligns the `\033[nA` count."""

        try:
            colunas = shutil.get_terminal_size().columns

        except Exception:
            return 100

        # -1: writing in the last column already wraps on some terminals.
        return max(20, colunas - 1)

    def _linhas(self, resumo, extra=None, misses=None):

        acao = resumo.get("acao") or "—"

        acao_ha = resumo.get("acao_ha")

        # "how long ago" is what tells a working bot apart
        # from a bot idled on an action that succeeded once.
        if acao_ha is not None and acao_ha > 1.0:
            acao = f"{acao}  ({acao_ha:.0f}s ago)"

        corrido = time.monotonic() - self.iniciado

        linhas = [
            f"Connected device: {self.device}",
            f"Flew: {resumo.get('voos', 0)} times",
            f"Renovated: {resumo.get('reformas', 0)} times",
            f"Action: {acao}",
        ]

        rodape = (
            f"state {resumo.get('estado', '?')}"
            f" | {resumo.get('acoes', 0)} actions"
            f" | {formata_duracao(corrido)} running"
        )

        if extra:
            rodape += f" | {extra}"

        linhas.append(rodape)

        # Only exists with debug on; always shown (even with nothing to
        # report) because the block has fixed height.
        if self.misses_line:

            linhas.append(
                f"No detection: {misses}"
                if misses
                else "No detection: —"
            )

        # Locks the size: a new line without adjusting the count would eat the
        # line above instead of failing visibly.
        assert len(linhas) == self.linhas, (
            len(linhas),
            self.linhas,
        )

        largura = self._largura()

        return [
            linha
            if len(linha) <= largura
            else linha[: largura - 1] + "…"
            for linha in linhas
        ]

    def update(self, resumo, extra=None, misses=None, force=False):
        """Redraws if the interval elapsed. `resumo` is StateMachine.summary(), `misses` is Detector.miss_report()."""

        agora = time.monotonic()

        if not force and agora - self.ultimo < self.interval:
            return

        # Action/state/counts + misses define the content;
        # "extra" (fps, battery) fluctuates on its own and does not justify
        # a redraw — see note in __init__.
        chave = (
            resumo.get("estado"),
            resumo.get("acao"),
            resumo.get("acoes"),
            resumo.get("voos"),
            resumo.get("reformas"),
            misses,
        )

        if (
            not force
            and chave == self._chave
            and agora - self.ultimo < 30
        ):
            return

        self.ultimo = agora
        self._chave = chave

        linhas = self._linhas(resumo, extra, misses)

        if not self.ansi:

            # Without ANSI, print ONCE and stop, instead of the block repeated a thousand times.
            if not self.desenhado:

                self.stream.write("\n".join(linhas) + "\n")
                self.stream.flush()

                self.desenhado = True

            return

        saida = []

        if self.desenhado:

            # Move up to the top of the previous block.
            saida.append(f"\033[{self.linhas}A")

        for linha in linhas:

            # \033[2K clears the whole line, otherwise the tail of the previous text stays on screen.
            saida.append(f"\r\033[2K{linha}\n")

        self.stream.write("".join(saida))
        self.stream.flush()

        self.desenhado = True

    def close(self):
        """Leaves the cursor below the block, so shutdown/traceback logs do not write over it."""

        if self.desenhado and self.ansi:

            self.stream.write("\n")
            self.stream.flush()

        self.desenhado = False
