"""
Painel de status: bloco fixo reescrito no lugar, sem acumular linhas.

Responde "o que está acontecendo AGORA" — o log conta a história
evento a evento, o que exige rolar o terminal para achar a linha atual.
Painel e log disputam o mesmo terminal, por isso o main baixa o log do
console para WARNING enquanto o painel está ligado (desligue
STATUS_PANEL para voltar ao log linha-por-linha).
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
    A leitura do BatteryMonitor (nível, carregando, idade) em palavra curta.

    Marca a leitura VELHA em vez de esconder — um número parado parece
    atual, e é assim que se passa uma hora sem perceber que o adb travou.
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


# Fixo de propósito: a reescrita precisa saber quantas linhas subir, e
# contar o que foi impresso deixa o desenho torto. A linha de
# quase-acerto é a única opcional, decidida UMA vez na construção — se
# entrasse e saísse, deslocaria o bloco e o `\033[nA` subiria errado.
LINHAS_BASE = 5


class StatusPanel:
    """Bloco de status reescrito no lugar; `update(...)` decide sozinho quando redesenhar."""

    def __init__(
        self,
        device,
        interval=0.5,
        stream=None,
        misses_line=None,
    ):

        self.device = device or "?"

        # "Sem deteccao" só faz sentido com DETECTOR_DEBUG_MISSES ligado
        # (sem ele o detector nem calcula o melhor match).
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

        self.iniciado = time.monotonic()

        # Sem ANSI (arquivo, pipe, log de CI) a sequência de cursor
        # viraria lixo no meio do texto; aí o painel imprime uma vez e se cala.
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

        # O console legado do Windows só interpreta ANSI depois de
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING; sem ligar, o painel
        # apareceria como "←[2K" na tela.
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
        """Largura útil do terminal — linha mais comprida QUEBRA em duas e desalinha a contagem de `\033[nA`."""

        try:
            colunas = shutil.get_terminal_size().columns

        except Exception:
            return 100

        # -1: escrever na última coluna já provoca quebra em alguns terminais.
        return max(20, colunas - 1)

    def _linhas(self, resumo, extra=None, misses=None):

        acao = resumo.get("acao") or "—"

        acao_ha = resumo.get("acao_ha")

        # "há quanto tempo" é o que diferencia bot trabalhando
        # de bot parado numa ação que deu certo uma vez.
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

        # Só existe com o debug ligado; aparece sempre (mesmo sem nada a
        # relatar) porque o bloco tem altura fixa.
        if self.misses_line:

            linhas.append(
                f"No detection: {misses}"
                if misses
                else "No detection: —"
            )

        # Trava o tamanho: uma linha nova sem ajustar a conta comeria a
        # linha de cima em vez de falhar visivelmente.
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
        """Redesenha se já passou o intervalo. `resumo` é StateMachine.summary(), `misses` é Detector.miss_report()."""

        agora = time.monotonic()

        if not force and agora - self.ultimo < self.interval:
            return

        self.ultimo = agora

        linhas = self._linhas(resumo, extra, misses)

        if not self.ansi:

            # Sem ANSI, imprime UMA vez e para, em vez do bloco repetido mil vezes.
            if not self.desenhado:

                self.stream.write("\n".join(linhas) + "\n")
                self.stream.flush()

                self.desenhado = True

            return

        saida = []

        if self.desenhado:

            # Sobe ao topo do bloco anterior.
            saida.append(f"\033[{self.linhas}A")

        for linha in linhas:

            # \033[2K limpa a linha inteira, senão o rabo do texto anterior fica na tela.
            saida.append(f"\r\033[2K{linha}\n")

        self.stream.write("".join(saida))
        self.stream.flush()

        self.desenhado = True

    def close(self):
        """Deixa o cursor abaixo do bloco, para log de encerramento/traceback não escrever em cima."""

        if self.desenhado and self.ansi:

            self.stream.write("\n")
            self.stream.flush()

        self.desenhado = False
