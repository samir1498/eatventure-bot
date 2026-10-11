"""Android device choice: USB+wifi from the same handset show as two entries in `adb devices`, so resolve which serial to use (--device > DEVICE_SERIAL > single connected > prompt)."""

import subprocess
import sys

from core import log, config

logger = log.get("devices")
ADB = config.ADB_PATH


def listar():
    """Serials in 'device' state — 'unauthorized'/'offline' are left out for not accepting commands."""

    resultado = subprocess.run(
        [ADB, "devices"],
        capture_output=True,
        text=True,
        check=True,
    )

    seriais = []

    for linha in resultado.stdout.splitlines():

        linha = linha.strip()

        if not linha or linha.startswith("List of devices"):
            continue

        partes = linha.split()

        if len(partes) >= 2 and partes[1] == "device":

            seriais.append(partes[0])

    return seriais


def _propriedade(serial, nome):
    """getprop, or None on failure — purely informational, failure interrupts nothing."""

    try:

        resultado = subprocess.run(
            [ADB, "-s", serial, "shell", "getprop", nome],
            capture_output=True,
            text=True,
            timeout=5,
        )

    except (OSError, subprocess.SubprocessError):

        return None

    if resultado.returncode != 0:
        return None

    return resultado.stdout.strip() or None


def descrever(seriais):
    """[(serial, model, wifi, hardware)]; `hardware` is ro.serialno, which reveals duplicate entries of the same handset."""

    fichas = []

    for serial in seriais:

        fichas.append(
            (
                serial,
                _propriedade(serial, "ro.product.model"),
                ":" in serial,
                _propriedade(serial, "ro.serialno"),
            )
        )

    return fichas


def rotular(fichas):
    """One line per device; marks the duplicate because wifi drops on its own and kills capture mid-session."""

    # hardware -> first position (1-based) where it appeared
    primeira = {}

    for indice, (_, _, _, hardware) in enumerate(fichas, start=1):

        if hardware and hardware not in primeira:

            primeira[hardware] = indice

    linhas = []

    for indice, ficha in enumerate(fichas, start=1):

        serial, modelo, wifi, hardware = ficha

        partes = [f"{serial:<22}"]

        if modelo:
            partes.append(f"{modelo:<20}")

        partes.append("wifi" if wifi else "USB ")

        if hardware and primeira.get(hardware, indice) != indice:

            partes.append(
                f"(mesmo aparelho do {primeira[hardware]})"
            )

        linhas.append(f"  {indice} - " + " ".join(partes).rstrip())

    return linhas


def padrao(fichas):
    """Index (1-based) suggested: first device over USB — wifi drops on its own and kills capture."""

    for indice, (_, _, wifi, _) in enumerate(fichas, start=1):

        if not wifi:
            return indice

def escolher(seriais, pedido=None, perguntar=None, fichas=None):
    """
    Returns the serial to use.

    `pedido`    pinned serial (--device or config)
    `perguntar` function receiving the text and returning the answer;
                None = no prompt (raises an error instead)
    `fichas`    descrever() result; None = describe here

    Everything touching the outside world comes in via parameters, so the decision
    is testable with no device plugged in.
    """

    if not seriais:

        raise RuntimeError(
            "Nenhum dispositivo Android encontrado.\n"
            "Confira o cabo, a depuração USB e 'adb devices'."
        )

    if pedido:

        if pedido in seriais:
            return pedido

        raise RuntimeError(
            f"Dispositivo '{pedido}' não está conectado.\n"
            "Disponíveis:\n"
            + "\n".join(f"  {s}" for s in seriais)
        )

    if len(seriais) == 1:
        return seriais[0]

    if fichas is None:
        fichas = descrever(seriais)

    sugerido = padrao(fichas)

    texto = "\n".join(
        [
            "",
            "Mais de um dispositivo conectado:",
            "",
        ]
        + rotular(fichas)
        + [
            "",
            f"Qual usar? [{sugerido}]: ",
        ]
    )

    def sem_resposta():

        return RuntimeError(
            "Mais de um dispositivo conectado e não há como "
            "perguntar (entrada não interativa).\n"
            + "\n".join(rotular(fichas))
            + "\n\nUse --device SERIAL ou defina DEVICE_SERIAL "
            "no config."
        )

    if perguntar is None:

        raise sem_resposta()

    while True:

        # EOF = redirected input; isatty() alone does not catch this on
        # Windows (`python main.py < NUL` passes the check and only fails here).
        try:

            resposta = (perguntar(texto) or "").strip()

        except EOFError:

            raise sem_resposta() from None

        if not resposta:
            return fichas[sugerido - 1][0]

        if resposta.isdigit():

            escolha = int(resposta)

            if 1 <= escolha <= len(fichas):
                return fichas[escolha - 1][0]

            print(f"Escolha entre 1 e {len(fichas)}.")

            continue

        if resposta in seriais:
            return resposta

        print("Número da lista ou serial completo.")


def resolver(pedido=None):
    """Normal path: list, choose (prompting in the terminal if needed) and log the chosen one."""

    seriais = listar()

    # Without a terminal, hanging on an invisible input is worse than a clear error.
    perguntar = input if sys.stdin and sys.stdin.isatty() else None

    serial = escolher(seriais, pedido=pedido, perguntar=perguntar)

    logger.info("Device: %s", serial)

    return serial
