"""
Shows the state-machine priorities (derived from the list order in
core/state_machine.py) and checks rules without templates, nonexistent
actions, and templates without rules — things the runtime never warns about.

    python tools/regras.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from actions.manager import ACTION_TABLE          # noqa: E402
from core import log                              # noqa: E402
from core.config import (                         # noqa: E402
    DISMISS_ACTIONS,
    DISMISS_ATTEMPTS_BEFORE_SCROLL,
    STATE_TIMEOUTS,
)
from core.state_machine import (                  # noqa: E402
    DELIVERY,
    FOOD,
    HELPER,
    NEW_POINT,
    NORMAL,
    RENOVATE,
    STATE_RULES,
    UPGRADE,
)
from vision.detector import Detector              # noqa: E402

ORDEM = [NORMAL, UPGRADE, RENOVATE, DELIVERY, HELPER, FOOD, NEW_POINT]

# States whose rules are NOT applied by the table: they have
# their own handler because of the wait after the press.
HANDLER_PROPRIO = {FOOD, NEW_POINT}


def linha(largura=64):

    print("-" * largura)


def main():

    log.setup("ERROR")

    detector = Detector()

    com_template = {t["category"] for t in detector.templates}

    problemas = []
    usadas = set()

    for state in ORDEM:

        rules = STATE_RULES.get(state, [])

        timeout = STATE_TIMEOUTS.get(state)

        print()
        linha()

        cabecalho = f" {state}"

        if timeout:
            cabecalho += f"   (timeout {timeout:.0f}s)"
        else:
            cabecalho += "   (sem timeout — estado base)"

        if state in HANDLER_PROPRIO:
            cabecalho += "  [handler próprio]"

        print(cabecalho)
        linha()

        for indice, (category, action, next_state) in enumerate(
            rules,
            start=1,
        ):

            usadas.add(category)

            marcas = []

            if category not in com_template:

                marcas.append("SEM TEMPLATE")

                problemas.append(
                    f"{state} prioridade {indice}: categoria "
                    f"'{category}' não tem template — a regra "
                    f"nunca pode disparar"
                )

            if action not in ACTION_TABLE:

                marcas.append("AÇÃO INEXISTENTE")

                problemas.append(
                    f"{state} prioridade {indice}: ação "
                    f"'{action}' não está na ACTION_TABLE"
                )

            if action in DISMISS_ACTIONS:

                marcas.append(
                    f"escala após "
                    f"{DISMISS_ATTEMPTS_BEFORE_SCROLL}"
                )

            comportamento = ACTION_TABLE.get(action, "?")

            destino = (
                f"-> {next_state}"
                if next_state
                else "(fica)"
            )

            aviso = (
                "  << " + ", ".join(marcas)
                if marcas
                else ""
            )

            print(
                f"  {indice:2d}. {category:14s} "
                f"{action:18s} {comportamento:8s} "
                f"{destino:16s}{aviso}"
            )

        if not rules:
            print("  (nenhuma regra)")

    orfaos = sorted(com_template - usadas)

    if orfaos:

        for category in orfaos:

            problemas.append(
                f"categoria '{category}' tem template mas "
                f"nenhuma regra a usa — custo de detecção "
                f"sem uso"
            )

    print()
    linha()

    if problemas:

        print(f" {len(problemas)} problema(s)")
        linha()

        for problema in problemas:
            print(f"  ! {problema}")

    else:

        print(" nenhum problema")
        linha()

    print()
    print("Prioridade = ordem da lista em")
    print("  src/core/state_machine.py")
    print()
    print("Para acompanhar as decisões em tempo real, ponha")
    print("  LOG_LEVEL = \"DEBUG\"")
    print("em src/core/config.py.")
    print()

    return 1 if problemas else 0


if __name__ == "__main__":

    sys.exit(main())
