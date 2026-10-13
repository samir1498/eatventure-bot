"""Log central; timestamp com milissegundos permite separar "detectou errado" de "detectou tarde"."""

import logging
import os
import sys


def _formatter():

    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname).1s "
            "[%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def setup(level="INFO"):

    root = logging.getLogger("eatventure")

    if root.handlers:
        return root

    root.setLevel(
        getattr(logging, level, logging.INFO)
    )

    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(_formatter())

    root.addHandler(handler)

    # O painel redesenha o stdout do tmux por cima das linhas de log,
    # então um aviso some em segundos. Com EATVENTURE_LOG_FILE o mesmo
    # registro vai para disco, onde sobrevive ao redesenho.
    path = os.environ.get("EATVENTURE_LOG_FILE")

    if path:

        arquivo = logging.FileHandler(path, encoding="utf-8")

        arquivo.setFormatter(_formatter())

        root.addHandler(arquivo)

    return root


def set_console_level(level):
    """
    Muda o nível do que APARECE no terminal, sem tocar no nível do
    logger — filtrar no logger perderia o registro para sempre, e o
    painel de status precisa só que o INFO pare de rolar na tela.
    """

    root = logging.getLogger("eatventure")

    alvo = (
        level
        if isinstance(level, int)
        else getattr(logging, str(level).upper(), logging.WARNING)
    )

    for handler in root.handlers:

        # FileHandler herda de StreamHandler: sem esta exceção o
        # arquivo em disco seria silenciado junto com o terminal.
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(alvo)

    return alvo


def get(name):

    return logging.getLogger(
        f"eatventure.{name}"
    )
