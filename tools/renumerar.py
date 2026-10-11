"""
Compacts template numbering: item_001..item_NNN, no gaps
(a gap leaves the "next number" ambiguous). Dry-runs by default, since
renaming is irreversible; processes in ascending order to never
overwrite, and skips with a warning if the destination exists.

    python tools/renumerar.py              # shows what would change
    python tools/renumerar.py --aplicar    # renames
    python tools/renumerar.py --aplicar --categoria food
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = ROOT / "src" / "vision" / "templates"

# The name may carry the resolution the crop was taken at
# ("1080x2400_item_001.png"): in default/ that is provenance, since it
# receives crops from different screens. The prefix is preserved on rename — the
# numbering is unique within the category, regardless of prefix.
NUMERADO = re.compile(r"^(?:(\d+x\d+)_)?item_(\d+)$")


def _partes(path):
    """(prefix ready to concatenate, number), or None if not a numbered item."""

    casou = NUMERADO.match(path.stem)

    if casou is None:
        return None

    prefixo, numero = casou.groups()

    return (
        f"{prefixo}_" if prefixo else "",
        int(numero),
    )


def numerados(category_dir):
    """[(number, path)] of item_NNN.png, with or without prefix, in ascending order."""

    encontrados = []

    for path in category_dir.glob("*.png"):

        partes = _partes(path)

        if partes is not None:

            encontrados.append((partes[1], path))

    # Name breaks ties: two prefixes may bring the same number, and without
    # a stable criterion the order would come from the folder and the plan would change on every run.
    encontrados.sort(key=lambda item: (item[0], item[1].name))

    return encontrados


def planejar(category_dir):
    """[(source, destination)] to make the sequence contiguous; empty = already compact."""

    mudancas = []

    for indice, (_, path) in enumerate(
        numerados(category_dir),
        start=1,
    ):

        prefixo = _partes(path)[0]

        destino = (
            category_dir
            / f"{prefixo}item_{indice:03d}.png"
        )

        if destino == path:
            continue

        mudancas.append((path, destino))

    return mudancas


def renumerar(category_dir, aplicar=False, log=print):
    """Compacts the category; returns the changes made (or planned, if aplicar=False)."""

    mudancas = planejar(category_dir)

    if not aplicar:
        return mudancas

    feitas = []

    for origem, destino in mudancas:

        if destino.exists():

            log(
                f"  ! {destino.name} já existe — "
                f"{origem.name} não foi renomeado"
            )

            continue

        origem.rename(destino)

        feitas.append((origem, destino))

    return feitas


def proximo_numero(category_dir):
    """Next template number, assuming the sequence is already compact: len + 1."""

    return len(numerados(category_dir)) + 1


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compacta a numeração dos templates. Sem "
            "--aplicar, só mostra."
        )
    )

    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="renomeia de verdade",
    )

    parser.add_argument(
        "--categoria",
        help="só esta categoria (padrão: todas)",
    )

    args = parser.parse_args()

    if not TEMPLATES_DIR.exists():

        print(f"Pasta não encontrada: {TEMPLATES_DIR}")

        return 1

    # TEMPLATES_DIR/default/<categoria> and TEMPLATES_DIR/<resolucao>/<categoria> —
    # two layers, not one: the group (default or resolution) is not a category.
    pastas = sorted(
        categoria
        for grupo in TEMPLATES_DIR.iterdir()
        if grupo.is_dir()
        for categoria in grupo.iterdir()
        if categoria.is_dir()
    )

    if args.categoria:

        pastas = [
            d for d in pastas if d.name == args.categoria
        ]

        if not pastas:

            print(f"Categoria '{args.categoria}' não existe.")

            return 1

    total = 0

    for pasta in pastas:

        mudancas = renumerar(pasta, aplicar=args.aplicar)

        if not mudancas:
            continue

        total += len(mudancas)

        verbo = "renomeado" if args.aplicar else "renomearia"

        print(f"\n{pasta.name}:")

        for origem, destino in mudancas:

            print(
                f"  {verbo}: {origem.name} -> {destino.name}"
            )

    print()

    if not total:

        print("Tudo já está compacto.")

        return 0

    if args.aplicar:

        print(f"{total} arquivo(s) renomeado(s).")

    else:

        print(
            f"{total} arquivo(s) mudariam. "
            f"Rode com --aplicar."
        )

    return 0


if __name__ == "__main__":

    sys.exit(main())
