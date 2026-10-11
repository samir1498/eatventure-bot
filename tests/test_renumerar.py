"""
Tests for template-numbering compaction. Runs in temp dirs.
Renaming is irreversible: if one file's destination is another's not-yet-
processed name, one overwrites the other and the template is gone.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "tools"))

from renumerar import (                          # noqa: E402
    numerados,
    planejar,
    proximo_numero,
    renumerar,
)


def monta(numeros, extras=()):
    """File content = its own number, to track what became what."""

    pasta = Path(tempfile.mkdtemp())

    for numero in numeros:

        caminho = pasta / f"item_{numero:03d}.png"

        caminho.write_text(str(numero), encoding="utf-8")

    for nome in extras:

        (pasta / nome).write_text("x", encoding="utf-8")

    return pasta


def sequencia(pasta):

    return [n for n, _ in numerados(pasta)]


def conteudos(pasta):

    return {
        n: p.read_text(encoding="utf-8")
        for n, p in numerados(pasta)
    }


def test_fecha_lacuna_no_meio():
    """Missing 3 between 2 and 4: 4 becomes 3 and 5 becomes 4."""

    pasta = monta([1, 2, 4, 5])

    renumerar(pasta, aplicar=True, log=lambda *a: None)

    assert sequencia(pasta) == [1, 2, 3, 4], sequencia(pasta)

    # Content follows: what was 4 is now 3.
    assert conteudos(pasta) == {
        1: "1",
        2: "2",
        3: "4",
        4: "5",
    }, conteudos(pasta)


def test_varias_lacunas():

    pasta = monta([1, 2, 3, 4, 6, 7, 10])

    renumerar(pasta, aplicar=True, log=lambda *a: None)

    assert sequencia(pasta) == [1, 2, 3, 4, 5, 6, 7]

    assert conteudos(pasta)[5] == "6"
    assert conteudos(pasta)[7] == "10"


def test_nao_perde_arquivo():
    """If the rename order were wrong, one would overwrite the other."""

    pasta = monta([2, 5, 9, 12, 13, 20])

    antes = set(conteudos(pasta).values())

    renumerar(pasta, aplicar=True, log=lambda *a: None)

    depois = set(conteudos(pasta).values())

    assert antes == depois, (antes, depois)

    assert sequencia(pasta) == [1, 2, 3, 4, 5, 6]


def test_ja_compacto_nao_mexe():

    pasta = monta([1, 2, 3])

    assert planejar(pasta) == []

    renumerar(pasta, aplicar=True, log=lambda *a: None)

    assert conteudos(pasta) == {1: "1", 2: "2", 3: "3"}


def test_sem_aplicar_nao_renomeia():

    pasta = monta([1, 5, 9])

    mudancas = renumerar(pasta, aplicar=False)

    assert len(mudancas) == 2, mudancas

    # Nothing changed on disk: the CLI default only shows.
    assert sequencia(pasta) == [1, 5, 9], sequencia(pasta)


def test_ignora_nome_fora_do_padrao():

    pasta = monta(
        [1, 4],
        extras=("referencia.png", "item_abc.png"),
    )

    renumerar(pasta, aplicar=True, log=lambda *a: None)

    assert sequencia(pasta) == [1, 2]

    # The intruders stay there, intact.
    assert (pasta / "referencia.png").exists()
    assert (pasta / "item_abc.png").exists()


def test_ordem_crescente_nunca_gera_colisao():
    """
    Ascending order makes collision impossible: each file's target is
    <= its number, and anyone not yet processed has a BIGGER number than
    the source. Verified by brute force, not by argument.
    """

    import itertools

    for tamanho in range(1, 6):

        for numeros in itertools.combinations(range(1, 12), tamanho):

            pasta = monta(list(numeros))

            for origem, destino in planejar(pasta):

                # At rename time, the destination can never
                # be a file not yet processed.
                assert not destino.exists() or destino == origem, (
                    numeros,
                    origem.name,
                    destino.name,
                )

                origem.rename(destino)

            assert sequencia(pasta) == list(
                range(1, tamanho + 1)
            ), (numeros, sequencia(pasta))


def test_guarda_de_sobrescrita(monkeypatch=None):
    """
    Defensive guard: an occupied destination is skipped with a warning instead of destroying the
    file. Collision never happens on the normal path, so the plan is
    injected by hand — a defensive branch that never runs may be broken
    without anyone knowing.
    """

    import renumerar as mod

    pasta = monta([1, 2])

    original = mod.planejar

    # Deliberately invalid plan: sends 2 to become 1.
    mod.planejar = lambda d: [
        (d / "item_002.png", d / "item_001.png")
    ]

    avisos = []

    try:

        feitas = mod.renumerar(
            pasta,
            aplicar=True,
            log=avisos.append,
        )

    finally:

        mod.planejar = original

    assert feitas == [], feitas

    assert avisos, "deveria ter avisado"

    # Both files survive, with original content.
    assert conteudos(pasta) == {1: "1", 2: "2"}, conteudos(pasta)


def test_proximo_numero_depois_de_compactar():

    pasta = monta([1, 2, 4, 5])

    renumerar(pasta, aplicar=True, log=lambda *a: None)

    # 4 compact files -> the new one is 5.
    assert proximo_numero(pasta) == 5, proximo_numero(pasta)


def test_pasta_vazia():

    pasta = monta([])

    assert planejar(pasta) == []

    assert proximo_numero(pasta) == 1


def main():

    testes = [
        valor
        for nome, valor in sorted(globals().items())
        if nome.startswith("test_") and callable(valor)
    ]

    falhas = 0

    for teste in testes:

        try:

            teste()

            print(f"  ok    {teste.__name__}")

        except AssertionError as erro:

            falhas += 1

            print(f"  FALHA {teste.__name__}: {erro}")

        except Exception as erro:

            falhas += 1

            print(
                f"  ERRO  {teste.__name__}: "
                f"{type(erro).__name__}: {erro}"
            )

    print()

    if falhas:

        print(f"{falhas}/{len(testes)} falharam.")

        return 1

    print(f"{len(testes)}/{len(testes)} passaram.")

    return 0


if __name__ == "__main__":

    sys.exit(main())
