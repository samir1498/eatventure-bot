"""
Tests for the template selector's scale and coordinates. A mapping
error here raises no error — it silently saves the WRONG crop,
and only shows up weeks later as "the bot clicks the wrong place". That's
why the math is tested apart from the GUI.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "tools"))

from selector_layout import (                    # noqa: E402
    escala,
    para_canvas,
    para_imagem,
    tamanho_canvas,
)


# Device screen, as it is: tall and narrow.
LARGURA = 1080
ALTURA = 2400


def test_a_altura_e_quem_manda():
    """
    The window follows the screen HEIGHT. On a tall narrow screen the
    width has slack to spare, and doubling the width cap changes nothing.
    """

    valor = escala(LARGURA, ALTURA, 3268, 1152)

    assert abs(valor - 1152 / ALTURA) < 1e-9, valor

    dobrado = escala(LARGURA, ALTURA, 6536, 1152)

    assert dobrado == valor, (valor, dobrado)


def test_fracao_de_80_por_cento_de_1440():
    """Real case: 3440x1440 monitor, 0.80 fraction."""

    valor = escala(
        LARGURA,
        ALTURA,
        int(3440 * 0.95),
        int(1440 * 0.80),
    )

    assert abs(valor - 0.480) < 0.001, valor

    assert tamanho_canvas(LARGURA, ALTURA, valor) == (518, 1152)


def test_largura_ainda_protege():
    """Width must still cap when it's actually the bottleneck."""

    valor = escala(LARGURA, ALTURA, 300, 5000)

    assert abs(valor - 300 / LARGURA) < 1e-9, valor


def test_nao_amplia_por_padrao():
    """Upscaling creates no detail: it just blurs the crop."""

    assert escala(100, 100, 2000, 2000) == 1.0

    assert escala(
        100,
        100,
        2000,
        2000,
        permitir_ampliar=True,
    ) == 20.0


def test_mantem_a_proporcao():
    """The window ratio must match the image ratio."""

    valor = escala(LARGURA, ALTURA, 3268, 1152)

    largura_janela, altura_janela = tamanho_canvas(
        LARGURA,
        ALTURA,
        valor,
    )

    proporcao_imagem = LARGURA / ALTURA
    proporcao_janela = largura_janela / altura_janela

    assert abs(proporcao_imagem - proporcao_janela) < 0.002, (
        proporcao_imagem,
        proporcao_janela,
    )


def test_janela_cabe_no_limite():

    for max_altura in (600, 900, 1152, 1296, 1400):

        valor = escala(LARGURA, ALTURA, 3268, max_altura)

        largura_janela, altura_janela = tamanho_canvas(
            LARGURA,
            ALTURA,
            valor,
        )

        assert altura_janela <= max_altura, (
            max_altura,
            altura_janela,
        )

        assert largura_janela <= 3268, (
            max_altura,
            largura_janela,
        )


def test_ida_e_volta_em_toda_a_area():
    """
    window -> image -> window must return to the same pixel, within
    rounding error. Tested over the WHOLE area, not just corners:
    a sign error in the middle of the image would pass a corner test.
    """

    for max_altura in (900, 1152, 1296):

        valor = escala(LARGURA, ALTURA, 3268, max_altura)

        largura_janela, altura_janela = tamanho_canvas(
            LARGURA,
            ALTURA,
            valor,
        )

        folga = 1 / valor + 1

        for x_janela in range(0, largura_janela, 17):

            for y_janela in range(0, altura_janela, 37):

                x_imagem, y_imagem = para_imagem(
                    x_janela,
                    y_janela,
                    LARGURA,
                    ALTURA,
                    valor,
                )

                assert 0 <= x_imagem <= LARGURA
                assert 0 <= y_imagem <= ALTURA

                x_volta, y_volta = para_canvas(
                    x_imagem,
                    y_imagem,
                    valor,
                )

                assert abs(x_volta - x_janela) <= folga, (
                    max_altura,
                    x_janela,
                    x_imagem,
                    x_volta,
                )

                assert abs(y_volta - y_janela) <= folga, (
                    max_altura,
                    y_janela,
                    y_imagem,
                    y_volta,
                )


def test_coordenada_presa_na_imagem():
    """Numpy silently accepts negative indices and crops from the other side of the image."""

    valor = escala(LARGURA, ALTURA, 3268, 1152)

    for x_janela, y_janela in (
        (-50, -50),
        (-1, 10),
        (99999, 99999),
        (10, 99999),
    ):

        x_imagem, y_imagem = para_imagem(
            x_janela,
            y_janela,
            LARGURA,
            ALTURA,
            valor,
        )

        assert 0 <= x_imagem <= LARGURA, (x_janela, x_imagem)
        assert 0 <= y_imagem <= ALTURA, (y_janela, y_imagem)


def test_origem_e_fim_batem():

    valor = escala(LARGURA, ALTURA, 3268, 1152)

    assert para_imagem(0, 0, LARGURA, ALTURA, valor) == (0, 0)

    largura_janela, altura_janela = tamanho_canvas(
        LARGURA,
        ALTURA,
        valor,
    )

    x_imagem, y_imagem = para_imagem(
        largura_janela,
        altura_janela,
        LARGURA,
        ALTURA,
        valor,
    )

    # 1 px tolerance: the window width is rounded.
    assert LARGURA - x_imagem <= 1, x_imagem
    assert ALTURA - y_imagem <= 1, y_imagem


def test_meio_da_janela_e_o_meio_da_imagem():
    """Catches a swapped axis or forgotten offset."""

    valor = escala(LARGURA, ALTURA, 3268, 1152)

    largura_janela, altura_janela = tamanho_canvas(
        LARGURA,
        ALTURA,
        valor,
    )

    x_imagem, y_imagem = para_imagem(
        largura_janela // 2,
        altura_janela // 2,
        LARGURA,
        ALTURA,
        valor,
    )

    assert abs(x_imagem - LARGURA // 2) <= 3, x_imagem
    assert abs(y_imagem - ALTURA // 2) <= 3, y_imagem


def _arrasta(imagem, valor, canto_a, canto_b):
    """Simulates the mouse drag the way template_selector does it."""

    altura, largura = imagem.shape[:2]

    x1, y1 = para_imagem(
        canto_a[0],
        canto_a[1],
        largura,
        altura,
        valor,
    )

    x2, y2 = para_imagem(
        canto_b[0],
        canto_b[1],
        largura,
        altura,
        valor,
    )

    return imagem[
        min(y1, y2):max(y1, y2),
        min(x1, x2):max(x1, x2),
    ]


def test_arrasto_recorta_a_regiao_certa():
    """
    End-to-end proof, no GUI: white marker at a known position, and the
    drag over it on the shrunk window must come out white. If the
    mapping had an offset, the crop would come out black — template of
    the wrong screen region.
    """

    imagem = np.zeros((ALTURA, LARGURA, 3), np.uint8)

    # Asymmetric position, so a swapped axis can't pass by accident.
    alvo = (300, 1700, 460, 1820)   # x1, y1, x2, y2

    imagem[alvo[1]:alvo[3], alvo[0]:alvo[2]] = 255

    for max_altura in (900, 1152, 1296):

        valor = escala(LARGURA, ALTURA, 3268, max_altura)

        # Well inside the marker, so rounding doesn't touch the edge.
        canto_a = para_canvas(alvo[0] + 12, alvo[1] + 12, valor)
        canto_b = para_canvas(alvo[2] - 12, alvo[3] - 12, valor)

        crop = _arrasta(imagem, valor, canto_a, canto_b)

        assert crop.size > 0, max_altura

        assert crop.min() == 255, (
            max_altura,
            crop.mean(),
            "recortou fora do marcador",
        )


def test_arrasto_para_fora_nao_estoura():

    imagem = np.zeros((ALTURA, LARGURA, 3), np.uint8)

    valor = escala(LARGURA, ALTURA, 3268, 1152)

    crop = _arrasta(imagem, valor, (-500, -500), (99999, 99999))

    assert crop.shape[:2] == (ALTURA, LARGURA), crop.shape


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
