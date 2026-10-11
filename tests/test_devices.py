"""
Device-selection tests. The records are injected, nothing talks to adb:
what's under test is the DECISION of which serial comes out.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from core import devices                         # noqa: E402


# The real case: same phone over USB and over wifi.
USB = "e2615705"
WIFI = "192.168.1.12:37889"

DOIS = [USB, WIFI]

FICHAS_DOIS = [
    (USB, "Poco X6 Pro", False, "e2615705"),
    (WIFI, "Poco X6 Pro", True, "e2615705"),
]


def _responde(*respostas):
    """Blows up if asked more times than expected — an infinite loop is worse than a red test."""

    fila = list(respostas)

    def perguntar(_texto):

        if not fila:
            raise AssertionError("perguntou demais")

        return fila.pop(0)

    return perguntar


def test_um_device_nao_pergunta():

    assert devices.escolher([USB], perguntar=_responde()) == USB


def test_serial_fixado_ganha_da_pergunta():
    """--device / DEVICE_SERIAL exist so you don't see the prompt every run."""

    escolhido = devices.escolher(
        DOIS,
        pedido=WIFI,
        perguntar=_responde(),
        fichas=FICHAS_DOIS,
    )

    assert escolhido == WIFI


def test_serial_fixado_inexistente_e_erro_claro():

    try:

        devices.escolher(DOIS, pedido="sumiu", fichas=FICHAS_DOIS)

    except RuntimeError as erro:

        # The error must show what EXISTS, otherwise the user guesses the serial.
        assert USB in str(erro), str(erro)
        assert WIFI in str(erro), str(erro)

        return

    raise AssertionError("deveria ter levantado")


def test_nenhum_device():

    try:

        devices.escolher([], perguntar=_responde())

    except RuntimeError as erro:

        assert "Nenhum" in str(erro), str(erro)

        return

    raise AssertionError("deveria ter levantado")


def test_sem_terminal_nao_trava():
    """With no way to ask (pipe, cron, IDE), hanging on an invisible input is worse than an error."""

    try:

        devices.escolher(DOIS, perguntar=None, fichas=FICHAS_DOIS)

    except RuntimeError as erro:

        texto = str(erro)

        assert "--device" in texto, texto
        assert "DEVICE_SERIAL" in texto, texto
        assert USB in texto, texto

        return

    raise AssertionError("deveria ter levantado")


def test_escolhe_pelo_numero():

    for resposta, esperado in (("1", USB), ("2", WIFI)):

        escolhido = devices.escolher(
            DOIS,
            perguntar=_responde(resposta),
            fichas=FICHAS_DOIS,
        )

        assert escolhido == esperado, (resposta, escolhido)


def test_enter_aceita_a_sugestao():

    escolhido = devices.escolher(
        DOIS,
        perguntar=_responde(""),
        fichas=FICHAS_DOIS,
    )

    assert escolhido == USB, escolhido


def test_aceita_serial_digitado():
    """Pasting the serial is natural after reading the list."""

    escolhido = devices.escolher(
        DOIS,
        perguntar=_responde(WIFI),
        fichas=FICHAS_DOIS,
    )

    assert escolhido == WIFI


def test_resposta_invalida_pergunta_de_novo():
    """An out-of-range answer must not land on some device by accident."""

    escolhido = devices.escolher(
        DOIS,
        perguntar=_responde("9", "banana", "0", "-1", "2"),
        fichas=FICHAS_DOIS,
    )

    assert escolhido == WIFI, escolhido


def test_pergunta_mostra_os_seriais():

    textos = []

    def perguntar(texto):

        textos.append(texto)

        return "1"

    devices.escolher(
        DOIS,
        perguntar=perguntar,
        fichas=FICHAS_DOIS,
    )

    texto = textos[0]

    assert USB in texto, texto
    assert WIFI in texto, texto

    # Without the model and the USB/wifi tag, choosing is guessing.
    assert "Poco X6 Pro" in texto, texto
    assert "wifi" in texto, texto
    assert "USB" in texto, texto


def test_sugere_usb_sobre_wifi():
    """USB over wifi on purpose: the adb wifi connection drops on its own."""

    # USB second in the list.
    fichas = [
        (WIFI, "Poco X6 Pro", True, "e2615705"),
        (USB, "Poco X6 Pro", False, "e2615705"),
    ]

    assert devices.padrao(fichas) == 2, devices.padrao(fichas)

    assert devices.padrao(FICHAS_DOIS) == 1


def test_sugere_o_primeiro_quando_todos_wifi():

    fichas = [
        ("10.0.0.2:5555", "A", True, "aaa"),
        ("10.0.0.3:5555", "B", True, "bbb"),
    ]

    assert devices.padrao(fichas) == 1


def test_marca_o_mesmo_aparelho():
    """Two entries with the SAME ro.serialno are the same phone."""

    linhas = devices.rotular(FICHAS_DOIS)

    assert "mesmo aparelho" not in linhas[0], linhas[0]
    assert "mesmo aparelho do 1" in linhas[1], linhas[1]


def test_nao_marca_aparelhos_diferentes():

    fichas = [
        (USB, "Poco", False, "aaa"),
        ("outro", "Galaxy", False, "bbb"),
    ]

    for linha in devices.rotular(fichas):

        assert "mesmo aparelho" not in linha, linha


def test_rotula_sem_getprop():
    """getprop can fail (slow device, no permission); the list still has to come out."""

    fichas = [
        (USB, None, False, None),
        (WIFI, None, True, None),
    ]

    linhas = devices.rotular(fichas)

    assert USB in linhas[0], linhas[0]
    assert WIFI in linhas[1], linhas[1]

    for linha in linhas:

        assert "None" not in linha, linha


def test_escolha_funciona_sem_getprop():

    fichas = [
        (USB, None, False, None),
        (WIFI, None, True, None),
    ]

    escolhido = devices.escolher(
        DOIS,
        perguntar=_responde(""),
        fichas=fichas,
    )

    assert escolhido == USB, escolhido


# These tests verify the "-s <serial>", not the adb binary: config
# resolves the full path (PATH or tools/scrcpy), so comparing against
# the literal "adb" would fail on a path like
# "tools/scrcpy/adb.EXE" — which is the RIGHT path.

def _sem_o_binario(comando):
    """Checks that the command starts with some adb and returns the rest, which is what's tested."""

    assert comando, comando

    assert "adb" in Path(comando[0]).name.lower(), comando

    return comando[1:]


def test_captura_usa_o_serial():
    """Without -s, with two devices listed, adb refuses EVERY call."""

    from capture.screen import ScreenCapture

    capture = ScreenCapture(USB)

    comando = capture._adb("shell", "ls")

    assert _sem_o_binario(comando)[:2] == ["-s", USB], comando


def test_captura_sem_serial_continua_valendo():
    """One device only: -s is dispensable, and the no-argument constructor keeps working."""

    from capture.screen import ScreenCapture

    comando = ScreenCapture()._adb("devices")

    assert _sem_o_binario(comando) == ["devices"], comando


def test_toques_usam_o_serial():

    from actions.android import AndroidActions

    comando = AndroidActions(USB)._base()

    assert _sem_o_binario(comando)[:2] == ["-s", USB], comando


def test_screenshot_usa_o_serial():
    """The template_selector screencap has the SAME problem: without -s, no new template comes out."""

    sys.path.insert(0, str(ROOT / "tests"))

    from android_screenshot import AndroidScreenshot

    comando = AndroidScreenshot(serial=USB)._adb(
        "exec-out",
        "screencap",
        "-p",
    )

    assert _sem_o_binario(comando) == [
        "-s",
        USB,
        "exec-out",
        "screencap",
        "-p",
    ], comando


def test_screenshot_sem_serial_continua_valendo():

    sys.path.insert(0, str(ROOT / "tests"))

    from android_screenshot import AndroidScreenshot

    comando = AndroidScreenshot()._adb("exec-out")

    assert _sem_o_binario(comando) == ["exec-out"], comando


def test_selector_repassa_o_serial():
    """The chosen serial has to travel through TemplateSelector down to the screencap."""

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "tools"))

    from template_selector import TemplateSelector

    selector = TemplateSelector(USB)

    comando = selector.capture._adb("exec-out")

    assert _sem_o_binario(comando)[:2] == ["-s", USB], comando


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
