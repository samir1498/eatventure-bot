"""
Dataset-recording tests. Records into a temp dir, never into
DATASET_DIR. A wrong label here raises no error — it trains the model
to click the wrong place, and the defect only shows after hours of GPU.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core import log                              # noqa: E402
from core import state_machine as sm              # noqa: E402
from core.config import (                         # noqa: E402
    ACTION_SETTLE,
    DISMISS_POINT,
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
)
from dataset.recorder import (                    # noqa: E402
    CHANGED,
    UNCHANGED,
    UNKNOWN,
    DatasetRecorder,
)

log.setup("ERROR")


def frame(valor=40, largura=1080, altura=2400):
    """Frame with a gradient, so the similarity hash tells them apart."""

    img = np.full((altura, largura, 3), valor, np.uint8)

    # Value-dependent position: two different frames never collide on the hash.
    y = (valor * 7) % (altura - 200)

    img[y:y + 180, 100:400] = 255 - valor

    return img


def deteccao(category, x=100, y=200, w=80, h=80, conf=0.97):

    return {
        "category": category,
        "name": f"{category}_item_001.png",
        "confidence": conf,
        "color_similarity": 0.99,
        "min_threshold": 0.9,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
    }


class Gravador:
    """
    Recorder in a temp directory, already started. Negatives OFF
    by default (interval 0): one extra would change the totals in the
    counting tests.
    """

    def __init__(self, negativas=0.0):

        self.negativas = negativas

    def __enter__(self):

        import dataset.recorder as mod

        self._taxa = mod.DATASET_NEGATIVE_INTERVAL

        mod.DATASET_NEGATIVE_INTERVAL = self.negativas

        self.dir = Path(tempfile.mkdtemp())

        self.rec = DatasetRecorder(directory=self.dir)

        self.rec.start()

        return self

    def __exit__(self, *a):

        import dataset.recorder as mod

        self.rec.stop()

        mod.DATASET_NEGATIVE_INTERVAL = self._taxa

        shutil.rmtree(self.dir, ignore_errors=True)

    def registros(self, espera=5.0):
        """Reads the index, waiting for the thread to drain the queue."""

        indice = self.dir / "samples.jsonl"

        limite = time.monotonic() + espera

        while time.monotonic() < limite:

            if self.rec.queue.empty() and indice.exists():

                # Some slack for the write to finish.
                time.sleep(0.05)

                break

            time.sleep(0.02)

        if not indice.exists():
            return []

        return [
            json.loads(linha)
            for linha in indice.read_text(
                encoding="utf-8"
            ).splitlines()
            if linha.strip()
        ]


def test_grava_imagem_e_rotulo():

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(340, 1540),
            state="NORMAL",
            frame=frame(50),
            detections=[alvo],
            lag=0.4,
            cycle_count=2,
        )

        # Later frame without the target: the action worked.
        g.rec.observe(
            frame(51),
            [],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        regs = g.registros()

        assert len(regs) == 1, regs

        r = regs[0]

        assert r["action"] == "open_box"
        assert r["action_kind"] == "click"
        assert r["target_category"] == "box"
        assert r["click_x"] == 340 and r["click_y"] == 1540
        assert r["state"] == "NORMAL"
        assert r["frame_width"] == 1080
        assert r["frame_height"] == 2400
        assert r["cycle_count"] == 2
        assert r["detect_lag_ms"] == 400

        # The image really exists, at the noted path.
        caminho = g.dir / r["image"]

        assert caminho.exists(), r["image"]

        assert caminho.stat().st_size > 0

        # The path is RELATIVE: moving the folder doesn't invalidate
        # the index.
        assert not Path(r["image"]).is_absolute()


def test_grava_todas_as_caixas_do_frame():
    """The bot acted on one; the detector saw three — the others are free labels."""

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        todas = [
            alvo,
            deteccao("food", x=700, y=900),
            deteccao("upgrade", x=50, y=2000),
        ]

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(340, 1540),
            state="NORMAL",
            frame=frame(60),
            detections=todas,
            lag=0.3,
        )

        g.rec.observe(frame(61), [], time.monotonic() + 1.0, "NORMAL")

        r = g.registros()[0]

        assert len(r["boxes"]) == 3, r["boxes"]

        categorias = sorted(b["category"] for b in r["boxes"])

        assert categorias == ["box", "food", "upgrade"]

        # Exactly one marked as the one acted upon.
        agidas = [b for b in r["boxes"] if b["acted"]]

        assert len(agidas) == 1, agidas

        assert agidas[0]["category"] == "box"
        assert agidas[0]["x"] == 300
        assert agidas[0]["y"] == 1500


def test_geometria_da_caixa_e_preservada():
    """x/y/width/height are the detection label: getting this wrong trains the model to frame wrong."""

    with Gravador() as g:

        alvo = deteccao("box", x=317, y=1523, w=96, h=88)

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(365, 1567),
            state="NORMAL",
            frame=frame(70),
            detections=[alvo],
            lag=0.2,
        )

        g.rec.observe(frame(71), [], time.monotonic() + 1.0, "NORMAL")

        caixa = g.registros()[0]["boxes"][0]

        assert (
            caixa["x"],
            caixa["y"],
            caixa["width"],
            caixa["height"],
        ) == (317, 1523, 96, 88), caixa


def test_resultado_changed_quando_o_alvo_sai():

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(340, 1540),
            state="NORMAL",
            frame=frame(80),
            detections=[alvo],
            lag=0.3,
        )

        g.rec.observe(frame(81), [], time.monotonic() + 1.0, "NORMAL")

        r = g.registros()[0]

        assert r["outcome"] == CHANGED, r["outcome"]

        assert r["outcome_after_ms"] is not None


def test_resultado_unchanged_quando_o_alvo_fica():
    """The label that breaks the behavior-cloning ceiling: allows training only on actions that worked."""

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(340, 1540),
            state="NORMAL",
            frame=frame(90),
            detections=[alvo],
            lag=0.3,
        )

        # Same target, same place: it didn't take.
        g.rec.observe(
            frame(91),
            [deteccao("box", x=302, y=1503)],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        assert g.registros()[0]["outcome"] == UNCHANGED


def test_outra_instancia_da_categoria_nao_conta_como_unchanged():
    """Comparing only the category would give a false "didn't take" when there's another food in another corner."""

    with Gravador() as g:

        alvo = deteccao("food", x=100, y=300)

        g.rec.on_action(
            action="food",
            action_kind="click",
            detection=alvo,
            click=(140, 340),
            state="NORMAL",
            frame=frame(100),
            detections=[alvo],
            lag=0.3,
        )

        # Food yes, but on the other side of the screen.
        g.rec.observe(
            frame(101),
            [deteccao("food", x=900, y=2000)],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        assert g.registros()[0]["outcome"] == CHANGED


def test_frame_anterior_a_acao_nao_julga():
    """Judging by a pre-action frame would produce "unchanged" for every action that worked."""

    with Gravador() as g:

        agora = time.monotonic()

        alvo = deteccao("box")

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(140, 240),
            state="NORMAL",
            frame=frame(110),
            detections=[alvo],
            lag=0.5,
        )

        # BEFORE frame: nothing recorded yet.
        g.rec.observe(frame(111), [], agora - 1.0, "NORMAL")

        assert g.rec.pending is not None

        assert g.registros(espera=0.3) == []

        # After frame: now yes.
        g.rec.observe(frame(112), [], time.monotonic() + 1.0, "NORMAL")

        assert len(g.registros()) == 1


def test_acao_sem_veredito_e_gravada_no_stop():
    """Session ending with a pending action: the image and boxes are worth it even without the verdict."""

    d = Path(tempfile.mkdtemp())

    try:

        rec = DatasetRecorder(directory=d)

        rec.start()

        alvo = deteccao("box")

        rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(140, 240),
            state="NORMAL",
            frame=frame(120),
            detections=[alvo],
            lag=0.3,
        )

        rec.stop()

        linhas = (d / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()

        assert len(linhas) == 1, linhas

        assert json.loads(linhas[0])["outcome"] == UNKNOWN

    finally:

        shutil.rmtree(d, ignore_errors=True)


def test_todas_as_acoes_do_bot_entram():
    """
    An action the bot takes but the dataset ignores is a hole that only shows when
    the model can't do that thing. Includes `click` (build) and `plane`,
    which used to be left out.
    """

    from actions.manager import ACTION_TABLE
    from core.config import DATASET_ACTIONS

    faltando = set(ACTION_TABLE) - DATASET_ACTIONS

    assert not faltando, faltando

    # And exploration, which doesn't go through ACTION_TABLE.
    assert "swipe_up" in DATASET_ACTIONS
    assert "swipe_down" in DATASET_ACTIONS


def test_acao_desconhecida_nao_grava():
    """The filter allows dropping an action from the dataset without dropping it from the bot."""

    with Gravador() as g:

        alvo = deteccao("box")

        g.rec.on_action(
            action="acao_que_nao_existe",
            action_kind="click",
            detection=alvo,
            click=(140, 240),
            state="NORMAL",
            frame=frame(130),
            detections=[alvo],
            lag=0.3,
        )

        g.rec.observe(frame(131), [], time.monotonic() + 1.0, "NORMAL")

        assert g.registros(espera=0.5) == []


def test_build_e_plane_agora_gravam():

    with Gravador() as g:

        for acao, categoria, valor in (
            ("click", "build", 132),
            ("plane", "plane", 138),
        ):

            alvo = deteccao(categoria)

            g.rec.on_action(
                action=acao,
                action_kind="click",
                detection=alvo,
                click=(140, 240),
                state="NORMAL",
                frame=frame(valor),
                detections=[alvo],
                lag=0.3,
            )

            g.rec.observe(
                frame(valor + 1),
                [],
                time.monotonic() + 1.0,
                "NORMAL",
            )

        acoes = sorted(r["action"] for r in g.registros())

        assert acoes == ["click", "plane"], acoes


def test_bot_travado_para_de_gravar():
    """
    Replaces content-based dedup, MEASURED and discarded: the screen
    animates on its own, and the thumbnail gap between frames of the same screen
    (1.9-5.8) overlaps that of distinct screens (2.5-59.8) — any
    threshold would throw away good samples. What separates cleanly is the VERDICT:
    stuck session = 35/35 unchanged on the same action; productive = open_box
    5x in a row, all changed.
    """

    from core.config import DATASET_MAX_UNCHANGED_STREAK

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        # 10 rounds where the target NEVER leaves: stuck bot.
        for indice in range(10):

            g.rec.on_action(
                action="open_box",
                action_kind="click",
                detection=alvo,
                click=(348, 1544),
                state="NORMAL",
                frame=frame(10 + indice * 7),
                detections=[alvo],
                lag=0.3,
            )

            # The same target is still there.
            g.rec.observe(
                frame(11 + indice * 7),
                [deteccao("box", x=301, y=1501)],
                time.monotonic() + 1.0,
                "NORMAL",
            )

        regs = g.registros()

        assert len(regs) == DATASET_MAX_UNCHANGED_STREAK, (
            f"{len(regs)} gravadas, "
            f"{DATASET_MAX_UNCHANGED_STREAK} esperadas"
        )

        assert all(r["outcome"] == UNCHANGED for r in regs)


def test_repeticao_produtiva_grava_tudo():
    """What makes the cut safe: opening 5 boxes in a row is progress, not a stuck bot."""

    with Gravador() as g:

        for indice in range(5):

            alvo = deteccao("box", x=300 + indice * 20, y=1500)

            g.rec.on_action(
                action="open_box",
                action_kind="click",
                detection=alvo,
                click=(348, 1544),
                state="NORMAL",
                frame=frame(20 + indice * 9),
                detections=[alvo],
                lag=0.3,
            )

            # The target left: the box opened.
            g.rec.observe(
                frame(21 + indice * 9),
                [],
                time.monotonic() + 1.0,
                "NORMAL",
            )

        regs = g.registros()

        assert len(regs) == 5, f"perdeu amostra produtiva: {len(regs)}"

        assert all(r["outcome"] == CHANGED for r in regs)


def test_contagem_zera_ao_mudar_de_acao():
    """Stuck on one action must not silence the others."""

    from core.config import DATASET_MAX_UNCHANGED_STREAK

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        # Blows the count on open_box.
        for indice in range(6):

            g.rec.on_action(
                action="open_box",
                action_kind="click",
                detection=alvo,
                click=(348, 1544),
                state="NORMAL",
                frame=frame(30 + indice * 6),
                detections=[alvo],
                lag=0.3,
            )

            g.rec.observe(
                frame(31 + indice * 6),
                [deteccao("box", x=301, y=1501)],
                time.monotonic() + 1.0,
                "NORMAL",
            )

        antes = len(g.registros())

        # Another action, also unchanged: must record.
        outro = deteccao("food", x=700, y=900)

        g.rec.on_action(
            action="food",
            action_kind="click",
            detection=outro,
            click=(757, 940),
            state="NORMAL",
            frame=frame(180),
            detections=[outro],
            lag=0.3,
        )

        g.rec.observe(
            frame(181),
            [deteccao("food", x=701, y=901)],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        depois = g.registros()

        assert len(depois) == antes + 1, (antes, len(depois))

        assert depois[-1]["action"] == "food"


def test_telas_diferentes_nao_sao_descartadas():
    """No productive sample is lost to image similarity — there's no content filter anymore."""

    with Gravador() as g:

        alvo = deteccao("box")

        for valor in (10, 60, 110, 160):

            g.rec.on_action(
                action="open_box",
                action_kind="click",
                detection=alvo,
                click=(140, 240),
                state="NORMAL",
                frame=frame(valor),
                detections=[alvo],
                lag=0.3,
            )

            g.rec.observe(
                frame(valor + 1),
                [],
                time.monotonic() + 1.0,
                "NORMAL",
            )

        assert len(g.registros()) == 4, g.rec.stats()


def test_fila_cheia_descarta_sem_travar():
    """The guarantee that matters most: the bot NEVER waits on disk."""

    d = Path(tempfile.mkdtemp())

    try:

        rec = DatasetRecorder(directory=d)

        # Without start(): the thread doesn't consume, the queue fills.
        rec.running = True

        alvo = deteccao("box")

        inicio = time.monotonic()

        for i in range(200):

            rec.on_action(
                action="open_box",
                action_kind="click",
                detection=alvo,
                click=(140, 240),
                state="NORMAL",
                frame=frame(i % 200),
                detections=[alvo],
                lag=0.3,
            )

            rec.observe(
                frame(i % 200),
                [],
                time.monotonic() + 1.0,
                "NORMAL",
            )

        gasto = time.monotonic() - inicio

        rec.running = False

        assert rec.dropped > 0, "a fila deveria ter enchido"

        # Must not have blocked: 200 calls in far less
        # than a second.
        assert gasto < 2.0, gasto

    finally:

        shutil.rmtree(d, ignore_errors=True)


class AcoesFalsas:

    def __init__(self, device=(1080, 2400), frame_size=(1080, 2400)):

        from actions.manager import ActionManager

        self.real = ActionManager()
        self.real.device_size = device
        self.real.frame_size = frame_size

        self.executadas = []

    def execute(self, action, detection=None):

        self.executadas.append(action)

        return True

    def is_busy(self):
        return False

    def kind_of(self, action):
        return self.real.kind_of(action)

    def target_frame(self, action, detection):
        return self.real.target_frame(action, detection)

    def swipe(self, direction):

        self.executadas.append(f"swipe_{direction}")

        return True


def test_maquina_de_estados_grava_a_acao():
    """End to end: the machine acts and the sample lands on disk, with the click in image space."""

    with Gravador() as g:

        acoes = AcoesFalsas()

        machine = sm.StateMachine(acoes, g.rec)

        machine.action_cooldown = 0.0
        machine.action_settle = 0.0
        machine.last_action_time = 0.0

        alvo = deteccao("box", x=300, y=1500, w=96, h=88)

        agora = time.monotonic()

        machine.update([alvo], 0.1, frame(150), agora)

        assert acoes.executadas == ["open_box"], acoes.executadas

        # Later frame without the target.
        machine.update([], 0.1, frame(151), time.monotonic() + 1.0)

        regs = g.registros()

        assert len(regs) == 1, regs

        r = regs[0]

        assert r["action"] == "open_box"

        # Box center: 300 + 96//2, 1500 + 88//2
        assert (r["click_x"], r["click_y"]) == (348, 1544), r


def test_click_do_dismiss_usa_o_ponto_neutro():
    """Fixed-point action: the label is the neutral point in frame space, not the detection center."""

    with Gravador() as g:

        acoes = AcoesFalsas()

        machine = sm.StateMachine(acoes, g.rec)

        machine.action_cooldown = 0.0
        machine.action_settle = 0.0
        machine.last_action_time = 0.0

        alvo = deteccao("gray_max", x=700, y=1300)

        machine.update([alvo], 0.1, frame(160), time.monotonic())

        machine.update([], 0.1, frame(161), time.monotonic() + 1.0)

        r = g.registros()[0]

        assert r["action"] == "gray_max"
        assert r["action_kind"] == "dismiss"

        # Frame == reference, so the point is itself.
        assert (r["click_x"], r["click_y"]) == DISMISS_POINT, r


def test_ponto_neutro_escala_em_frame_menor():
    """If the frame comes shrunk, the label must scale with it — the recorded image is the frame."""

    with Gravador() as g:

        acoes = AcoesFalsas(
            device=(540, 1200),
            frame_size=(540, 1200),
        )

        machine = sm.StateMachine(acoes, g.rec)

        machine.action_cooldown = 0.0
        machine.action_settle = 0.0
        machine.last_action_time = 0.0

        alvo = deteccao("gray_max", x=100, y=200)

        machine.update(
            [alvo],
            0.1,
            frame(170, largura=540, altura=1200),
            time.monotonic(),
        )

        machine.update(
            [],
            0.1,
            frame(171, largura=540, altura=1200),
            time.monotonic() + 1.0,
        )

        r = g.registros()[0]

        esperado = (
            DISMISS_POINT[0] * 540 // REFERENCE_WIDTH,
            DISMISS_POINT[1] * 1200 // REFERENCE_HEIGHT,
        )

        assert (r["click_x"], r["click_y"]) == esperado, (
            (r["click_x"], r["click_y"]),
            esperado,
        )

        assert r["frame_width"] == 540
        assert r["frame_height"] == 1200


def test_rotulo_bate_com_o_toque_real():
    """
    The label (target_frame) and the tap (_dispatch) are computed on
    deliberately different paths — the tap doesn't go through the frame, to
    avoid stacking rounding. This test ensures both agree.
    """

    from actions.manager import ActionManager

    manager = ActionManager()

    manager.device_size = (1080, 2400)
    manager.frame_size = (1080, 2400)

    alvo = deteccao("box", x=317, y=1523, w=96, h=88)

    # Targeted action: box center.
    ponto = manager.target_frame("open_box", alvo)

    assert ponto == (317 + 48, 1523 + 44), ponto

    assert manager._from_frame(*ponto) == ponto

    # Fixed-point action: the neutral point.
    for acao in ("gray_max", "dismiss"):

        rotulo = manager.target_frame(acao, alvo)

        toque = manager._from_reference(*DISMISS_POINT)

        assert rotulo == toque, (acao, rotulo, toque)

    # Scroll has no point target.
    assert manager.target_frame("scroll_bottom", alvo) is None


def test_negativas_sao_gravadas():
    """A detector trained only on screens with a target learns there's always a target."""

    # Tiny interval: each observe() yields a negative.
    with Gravador(negativas=0.0001) as g:

        for valor in (10, 70, 130):

            g.rec.observe(
                frame(valor),
                [],
                time.monotonic(),
                "NORMAL",
            )

        regs = g.registros()

        assert len(regs) == 3, len(regs)

        for r in regs:

            assert r["action"] is None
            assert r["boxes"] == []
            assert r["click_x"] is None


def test_sem_recorder_a_maquina_funciona_igual():
    """The dataset is optional: recorder=None must change nothing about bot behavior."""

    acoes = AcoesFalsas()

    machine = sm.StateMachine(acoes)

    machine.action_cooldown = 0.0
    machine.action_settle = 0.0
    machine.last_action_time = 0.0

    machine.update([deteccao("box")], 0.1, frame(180), time.monotonic())

    assert acoes.executadas == ["open_box"], acoes.executadas


def test_swipe_de_exploracao_e_gravado():
    """
    Exploration doesn't go through _act, so it needed its own hook.
    The direction goes in the NAME because up/down are different labels for
    training, and that way no new column is needed in the index or the DB.
    """

    with Gravador() as g:

        acoes = AcoesFalsas()

        machine = sm.StateMachine(acoes, g.rec)

        machine.action_cooldown = 0.0
        machine.action_settle = 0.0
        machine.last_action_time = 0.0

        # Empty screen with exploration time expired.
        machine.explore_anchor = (
            time.monotonic()
            - machine.exploration_interval()
            - 1.0
        )

        machine.update([], 0.1, frame(190), time.monotonic())

        # Very different frame: the scroll worked.
        machine.update(
            [],
            0.1,
            frame(250),
            time.monotonic() + 1.0,
        )

        regs = [
            r for r in g.registros()
            if (r["action"] or "").startswith("swipe_")
        ]

        assert regs, [r["action"] for r in g.registros()]

        r = regs[0]

        assert r["action"] in ("swipe_up", "swipe_down"), r["action"]
        assert r["action_kind"] == "swipe", r["action_kind"]

        # Swipe is not a point tap.
        assert r["click_x"] is None
        assert r["click_y"] is None

        assert r["target_category"] is None


def test_acao_sem_alvo_julga_pela_mudanca_de_tela():
    """
    For swipe/scroll/dismiss there's no "target left the screen" — the
    question is whether the action had an effect. Detecting a LARGE change is
    reliable (scrolling moves the whole view).
    """

    import numpy as np

    with Gravador() as g:

        antes = frame(60)

        # COMPLETELY different screen.
        depois = np.full_like(antes, 220)

        g.rec.on_action(
            action="scroll_bottom",
            action_kind="scroll",
            detection=None,
            click=None,
            state="NORMAL",
            frame=antes,
            detections=[],
            lag=0.3,
        )

        g.rec.observe(depois, [], time.monotonic() + 1.0, "NORMAL")

        assert g.registros()[0]["outcome"] == CHANGED


def test_acao_sem_alvo_com_tela_parada_e_unchanged():

    with Gravador() as g:

        antes = frame(70)

        g.rec.on_action(
            action="scroll_bottom",
            action_kind="scroll",
            detection=None,
            click=None,
            state="NORMAL",
            frame=antes,
            detections=[],
            lag=0.3,
        )

        # Exactly the same frame: nothing changed.
        g.rec.observe(
            antes.copy(),
            [],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        assert g.registros()[0]["outcome"] == UNCHANGED


def test_registro_tem_as_chaves_que_o_banco_espera():
    """The store does a positional INSERT; a missing key would become a silent KeyError on the writer thread."""

    esperadas = {
        "id", "session", "created_at",
        "image", "image_sha256", "phash",
        "frame_width", "frame_height",
        "state", "action", "action_kind",
        "click_x", "click_y",
        "target_category", "target_template",
        "target_confidence",
        "detect_lag_ms",
        "outcome", "outcome_after_ms",
        "cycle_count", "boxes",
    }

    with Gravador() as g:

        alvo = deteccao("box")

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(140, 240),
            state="NORMAL",
            frame=frame(190),
            detections=[alvo],
            lag=0.3,
        )

        g.rec.observe(frame(191), [], time.monotonic() + 1.0, "NORMAL")

        r = g.registros()[0]

        faltando = esperadas - set(r)

        assert not faltando, faltando

        # And the boxes carry what the boxes table expects.
        caixa_esperada = {
            "category", "template", "confidence",
            "color_similarity", "x", "y",
            "width", "height", "acted",
        }

        assert not caixa_esperada - set(r["boxes"][0])


def test_sha256_confere_com_o_arquivo():
    """The hash detects duplicates across sessions and verifies integrity — a wrong one is worse than missing."""

    import hashlib

    with Gravador() as g:

        alvo = deteccao("box")

        g.rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(140, 240),
            state="NORMAL",
            frame=frame(200),
            detections=[alvo],
            lag=0.3,
        )

        g.rec.observe(frame(201), [], time.monotonic() + 1.0, "NORMAL")

        r = g.registros()[0]

        conteudo = (g.dir / r["image"]).read_bytes()

        assert (
            hashlib.sha256(conteudo).hexdigest()
            == r["image_sha256"]
        )


def test_grava_em_jpg():
    """
    Before this test no other exercised the JPG path (all on the
    PNG default). JPG is 4.3x smaller (0.49 MB vs 2.10 MB/frame) — over an
    hours-long session, 1.8 vs 7.6 GB/hour.
    """

    import dataset.recorder as mod

    original = mod.DATASET_IMAGE_FORMAT

    mod.DATASET_IMAGE_FORMAT = "jpg"

    try:

        with Gravador() as g:

            alvo = deteccao("box", x=300, y=1500)

            g.rec.on_action(
                action="open_box",
                action_kind="click",
                detection=alvo,
                click=(348, 1544),
                state="NORMAL",
                frame=frame(210),
                detections=[alvo],
                lag=0.3,
            )

            g.rec.observe(
                frame(211),
                [],
                time.monotonic() + 1.0,
                "NORMAL",
            )

            r = g.registros()[0]

            assert r["image"].endswith(".jpg"), r["image"]

            caminho = g.dir / r["image"]

            assert caminho.exists()

            # Re-reads: the right extension doesn't guarantee a valid file.
            import cv2

            lido = cv2.imread(str(caminho))

            assert lido is not None, "JPG gravado não abre"

            assert lido.shape[:2] == (2400, 1080), lido.shape

            # The hash must be of the JPG bytes, not something else.
            import hashlib

            assert (
                hashlib.sha256(caminho.read_bytes()).hexdigest()
                == r["image_sha256"]
            )

    finally:

        mod.DATASET_IMAGE_FORMAT = original


def test_jpg_e_bem_menor_que_png():
    """
    Uses a REAL fixture on purpose: with a synthetic frame the math FLIPS
    (flat color compresses to nearly nothing in PNG; JPG pays a base cost). The
    margin depends on the SOURCE: adb screencap (this fixture) is 1.8x smaller,
    decoded H.264 (the real case, recorded by the recorder) is 4.3x — so
    the claim here is just "smaller", not a factor.
    """

    import cv2
    import dataset.recorder as mod

    real = cv2.imread(str(ROOT / "tests" / "images" / "eatventure.png"))

    assert real is not None, "fixture eatventure.png não abriu"

    original = mod.DATASET_IMAGE_FORMAT

    tamanhos = {}

    try:

        for formato in ("png", "jpg"):

            mod.DATASET_IMAGE_FORMAT = formato

            with Gravador() as g:

                alvo = deteccao("box")

                g.rec.on_action(
                    action="open_box",
                    action_kind="click",
                    detection=alvo,
                    click=(140, 240),
                    state="NORMAL",
                    frame=real,
                    detections=[alvo],
                    lag=0.3,
                )

                g.rec.observe(
                    frame(221),
                    [],
                    time.monotonic() + 1.0,
                    "NORMAL",
                )

                r = g.registros()[0]

                tamanhos[formato] = (
                    g.dir / r["image"]
                ).stat().st_size

    finally:

        mod.DATASET_IMAGE_FORMAT = original

    assert tamanhos["jpg"] < tamanhos["png"], tamanhos


class CursorFalso:
    """
    Captures the SQL and tuples, no server. Under test: the COLUMN -> VALUE
    mapping — the INSERT is positional with 21 values, and a swapped
    order would put click_x into frame_width IN SILENCE (the DB accepts,
    both ints; the defect would only show in training).
    """

    def __init__(self, conexao):

        self.conexao = conexao

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):

        if self.conexao.falhar:

            raise RuntimeError("banco caiu")

        self.conexao.chamadas.append(("execute", sql, params))

    def executemany(self, sql, seq):

        if self.conexao.falhar:

            raise RuntimeError("banco caiu")

        self.conexao.chamadas.append(
            ("executemany", sql, list(seq))
        )


class ConexaoFalsa:

    def __init__(self):

        self.chamadas = []
        self.commits = 0
        self.rollbacks = 0
        self.fechada = False
        self.autocommit = False
        self.falhar = False

    def cursor(self):
        return CursorFalso(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.fechada = True

    # ---- test conveniences ----

    def por_tabela(self, tabela):
        """The calls touching this table."""

        return [
            c for c in self.chamadas
            if f"INTO public.{tabela}" in c[1]
        ]

    def colunas(self, sql):
        """INSERT column names, in the order they appear."""

        import re

        dentro = re.search(r"\(([^)]*)\)\s*VALUES", sql, re.S)

        return [
            c.strip()
            for c in dentro.group(1).replace("\n", " ").split(",")
            if c.strip()
        ]

    def linhas(self, tabela):
        """[{column: value}] of rows inserted into the table — this is where the positional order is checked."""

        resultado = []

        for tipo, sql, params in self.por_tabela(tabela):

            nomes = self.colunas(sql)

            tuplas = params if tipo == "executemany" else [params]

            for valores in tuplas:

                assert len(valores) == len(nomes), (
                    tabela,
                    len(valores),
                    len(nomes),
                )

                resultado.append(dict(zip(nomes, valores)))

        return resultado


def store_falso(batch_size=20):

    from dataset.store import PostgresStore

    conexao = ConexaoFalsa()

    store = PostgresStore("dsn-falso", batch_size=batch_size)

    store.connection = conexao

    return store, conexao


def registro_exemplo(indice=0, sessao="sessao-1"):

    return {
        "id": f"amostra-{indice}",
        "session": sessao,
        "created_at": "2026-08-21T15:15:32.481+00:00",
        "image": f"images/2026-08-21/amostra-{indice}.jpg",
        "image_sha256": "9f2a" + str(indice),
        "phash": "0110100",
        "frame_width": 1080,
        "frame_height": 2400,
        "state": "NORMAL",
        "action": "open_box",
        "action_kind": "click",
        "click_x": 348,
        "click_y": 1544,
        "target_category": "box",
        "target_template": "/default/box/item_003.png",
        "target_confidence": 0.9712,
        "detect_lag_ms": 420,
        "outcome": "changed",
        "outcome_after_ms": 910,
        "cycle_count": 3,
        "boxes": [
            {
                "category": "box",
                "template": "item_003.png",
                "confidence": 0.9712,
                "color_similarity": 0.9931,
                "x": 300,
                "y": 1500,
                "width": 96,
                "height": 88,
                "acted": True,
            },
            {
                "category": "food",
                "template": "item_017.png",
                "confidence": 0.9410,
                "color_similarity": 0.9802,
                "x": 700,
                "y": 900,
                "width": 115,
                "height": 96,
                "acted": False,
            },
        ],
    }


def test_cada_coluna_recebe_o_campo_certo():
    """
    The test justifying this whole file.

    Positional INSERT with 21 values: if the tuple order doesn't
    match the columns, data goes in swapped and the
    DB accepts it, because the types are compatible.
    """

    store, conexao = store_falso(batch_size=1)

    registro = registro_exemplo()

    store.insert(registro)

    linhas = conexao.linhas("dataset_sample")

    assert len(linhas) == 1, linhas

    linha = linhas[0]

    esperado = {
        "id": registro["id"],
        "session_id": registro["session"],
        "created_at": registro["created_at"],
        "image": registro["image"],
        "image_sha256": registro["image_sha256"],
        "phash": registro["phash"],
        "frame_width": 1080,
        "frame_height": 2400,
        "state": "NORMAL",
        "action": "open_box",
        "action_kind": "click",
        "click_x": 348,
        "click_y": 1544,
        "target_category": "box",
        "target_template": "/default/box/item_003.png",
        "target_confidence": 0.9712,
        "detect_lag_ms": 420,
        "outcome": "changed",
        "outcome_after_ms": 910,
        "cycle_count": 3,

        # Denormalized: len(boxes).
        "box_count": 2,
    }

    assert linha == esperado, {
        chave: (linha.get(chave), valor)
        for chave, valor in esperado.items()
        if linha.get(chave) != valor
    }


def test_caixas_vao_com_a_geometria_certa():

    store, conexao = store_falso(batch_size=1)

    registro = registro_exemplo()

    store.insert(registro)

    linhas = conexao.linhas("dataset_box")

    assert len(linhas) == 2, linhas

    agiu = [linha for linha in linhas if linha["acted"]]

    assert len(agiu) == 1, linhas

    assert agiu[0] == {
        "sample_id": "amostra-0",
        "category": "box",
        "template": "item_003.png",
        "confidence": 0.9712,
        "color_similarity": 0.9931,
        "x": 300,
        "y": 1500,
        "width": 96,
        "height": 88,
        "acted": True,
    }, agiu[0]


def test_sessao_inserida_uma_vez_so():
    """Without this, each batch would try to register the session again — a wasted DB trip per batch, for hours."""

    store, conexao = store_falso(batch_size=1)

    for indice in range(4):

        store.insert(registro_exemplo(indice))

    sessoes = conexao.por_tabela("dataset_session")

    assert len(sessoes) == 1, sessoes

    # And two distinct sessions yield two records.
    store.insert(registro_exemplo(9, sessao="sessao-2"))

    assert len(conexao.por_tabela("dataset_session")) == 2


def test_lote_espera_o_batch_size():
    """One DB trip per sample would put network latency on the thread that also encodes the image."""

    store, conexao = store_falso(batch_size=3)

    store.insert(registro_exemplo(0))
    store.insert(registro_exemplo(1))

    assert conexao.chamadas == [], "foi ao banco antes da hora"

    store.insert(registro_exemplo(2))

    assert conexao.linhas("dataset_sample"), "não gravou no lote"

    assert conexao.commits == 1, conexao.commits


def test_close_grava_o_resto():
    """The last batch almost never fills batch_size; without a flush on close, it would be lost from the DB."""

    store, conexao = store_falso(batch_size=100)

    store.insert(registro_exemplo(0))
    store.insert(registro_exemplo(1))

    assert conexao.chamadas == []

    store.close()

    assert len(conexao.linhas("dataset_sample")) == 2

    assert conexao.fechada


def test_falha_faz_rollback_e_propaga():
    """
    Without rollback, the connection stays unusable for every following batch
    (Postgres aborts the transaction). The exception must PROPAGATE: the recorder
    relies on it to log and keep writing to file.
    """

    store, conexao = store_falso(batch_size=1)

    conexao.falhar = True

    try:

        store.insert(registro_exemplo())

    except RuntimeError:

        assert conexao.rollbacks == 1, conexao.rollbacks

        assert conexao.commits == 0

        return

    raise AssertionError("deveria ter propagado")


def test_recorder_sobrevive_a_falha_do_banco():
    """End to end: broken DB, file written."""

    class StoreQuebrado:

        def __init__(self):
            self.tentativas = 0

        def insert(self, registro):
            self.tentativas += 1
            raise RuntimeError("banco caiu")

        def close(self):
            pass

    import shutil
    import tempfile

    from dataset.recorder import DatasetRecorder

    pasta = Path(tempfile.mkdtemp())

    quebrado = StoreQuebrado()

    try:

        rec = DatasetRecorder(directory=pasta, store=quebrado)

        rec.start()

        alvo = deteccao("box")

        rec.on_action(
            action="open_box",
            action_kind="click",
            detection=alvo,
            click=(140, 240),
            state="NORMAL",
            frame=frame(230),
            detections=[alvo],
            lag=0.3,
        )

        rec.observe(
            frame(231),
            [],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        rec.stop()

        indice = pasta / "samples.jsonl"

        assert indice.exists(), "o arquivo é o que não se recupera"

        assert len(
            indice.read_text(encoding="utf-8").splitlines()
        ) == 1

        assert quebrado.tentativas == 1

    finally:

        shutil.rmtree(pasta, ignore_errors=True)


def test_importar_jsonl():
    """The recommended path: record to file now, load into the DB later, including old sessions."""

    import shutil
    import tempfile

    import dataset.store as mod

    pasta = Path(tempfile.mkdtemp())

    conexao = ConexaoFalsa()

    caminho = pasta / "samples.jsonl"

    caminho.write_text(
        "\n".join(
            json.dumps(registro_exemplo(i))
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )

    original = mod.PostgresStore.connect

    def conectar_falso(self):

        self.connection = conexao

        return self

    mod.PostgresStore.connect = conectar_falso

    try:

        total = mod.importar_jsonl(
            caminho,
            "dsn-falso",
            batch_size=2,
        )

    finally:

        mod.PostgresStore.connect = original

        shutil.rmtree(pasta, ignore_errors=True)

    assert total == 5, total

    assert len(conexao.linhas("dataset_sample")) == 5

    # 2 boxes per sample.
    assert len(conexao.linhas("dataset_box")) == 10

    assert conexao.fechada


def test_schema_configuravel_aparece_no_sql():
    """Anyone on their own schema needs it to reach the SQL."""

    from dataset.store import PostgresStore

    conexao = ConexaoFalsa()

    store = PostgresStore(
        "dsn-falso",
        batch_size=1,
        schema="eatventure",
    )

    store.connection = conexao

    store.insert(registro_exemplo())

    assert any(
        "eatventure.dataset_sample" in chamada[1]
        for chamada in conexao.chamadas
    ), [c[1][:60] for c in conexao.chamadas]


def test_ddl_cobre_as_colunas_do_insert():
    """
    The DDL lives in docs/schema.sql, out of the interpreter's reach —
    without this, code and script can diverge and the symptom is the FIRST
    run with a DB failing in the user's face. Checks against the
    .sql, not the .md, because the .sql is what EXECUTES.
    """

    import re

    doc = (
        ROOT / "docs" / "schema.sql"
    ).read_text(encoding="utf-8")

    store, conexao = store_falso(batch_size=1)

    store.insert(registro_exemplo())

    for tabela in ("dataset_sample", "dataset_box"):

        usadas = set(conexao.linhas(tabela)[0])

        bloco = re.search(
            rf"CREATE TABLE IF NOT EXISTS {tabela} \((.*?)\n\);",
            doc,
            re.S,
        )

        assert bloco, f"{tabela} não tem DDL em docs/schema.sql"

        declaradas = set()

        for linha in bloco.group(1).splitlines():

            linha = linha.strip()

            if not linha or linha.startswith("--"):
                continue

            nome = linha.split()[0]

            if nome.upper() in ("REFERENCES", "ON"):
                continue

            declaradas.add(nome)

        faltando = usadas - declaradas

        assert not faltando, (tabela, faltando)


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
