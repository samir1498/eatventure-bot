"""
Testes da gravação do dataset. Grava em pasta temporária, nunca em
DATASET_DIR. Um rótulo errado aqui não dá erro nenhum — treina o modelo
a clicar no lugar errado, e o defeito só aparece depois de horas de GPU.
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
    """Frame com gradiente, para o hash de similaridade distinguir um do outro."""

    img = np.full((altura, largura, 3), valor, np.uint8)

    # Posição dependente do valor: dois frames diferentes não colidem no hash.
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
    Recorder num diretório temporário, já iniciado. Negativas DESLIGADAS
    por padrão (intervalo 0): uma a mais mudaria o total nos testes de
    contagem.
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
        """Lê o índice, esperando a thread drenar a fila."""

        indice = self.dir / "samples.jsonl"

        limite = time.monotonic() + espera

        while time.monotonic() < limite:

            if self.rec.queue.empty() and indice.exists():

                # Uma folga para a escrita terminar.
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

        # Frame posterior sem o alvo: a ação funcionou.
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

        # A imagem existe de verdade, no caminho anotado.
        caminho = g.dir / r["image"]

        assert caminho.exists(), r["image"]

        assert caminho.stat().st_size > 0

        # O caminho é RELATIVO: mover a pasta não invalida o
        # índice.
        assert not Path(r["image"]).is_absolute()


def test_grava_todas_as_caixas_do_frame():
    """O bot agiu em uma; o detector viu três — as outras são rótulo grátis."""

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

        # Exatamente uma marcada como a que virou ação.
        agidas = [b for b in r["boxes"] if b["acted"]]

        assert len(agidas) == 1, agidas

        assert agidas[0]["category"] == "box"
        assert agidas[0]["x"] == 300
        assert agidas[0]["y"] == 1500


def test_geometria_da_caixa_e_preservada():
    """x/y/width/height são o rótulo de detecção: errar aqui treina o modelo a enquadrar errado."""

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
    """O rótulo que quebra o teto do behavior cloning: permite treinar só nas ações que funcionaram."""

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

        # Mesmo alvo, mesmo lugar: não pegou.
        g.rec.observe(
            frame(91),
            [deteccao("box", x=302, y=1503)],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        assert g.registros()[0]["outcome"] == UNCHANGED


def test_outra_instancia_da_categoria_nao_conta_como_unchanged():
    """Comparar só a categoria daria falso "não pegou" quando há outra comida em outro canto."""

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

        # Comida sim, mas do outro lado da tela.
        g.rec.observe(
            frame(101),
            [deteccao("food", x=900, y=2000)],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        assert g.registros()[0]["outcome"] == CHANGED


def test_frame_anterior_a_acao_nao_julga():
    """Julgar por um frame de antes da ação produziria "unchanged" para toda ação que funcionou."""

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

        # Frame de ANTES: nada é gravado ainda.
        g.rec.observe(frame(111), [], agora - 1.0, "NORMAL")

        assert g.rec.pending is not None

        assert g.registros(espera=0.3) == []

        # Frame de depois: agora sim.
        g.rec.observe(frame(112), [], time.monotonic() + 1.0, "NORMAL")

        assert len(g.registros()) == 1


def test_acao_sem_veredito_e_gravada_no_stop():
    """Sessão terminando com ação pendente: a imagem e as caixas valem mesmo sem o resultado."""

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
    Ação que o bot faz e o dataset ignora é buraco que só aparece quando
    o modelo não sabe fazer aquilo. Inclui `click` (build) e `plane`,
    que ficavam de fora.
    """

    from actions.manager import ACTION_TABLE
    from core.config import DATASET_ACTIONS

    faltando = set(ACTION_TABLE) - DATASET_ACTIONS

    assert not faltando, faltando

    # E a exploração, que não passa por ACTION_TABLE.
    assert "swipe_up" in DATASET_ACTIONS
    assert "swipe_down" in DATASET_ACTIONS


def test_acao_desconhecida_nao_grava():
    """O filtro permite tirar uma ação do dataset sem tirar do bot."""

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
    Substitui a deduplicação por conteúdo, MEDIDA e descartada: a tela
    anima sozinha, e a diferença de miniatura entre frames da mesma tela
    (1.9-5.8) se sobrepõe à de telas distintas (2.5-59.8) — qualquer
    limite jogaria fora amostra boa. O que separa limpo é o RESULTADO:
    sessão travada = 35/35 unchanged na mesma ação; produtiva = open_box
    5x seguidas, todas changed.
    """

    from core.config import DATASET_MAX_UNCHANGED_STREAK

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        # 10 rodadas em que o alvo NUNCA sai: bot preso.
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

            # O mesmo alvo continua lá.
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
    """O que torna o corte seguro: abrir 5 caixas seguidas é progresso, não bot travado."""

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

            # O alvo saiu: a caixa abriu.
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
    """Preso numa ação não pode calar as outras."""

    from core.config import DATASET_MAX_UNCHANGED_STREAK

    with Gravador() as g:

        alvo = deteccao("box", x=300, y=1500)

        # Estoura a contagem em open_box.
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

        # Outra ação, também unchanged: tem de gravar.
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
    """Nenhuma amostra produtiva é perdida por semelhança de imagem — não há mais filtro por conteúdo."""

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
    """A garantia que mais importa: o bot NUNCA espera o disco."""

    d = Path(tempfile.mkdtemp())

    try:

        rec = DatasetRecorder(directory=d)

        # Sem start(): a thread não consome, a fila enche.
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

        # Não pode ter bloqueado: 200 chamadas em muito menos
        # que um segundo.
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
    """Fim a fim: a máquina age e a amostra aparece no disco, com o clique no espaço da imagem."""

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

        # Frame posterior sem o alvo.
        machine.update([], 0.1, frame(151), time.monotonic() + 1.0)

        regs = g.registros()

        assert len(regs) == 1, regs

        r = regs[0]

        assert r["action"] == "open_box"

        # Centro da caixa: 300 + 96//2, 1500 + 88//2
        assert (r["click_x"], r["click_y"]) == (348, 1544), r


def test_click_do_dismiss_usa_o_ponto_neutro():
    """Ação de ponto fixo: o rótulo é o ponto neutro no espaço do frame, não o centro da detecção."""

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

        # Frame == referência, então o ponto é o próprio.
        assert (r["click_x"], r["click_y"]) == DISMISS_POINT, r


def test_ponto_neutro_escala_em_frame_menor():
    """Se o frame vier reduzido, o rótulo tem de escalar com ele — a imagem gravada é o frame."""

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
    O rótulo (target_frame) e o toque (_dispatch) são calculados por
    caminhos diferentes de propósito — o toque não passa pelo frame, para
    não acumular arredondamento. Este teste garante que os dois concordam.
    """

    from actions.manager import ActionManager

    manager = ActionManager()

    manager.device_size = (1080, 2400)
    manager.frame_size = (1080, 2400)

    alvo = deteccao("box", x=317, y=1523, w=96, h=88)

    # Ação com alvo: centro da caixa.
    ponto = manager.target_frame("open_box", alvo)

    assert ponto == (317 + 48, 1523 + 44), ponto

    assert manager._from_frame(*ponto) == ponto

    # Ação de ponto fixo: o ponto neutro.
    for acao in ("gray_max", "dismiss"):

        rotulo = manager.target_frame(acao, alvo)

        toque = manager._from_reference(*DISMISS_POINT)

        assert rotulo == toque, (acao, rotulo, toque)

    # Scroll não tem alvo pontual.
    assert manager.target_frame("scroll_bottom", alvo) is None


def test_negativas_sao_gravadas():
    """Um detector treinado só em telas com alvo aprende que sempre existe um alvo."""

    # Intervalo minúsculo: cada observe() gera uma negativa.
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
    """O dataset é opcional: recorder=None não pode mudar nada no comportamento do bot."""

    acoes = AcoesFalsas()

    machine = sm.StateMachine(acoes)

    machine.action_cooldown = 0.0
    machine.action_settle = 0.0
    machine.last_action_time = 0.0

    machine.update([deteccao("open_store")], 0.1, frame(180), time.monotonic())

    assert acoes.executadas == ["open_store_click"], acoes.executadas


def test_swipe_de_exploracao_e_gravado():
    """
    A exploração não passa por _act, então precisava de gancho próprio.
    A direção vai no NOME porque subir/descer são rótulos diferentes para
    o treino, e assim não precisa de coluna nova no índice nem no banco.
    """

    with Gravador() as g:

        acoes = AcoesFalsas()

        machine = sm.StateMachine(acoes, g.rec)

        machine.action_cooldown = 0.0
        machine.action_settle = 0.0
        machine.last_action_time = 0.0

        # Tela vazia e tempo de exploração vencido.
        machine.explore_anchor = (
            time.monotonic()
            - machine.exploration_interval()
            - 1.0
        )

        machine.update([], 0.1, frame(190), time.monotonic())

        # Frame bem diferente: a rolagem funcionou.
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

        # Swipe não é toque pontual.
        assert r["click_x"] is None
        assert r["click_y"] is None

        assert r["target_category"] is None


def test_acao_sem_alvo_julga_pela_mudanca_de_tela():
    """
    Para swipe/scroll/dismiss não existe "o alvo saiu da tela" — a
    pergunta é se a ação surtiu efeito. Detectar mudança GRANDE é
    confiável (rolagem move a vista inteira).
    """

    import numpy as np

    with Gravador() as g:

        antes = frame(60)

        # Tela COMPLETAMENTE diferente.
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

        # Exatamente o mesmo frame: nada mudou.
        g.rec.observe(
            antes.copy(),
            [],
            time.monotonic() + 1.0,
            "NORMAL",
        )

        assert g.registros()[0]["outcome"] == UNCHANGED


def test_registro_tem_as_chaves_que_o_banco_espera():
    """O store faz INSERT posicional; chave faltando viraria KeyError silencioso na thread de gravação."""

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

        # E as caixas têm o que a tabela de caixas espera.
        caixa_esperada = {
            "category", "template", "confidence",
            "color_similarity", "x", "y",
            "width", "height", "acted",
        }

        assert not caixa_esperada - set(r["boxes"][0])


def test_sha256_confere_com_o_arquivo():
    """O hash detecta duplicata entre sessões e confere integridade — errado é pior que ausente."""

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
    Antes deste teste nenhum outro exercitava o caminho JPG (todos no
    padrão PNG). JPG é 4.3x menor (0.49 MB vs 2.10 MB/frame) — numa sessão
    de horas, 1.8 vs 7.6 GB/hora.
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

            # Relê: extensão certa não garante arquivo válido.
            import cv2

            lido = cv2.imread(str(caminho))

            assert lido is not None, "JPG gravado não abre"

            assert lido.shape[:2] == (2400, 1080), lido.shape

            # O hash tem de ser dos bytes do JPG, não de outra coisa.
            import hashlib

            assert (
                hashlib.sha256(caminho.read_bytes()).hexdigest()
                == r["image_sha256"]
            )

    finally:

        mod.DATASET_IMAGE_FORMAT = original


def test_jpg_e_bem_menor_que_png():
    """
    Usa FIXTURE REAL de propósito: com frame sintético a conta se INVERTE
    (cor lisa comprime a quase nada no PNG; JPG paga custo de base). A
    margem depende da ORIGEM: adb screencap (este fixture) é 1.8x menor,
    H.264 decodificado (o caso real, gravado pelo recorder) é 4.3x — por
    isso a afirmação aqui é só "menor", não um fator.
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
    Captura os SQL e as tuplas, sem servidor. Sob teste: a correspondência
    COLUNA -> VALOR — o INSERT é posicional com 21 valores, e uma troca de
    ordem faria click_x entrar em frame_width EM SILÊNCIO (o banco aceita,
    ambos inteiros; o defeito só apareceria no treino).
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

    # ---- conveniências para os testes ----

    def por_tabela(self, tabela):
        """As chamadas que mexem nesta tabela."""

        return [
            c for c in self.chamadas
            if f"INTO public.{tabela}" in c[1]
        ]

    def colunas(self, sql):
        """Nomes das colunas do INSERT, na ordem em que aparecem."""

        import re

        dentro = re.search(r"\(([^)]*)\)\s*VALUES", sql, re.S)

        return [
            c.strip()
            for c in dentro.group(1).replace("\n", " ").split(",")
            if c.strip()
        ]

    def linhas(self, tabela):
        """[{coluna: valor}] das linhas inseridas na tabela — é aqui que a ordem posicional é conferida."""

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
    O teste que justifica todo este arquivo.

    INSERT posicional com 21 valores: se a ordem da tupla não
    corresponder à das colunas, os dados entram trocados e o
    banco aceita, porque os tipos são compatíveis.
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

        # Desnormalizado: len(boxes).
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
    """Sem isto, cada lote tentaria registrar a sessão de novo — ida ao banco à toa por lote, por horas."""

    store, conexao = store_falso(batch_size=1)

    for indice in range(4):

        store.insert(registro_exemplo(indice))

    sessoes = conexao.por_tabela("dataset_session")

    assert len(sessoes) == 1, sessoes

    # E duas sessões distintas geram dois registros.
    store.insert(registro_exemplo(9, sessao="sessao-2"))

    assert len(conexao.por_tabela("dataset_session")) == 2


def test_lote_espera_o_batch_size():
    """Uma ida ao banco por amostra colocaria latência de rede na thread que também codifica a imagem."""

    store, conexao = store_falso(batch_size=3)

    store.insert(registro_exemplo(0))
    store.insert(registro_exemplo(1))

    assert conexao.chamadas == [], "foi ao banco antes da hora"

    store.insert(registro_exemplo(2))

    assert conexao.linhas("dataset_sample"), "não gravou no lote"

    assert conexao.commits == 1, conexao.commits


def test_close_grava_o_resto():
    """O último lote quase nunca fecha o batch_size; sem flush no close, ele se perderia no banco."""

    store, conexao = store_falso(batch_size=100)

    store.insert(registro_exemplo(0))
    store.insert(registro_exemplo(1))

    assert conexao.chamadas == []

    store.close()

    assert len(conexao.linhas("dataset_sample")) == 2

    assert conexao.fechada


def test_falha_faz_rollback_e_propaga():
    """
    Sem rollback, a conexão fica inutilizável para todo lote seguinte
    (Postgres aborta a transação). A exceção tem de PROPAGAR: o recorder
    conta com isso para logar e seguir gravando em arquivo.
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
    """Fim a fim: banco quebrado, arquivo gravado."""

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
    """O caminho recomendado: gravar em arquivo agora, carregar no banco depois, inclusive sessões antigas."""

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

    # 2 caixas por amostra.
    assert len(conexao.linhas("dataset_box")) == 10

    assert conexao.fechada


def test_schema_configuravel_aparece_no_sql():
    """Quem usa schema próprio precisa que ele chegue no SQL."""

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
    O DDL vive em docs/schema.sql, fora do alcance do interpretador —
    sem isto, código e script podem divergir e o sintoma é a PRIMEIRA
    execução com banco falhando na cara do usuário. Confere contra o
    .sql, não o .md, porque é o .sql que se EXECUTA.
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
