"""
Training and AI-bot tests. They live here and not in IA/, because IA/ runs
real training and inference. Nothing here trains a model or touches a
device — the target is silent damage: train/inference features
diverging, leaky splits, or wrong coordinate mapping.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

for candidato in (
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "IA"),
):
    if candidato not in sys.path:
        sys.path.insert(0, candidato)

import features as feat                            # noqa: E402
import dataset_io as dio                           # noqa: E402


def frame_com_alvo(alvo=(300, 1500, 96, 88), cor=(40, 200, 255)):
    """Device-sized frame, with a colored block at the box position."""

    imagem = np.full((2400, 1080, 3), 30, np.uint8)

    x, y, w, h = alvo

    imagem[y:y + h, x:x + w] = cor

    return imagem


def registro(
    identificador="a",
    acao="open_box",
    estado="NORMAL",
    caixas=None,
    phash="0101",
    sessao="s1",
    outcome="changed",
):

    if caixas is None:

        caixas = [
            {
                "category": "box",
                "template": "item_001.png",
                "confidence": 0.97,
                "color_similarity": 0.99,
                "x": 300,
                "y": 1500,
                "width": 96,
                "height": 88,
                "acted": True,
            }
        ]

    return {
        "id": identificador,
        "session": sessao,
        "image": f"images/{identificador}.jpg",
        "phash": phash,
        "frame_width": 1080,
        "frame_height": 2400,
        "state": estado,
        "action": acao,
        "action_kind": "click",
        "click_x": 348,
        "click_y": 1544,
        "target_category": "box",
        "outcome": outcome,
        "boxes": caixas,
    }


def test_recorte_pega_o_objeto_e_nao_a_tela():
    """
    Previous-version defect: shrinking 1080x2400 to 32x32 turned the
    95x94 px target into 2.8x1.3 px. The crop must contain the real
    object, dominating the frame.
    """

    alvo = (300, 1500, 96, 88)

    imagem = frame_com_alvo(alvo, cor=(255, 255, 255))

    recorte = feat.crop_box(imagem, {
        "x": alvo[0], "y": alvo[1],
        "width": alvo[2], "height": alvo[3],
    })

    assert recorte is not None

    # With a 25% margin, the object covers about (1/1.5)^2 = 44%
    # of the crop. Far from 3 pixels on a whole screen.
    brancos = int((recorte > 200).all(axis=2).sum())

    fracao = brancos / (recorte.shape[0] * recorte.shape[1])

    assert 0.3 < fracao < 0.8, fracao

    # And for comparison: on the whole screen shrunk to 32x32, the same
    # target barely exists.
    inteira = feat.screen_features(imagem)

    assert inteira is not None


def test_features_tem_tamanho_declarado():
    """If FEATURE_SIZE lies, the check against `n_features_in_` passes and the model gets the wrong input."""

    imagem = frame_com_alvo()

    vetor = feat.box_features(imagem, {
        "x": 300, "y": 1500, "width": 96, "height": 88,
    })

    assert vetor is not None

    assert len(vetor) == feat.FEATURE_SIZE, (
        len(vetor),
        feat.FEATURE_SIZE,
    )

    assert vetor.dtype == np.float32


def test_features_sao_deterministicas():
    """Without this, training learns one thing and inference sees another."""

    imagem = frame_com_alvo()

    caixa = {"x": 300, "y": 1500, "width": 96, "height": 88}

    a = feat.box_features(imagem, caixa)
    b = feat.box_features(imagem, caixa)

    assert np.array_equal(a, b)


def test_features_distinguem_cores():
    """The HSV histogram exists because the game encodes by color."""

    caixa = {"x": 300, "y": 1500, "width": 96, "height": 88}

    azul = feat.box_features(
        frame_com_alvo(cor=(255, 60, 60)), caixa
    )

    verde = feat.box_features(
        frame_com_alvo(cor=(60, 255, 60)), caixa
    )

    distancia = float(np.abs(azul - verde).mean())

    assert distancia > 0.05, distancia


def test_recorte_fora_da_imagem_nao_estoura():
    """A partially out-of-frame box really happens (object on the edge)."""

    imagem = frame_com_alvo()

    for caixa in (
        {"x": -50, "y": -50, "width": 96, "height": 88},
        {"x": 1050, "y": 2380, "width": 96, "height": 88},
        {"x": 0, "y": 0, "width": 1, "height": 1},
    ):

        vetor = feat.box_features(imagem, caixa)

        # Either returns a correctly-sized vector, or None. What it
        # must not do is blow up or return the wrong size.
        if vetor is not None:
            assert len(vetor) == feat.FEATURE_SIZE, caixa


def test_recorte_degenerado_devolve_none():

    imagem = frame_com_alvo()

    assert feat.box_features(imagem, {
        "x": 500, "y": 500, "width": 0, "height": 0,
    }) is None


def test_split_nao_reparte_o_mesmo_grupo():
    """
    Error that invalidated the previous accuracy: 56% of samples have a
    repeated phash, and the random split put the same frame in train and test.
    """

    registros = [
        registro(f"r{i}", phash=f"h{i % 10}")
        for i in range(100)
    ]

    treino, teste, info = dio.split_groups(
        registros,
        train_ratio=0.7,
        mode="phash",
    )

    grupos_treino = {r["phash"] for r in treino}
    grupos_teste = {r["phash"] for r in teste}

    assert not (grupos_treino & grupos_teste), (
        grupos_treino & grupos_teste
    )

    assert len(treino) + len(teste) == 100

    assert info["grupos"] == 10


def test_split_por_sessao_separa_sessoes():
    """The toughest test: train on one session and evaluate on another."""

    registros = [
        registro(f"r{i}", sessao=f"s{i % 4}", phash=f"h{i}")
        for i in range(40)
    ]

    treino, teste, _ = dio.split_groups(
        registros,
        train_ratio=0.75,
        mode="session",
    )

    assert not (
        {r["session"] for r in treino}
        & {r["session"] for r in teste}
    )


def test_split_e_reproduzivel():
    """Same seed, same split — otherwise comparing two runs means nothing."""

    registros = [
        registro(f"r{i}", phash=f"h{i % 12}")
        for i in range(60)
    ]

    a = dio.split_groups(registros, seed=7)[0]
    b = dio.split_groups(registros, seed=7)[0]

    assert [r["id"] for r in a] == [r["id"] for r in b]

    c = dio.split_groups(registros, seed=8)[0]

    assert [r["id"] for r in a] != [r["id"] for r in c]


def test_split_nunca_deixa_teste_vazio():
    """With few groups, rounding up would send everything to train and empty the test set."""

    registros = [registro(f"r{i}", phash="unico") for i in range(5)]
    registros += [registro("outro", phash="segundo")]

    treino, teste, _ = dio.split_groups(
        registros,
        train_ratio=0.95,
    )

    assert treino and teste


def test_baseline_da_maioria():

    rotulos = ["a"] * 7 + ["b"] * 3

    fracao, classe = dio.majority_baseline(rotulos)

    assert classe == "a"

    assert abs(fracao - 0.7) < 1e-9


def test_baseline_do_estado():
    """The baseline that exposed the problem: 46.2% on the real dataset vs under 30% for the model."""

    registros = (
        [registro(f"a{i}", acao="upgrade_item", estado="UPGRADE")
         for i in range(8)]
        + [registro(f"b{i}", acao="close", estado="UPGRADE")
           for i in range(2)]
        + [registro(f"c{i}", acao="open_box", estado="NORMAL")
           for i in range(10)]
    )

    # UPGRADE: guesses upgrade_item, gets 8/10 right.
    # NORMAL: guesses open_box, gets 10/10 right.
    assert abs(dio.state_baseline(registros) - 0.9) < 1e-9


def test_teto_por_estado_e_categorias():
    """
    Justifies the architecture: if state + categories solves almost all
    action choice, the model only needs to learn DETECTION — the action
    comes from the priority table that already exists.
    """

    def com_categorias(identificador, cats, acao, estado="NORMAL"):

        return registro(
            identificador,
            acao=acao,
            estado=estado,
            caixas=[
                {
                    "category": cat,
                    "x": 10, "y": 10,
                    "width": 50, "height": 50,
                    "acted": indice == 0,
                }
                for indice, cat in enumerate(cats)
            ],
        )

    registros = [
        com_categorias("a", ["box"], "open_box"),
        com_categorias("b", ["box"], "open_box"),
        com_categorias("c", ["food"], "food"),
        com_categorias("d", ["food"], "food"),
    ]

    assert dio.ceiling_state_categories(registros) == 1.0

    # Same input with a different label = ceiling below 100%,
    # and that's exactly the irreducible noise limiting the model.
    registros.append(com_categorias("e", ["box"], "upgrade"))

    assert dio.ceiling_state_categories(registros) < 1.0


def test_filtro_por_outcome():
    """Training only on `changed` stops the model from inheriting the template matcher's mistakes."""

    registros = [
        registro("a", outcome="changed"),
        registro("b", outcome="unchanged"),
        registro("c", outcome=None, acao=None),
    ]

    assert len(dio.filter_by_outcome(registros, None)) == 3

    apenas = dio.filter_by_outcome(registros, ["changed"])

    assert [r["id"] for r in apenas] == ["a"]

    # Negatives enter when explicitly requested: without
    # them the model learns a target always exists.
    com_negativa = dio.filter_by_outcome(
        registros,
        ["changed", "negative"],
    )

    assert sorted(r["id"] for r in com_negativa) == ["a", "c"]


def test_negativos_nao_caem_sobre_as_caixas():
    """A negative over an object teaches the opposite: that the object is background."""

    caixas = [
        {"category": "box", "x": 300, "y": 1500,
         "width": 96, "height": 88, "acted": True},
        {"category": "food", "x": 700, "y": 900,
         "width": 90, "height": 77, "acted": False},
    ]

    reg = registro("a", caixas=caixas)

    negativos = dio.sample_negative_boxes(
        reg,
        quantas=30,
        tamanhos=[(96, 88), (90, 77)],
        seed=1,
    )

    assert negativos, "não sorteou nenhum negativo"

    for negativo in negativos:

        candidata = (
            negativo["x"], negativo["y"],
            negativo["width"], negativo["height"],
        )

        for caixa in caixas:

            ocupada = (
                caixa["x"], caixa["y"],
                caixa["width"], caixa["height"],
            )

            assert dio.iou(candidata, ocupada) <= 0.1, (
                negativo,
                caixa,
            )


def test_negativos_ficam_dentro_do_frame():

    reg = registro("a")

    for negativo in dio.sample_negative_boxes(
        reg,
        quantas=40,
        tamanhos=[(96, 88), (300, 150)],
        seed=3,
    ):

        assert negativo["x"] >= 0
        assert negativo["y"] >= 0
        assert negativo["x"] + negativo["width"] <= 1080
        assert negativo["y"] + negativo["height"] <= 2400


def test_iou():

    assert dio.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    assert dio.iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0

    meio = dio.iou((0, 0, 10, 10), (5, 0, 10, 10))

    assert abs(meio - (50 / 150)) < 1e-9


# =========================================================
# INDEX READING
# =========================================================

def test_leitura_ignora_linha_invalida_e_imagem_ausente():
    """
    samples.jsonl is a thread append: if the session died halfway,
    the last line may be truncated. That must not take down training.
    """

    pasta = Path(tempfile.mkdtemp())

    try:

        (pasta / "images").mkdir()

        import cv2

        cv2.imwrite(
            str(pasta / "images" / "existe.jpg"),
            np.full((100, 100, 3), 128, np.uint8),
        )

        linhas = [
            json.dumps({
                "id": "ok",
                "image": "images/existe.jpg",
                "boxes": [],
            }),
            "{isso nao e json",
            json.dumps({
                "id": "sem_imagem",
                "image": "images/nao_existe.jpg",
            }),
            json.dumps({"id": "sem_campo"}),
            "",
        ]

        (pasta / "samples.jsonl").write_text(
            "\n".join(linhas) + "\n",
            encoding="utf-8",
        )

        registros, aviso = dio.read_index(pasta)

        assert [r["id"] for r in registros] == ["ok"]

        assert aviso["invalidas"] == 1

        assert aviso["sem_imagem"] == 2

        # The resolved path must really exist.
        assert registros[0]["_path"].exists()

    finally:

        shutil.rmtree(pasta, ignore_errors=True)


def test_indice_ausente_da_erro_com_instrucao():

    pasta = Path(tempfile.mkdtemp())

    try:

        dio.read_index(pasta)

    except FileNotFoundError as erro:

        assert "DATASET_SAVE" in str(erro), str(erro)

        return

    finally:

        shutil.rmtree(pasta, ignore_errors=True)

    raise AssertionError("deveria ter levantado")


# =========================================================
# LABELS
# =========================================================

def test_negativa_vira_classe_propria():
    """
    "doing nothing" is a decision. Dropping negatives
    teaches that there's always something to do.
    """

    assert dio.action_label(registro("a", acao=None)) == "negative"

    assert dio.action_label(registro("a", acao="open_box")) == "open_box"


def test_todas_as_caixas_viram_exemplo():
    """
    That's what yields 65,730 examples from 29,297 frames: the
    boxes that did NOT become an action are labels too.
    """

    caixas = [
        {"category": "box", "x": 1, "y": 1,
         "width": 10, "height": 10, "acted": True},
        {"category": "food", "x": 20, "y": 20,
         "width": 10, "height": 10, "acted": False},
        {"category": "upgrade", "x": 40, "y": 40,
         "width": 10, "height": 10, "acted": False},
    ]

    rotulos = dio.box_labels(registro("a", caixas=caixas))

    assert len(rotulos) == 3

    assert [cat for _, cat, _ in rotulos] == [
        "box", "food", "upgrade",
    ]

    assert [agiu for _, _, agiu in rotulos] == [True, False, False]


def test_caixa_degenerada_e_ignorada():

    caixas = [
        {"category": "box", "x": 1, "y": 1,
         "width": 0, "height": 10, "acted": True},
        {"category": "", "x": 1, "y": 1,
         "width": 10, "height": 10, "acted": False},
    ]

    assert dio.box_labels(registro("a", caixas=caixas)) == []


def test_proponente_acha_bloco_colorido():
    """
    Proposer is EXPERIMENTAL: replaces the 184 templates with region
    proposals. Only verifies the geometric contract; on a real screen it
    needs `--debug-proposals` — no synthetic test proves that.
    """

    import proposer

    imagem = np.full((2400, 1080, 3), 40, np.uint8)

    # Block well saturated, real-button sized.
    imagem[1500:1588, 300:396] = (255, 120, 20)

    candidatos = proposer.propose(imagem)

    assert candidatos, "não propôs nada"

    # Some candidate must cover the block.
    alvo = (300, 1500, 96, 88)

    melhor = max(
        dio.iou(c, alvo) for c in candidatos
    )

    assert melhor > 0.3, (melhor, candidatos[:5])


def test_proponente_respeita_limites_de_tamanho():

    import proposer

    imagem = np.full((2400, 1080, 3), 40, np.uint8)

    # Too small and too big: none may pass.
    imagem[100:110, 100:110] = (255, 120, 20)
    imagem[500:900, 0:1080] = (255, 120, 20)

    for x, y, w, h in proposer.propose(imagem):

        assert w >= proposer.MIN_SIDE
        assert h >= proposer.MIN_SIDE
        assert w <= proposer.MAX_WIDTH
        assert h <= proposer.MAX_HEIGHT


def test_proponente_tem_teto_de_candidatos():
    """Each candidate costs a classification; without a cap, a color-filled screen kills FPS."""

    import proposer

    rng = np.random.default_rng(0)

    imagem = np.full((2400, 1080, 3), 40, np.uint8)

    for _ in range(200):

        x = int(rng.integers(0, 950))
        y = int(rng.integers(0, 2300))

        imagem[y:y + 60, x:x + 60] = (
            int(rng.integers(100, 255)),
            int(rng.integers(100, 255)),
            int(rng.integers(100, 255)),
        )

    assert len(proposer.propose(imagem)) <= proposer.MAX_PROPOSALS


def test_supressao_remove_repetidos_da_mesma_categoria():

    import proposer

    def det(categoria, x, y, confianca):

        return {
            "category": categoria,
            "x": x, "y": y,
            "width": 100, "height": 100,
            "confidence": confianca,
        }

    deteccoes = [
        det("box", 100, 100, 0.9),
        det("box", 105, 105, 0.7),     # same thing, worse
        det("food", 100, 100, 0.8),    # other category, stays
        det("box", 900, 900, 0.6),     # far away, stays
    ]

    mantidas = proposer.suppress(deteccoes)

    assert len(mantidas) == 3, mantidas

    caixas = [(d["category"], d["x"]) for d in mantidas]

    assert ("box", 105) not in caixas

    # The highest-confidence one survived.
    assert ("box", 100) in caixas


def test_deteccao_do_modelo_tem_o_formato_que_a_maquina_le():
    """If the format diverges from what vision/detector.py produces, the symptom is a stalled bot, not an error."""

    import bot_ai

    class ModeloFalso:

        def classify_batch(self, vetores):
            return [("box", 0.95)] * len(vetores)

    imagem = frame_com_alvo()

    caixas = [{"x": 300, "y": 1500, "width": 96, "height": 88}]

    deteccoes = bot_ai.detect_with_model(
        ModeloFalso(),
        imagem,
        caixas,
        min_confidence=0.5,
    )

    assert len(deteccoes) == 1

    deteccao = deteccoes[0]

    # The keys that core/state_machine.py and vision/detector.py
    # use.
    for chave in (
        "category", "name", "confidence",
        "color_similarity", "x", "y", "width", "height",
    ):
        assert chave in deteccao, chave

    assert deteccao["category"] == "box"

    assert deteccao["x"] == 300
    assert deteccao["width"] == 96


def test_maquina_de_estados_aceita_as_deteccoes_do_modelo():
    """
    End to end, no device: model detection goes in, action comes out, and the target
    is the BOX — fix for the previous defect, which always tapped the
    screen center.
    """

    import bot_ai
    from core import state_machine as sm

    class AcoesFalsas:

        def __init__(self):
            self.feitas = []

        def execute(self, action, detection=None):
            self.feitas.append((action, detection))
            return True

        def is_busy(self):
            return False

        def swipe(self, direction):
            return True

    class ModeloFalso:

        def classify_batch(self, vetores):
            return [("box", 0.95)] * len(vetores)

    imagem = frame_com_alvo()

    deteccoes = bot_ai.detect_with_model(
        ModeloFalso(),
        imagem,
        [{"x": 300, "y": 1500, "width": 96, "height": 88}],
        min_confidence=0.5,
    )

    acoes = AcoesFalsas()

    maquina = sm.StateMachine(acoes)

    maquina.action_cooldown = 0.0
    maquina.action_settle = 0.0
    maquina.last_action_time = 0.0

    maquina.update(deteccoes, 0.0)

    assert acoes.feitas, "a máquina não agiu"

    acao, alvo = acoes.feitas[0]

    assert acao == "open_box", acao

    # The target is the detected box, not the screen center.
    assert alvo["x"] == 300 and alvo["y"] == 1500

    centro_da_tela = (1080 // 2, 2400 // 2)

    assert (
        alvo["x"] + alvo["width"] // 2,
        alvo["y"] + alvo["height"] // 2,
    ) != centro_da_tela


def test_background_nao_vira_deteccao():
    """"background" exists so the model can say "nothing here"; if it became a detection, it would act on scenery."""

    import bot_ai

    class ModeloFalso:

        def classify_batch(self, vetores):
            return [("background", 0.99)] * len(vetores)

    deteccoes = bot_ai.detect_with_model(
        ModeloFalso(),
        frame_com_alvo(),
        [{"x": 300, "y": 1500, "width": 96, "height": 88}],
        min_confidence=0.5,
    )

    assert deteccoes == []


def test_confianca_baixa_e_descartada():

    import bot_ai

    class ModeloFalso:

        def classify_batch(self, vetores):
            return [("box", 0.20)] * len(vetores)

    assert bot_ai.detect_with_model(
        ModeloFalso(),
        frame_com_alvo(),
        [{"x": 300, "y": 1500, "width": 96, "height": 88}],
        min_confidence=0.60,
    ) == []


def test_deteccoes_saem_ordenadas_por_confianca():
    """The StateMachine takes the FIRST detection of each category: order decides the target."""

    import bot_ai

    class ModeloFalso:

        def __init__(self):
            self.confiancas = [0.6, 0.95, 0.8]

        def classify_batch(self, vetores):
            return [
                ("box", self.confiancas[i])
                for i in range(len(vetores))
            ]

    deteccoes = bot_ai.detect_with_model(
        ModeloFalso(),
        frame_com_alvo(),
        [
            {"x": 100, "y": 100, "width": 96, "height": 88},
            {"x": 300, "y": 1500, "width": 96, "height": 88},
            {"x": 500, "y": 800, "width": 96, "height": 88},
        ],
        min_confidence=0.5,
    )

    confiancas = [d["confidence"] for d in deteccoes]

    assert confiancas == sorted(confiancas, reverse=True), confiancas


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
