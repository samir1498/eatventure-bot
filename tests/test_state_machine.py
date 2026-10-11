"""
Testes da máquina de estados. Não precisa de device: o ActionManager é
falso e o relógio é controlado, então dá para testar cooldown, timeout
de estado e prioridade sem esperar em tempo real.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from core import log                              # noqa: E402
from core.config import (                         # noqa: E402
    ACTION_COOLDOWN,
    ACTION_SETTLE,
    DISMISS_ACTIONS,
    DISMISS_ATTEMPTS_BEFORE_SCROLL,
    EXPLORATION_DELAY,
    EXPLORATION_DELAY_AFTER_ACTION,
    MAX_DETECTION_AGE,
    REPEATED_ACTION_WARNING,
    STATE_ENTRY_SETTLE,
    STATE_TIMEOUTS,
    SWIPE_WAITING_TIME,
    UPGRADE_FOOD_STUCK_UPDATES,
    UPGRADE_FOOD_SUPPRESS_UPDATES,
    UP_FOOD_WAIT,
)
from core import state_machine as sm              # noqa: E402


class Clock:

    def __init__(self):

        self.now = 10_000.0

    def monotonic(self):

        return self.now

    def advance(self, seconds):

        self.now += seconds


class FakeActions:

    def __init__(self):

        self.calls = []
        self.busy = False

    def execute(self, action, detection):

        self.calls.append(
            (
                action,
                detection["category"] if detection else None,
            )
        )

        return True

    def swipe(self, direction):

        self.calls.append(("swipe", direction))

        return True

    def is_busy(self):

        return self.busy

    # Conveniência para os testes
    @property
    def actions(self):

        return [call[0] for call in self.calls]


def detection(category, confidence=0.99, x=100, y=200):

    return {
        "category": category,
        "name": f"{category}.png",
        "confidence": confidence,
        "color_similarity": 0.95,
        "min_threshold": 0.9,
        "x": x,
        "y": y,
        "width": 80,
        "height": 80,
    }


def build():
    """Devolve (machine, actions, clock) com o cooldown já vencido, para o primeiro update poder agir."""

    clock = Clock()

    time.monotonic = clock.monotonic

    actions = FakeActions()

    machine = sm.StateMachine(actions)

    # Deixa o cooldown vencido.
    machine.last_action_time = clock.now - 100

    return machine, actions, clock


def test_prioridade_upgrade_vence_food():

    machine, actions, _ = build()

    machine.update([
        detection("food"),
        detection("upgrade"),
    ])

    assert actions.actions == ["upgrade"], actions.actions
    assert machine.state == sm.UPGRADE, machine.state


def test_fechar_vem_antes_do_jogo():
    """close/open_store/gray_max fecham o que não deveria estar aberto: prioridade sobre plane e food."""

    machine, actions, _ = build()

    machine.update([
        detection("plane"),
        detection("close"),
    ])

    assert actions.actions == ["close"], actions.actions

    # Fechar uma aba não muda de estado.
    assert machine.state == sm.NORMAL, machine.state


def test_uma_acao_por_cooldown():

    machine, actions, clock = build()

    detections = [detection("open_store")]

    machine.update(detections)

    # Segundo frame imediatamente depois: cooldown barra.
    machine.update(detections)

    assert len(actions.calls) == 1, actions.calls

    clock.advance(ACTION_COOLDOWN + 0.01)

    machine.update(detections)

    assert len(actions.calls) == 2, actions.calls


def test_acao_em_andamento_bloqueia():
    """O long press de comida dura 4s: enquanto roda, nenhum outro clique pode ser enfileirado."""

    machine, actions, _ = build()

    actions.busy = True

    machine.update([detection("food")])

    assert actions.calls == [], actions.calls
    assert machine.state == sm.NORMAL, machine.state


def test_timeout_do_estado_volta_para_normal():
    """Antes, um modal sem template travava o bot para sempre: RENOVATE só saía achando a moeda."""

    machine, actions, clock = build()

    machine._enter(sm.RENOVATE)

    # Nada na tela: sem timeout, ficaria preso aqui.
    machine.update([])

    assert machine.state == sm.RENOVATE, machine.state

    clock.advance(STATE_TIMEOUTS["RENOVATE"] + 0.1)

    machine.update([])

    assert machine.state == sm.NORMAL, machine.state


def test_delivery_e_helper_abrem_painel_e_usam_botao_interno():
    for entrada, estado, botao in (
        ("delivery", sm.DELIVERY, "accept_delivery"),
        ("helper", sm.HELPER, "invite_helper"),
    ):
        machine, actions, clock = build()

        assert entrada in machine.wanted_categories()
        machine.update([detection(entrada)])

        assert actions.actions == [entrada]
        assert machine.state == estado
        assert botao in machine.wanted_categories()
        assert entrada not in machine.wanted_categories()

        clock.advance(STATE_ENTRY_SETTLE[estado] + 0.01)
        machine.update([detection(botao), detection("close")])

        assert actions.actions == [entrada, botao]
        assert machine.state == sm.NORMAL


def test_delivery_e_helper_saem_por_timeout_sem_botao():
    for estado in (sm.DELIVERY, sm.HELPER):
        machine, actions, clock = build()
        machine._enter(estado)
        clock.advance(STATE_TIMEOUTS[estado] + 0.01)
        machine.update([])

        assert machine.state == sm.NORMAL
        assert actions.actions == []


def test_deteccao_velha_nao_clica():

    machine, actions, _ = build()

    machine.update(
        [detection("food")],
        lag=MAX_DETECTION_AGE + 0.1,
    )

    assert actions.calls == [], actions.calls


def test_upgrade_permanece_enquanto_houver_item():

    machine, actions, clock = build()

    machine._enter(sm.UPGRADE)

    # Painel abre com animação: STATE_ENTRY_SETTLE segura a mão antes de
    # agir. Ver test_upgrade_nao_fecha_no_meio_da_animacao.
    clock.advance(STATE_ENTRY_SETTLE[sm.UPGRADE] + 0.01)

    machine.update([detection("up_upgrade")])

    assert actions.actions == ["upgrade_item"], actions.actions
    assert machine.state == sm.UPGRADE, machine.state

    # Sem mais itens: fecha e volta.
    clock.advance(ACTION_COOLDOWN + 0.01)

    machine.update([detection("close")])

    assert actions.actions[-1] == "close", actions.actions
    assert machine.state == sm.NORMAL, machine.state


def test_upgrade_nao_fecha_no_meio_da_animacao():
    """
    O bug relatado: ao abrir a tela de upgrade, o bot fechava ela na hora.
    Enquanto o painel entra, o "X" já casa com o template e os botões de
    upgrade ainda não — a regra 1 não acha nada, a regra 2 acha, e o bot
    desfaz o que acabou de fazer. ACTION_SETTLE não cobre isto: é contado
    do TOQUE que abriu, dimensionado para a animação de FECHAR.
    """

    machine, actions, clock = build()

    espera = STATE_ENTRY_SETTLE[sm.UPGRADE]

    assert espera > ACTION_SETTLE, (
        "uma espera de entrada menor que a de ação não teria "
        "efeito nenhum"
    )

    machine._enter(sm.UPGRADE)

    # No meio da animação, o "X" é a única coisa que casa. Já passado o
    # ACTION_SETTLE, para provar que é a espera de ENTRADA que segura.
    clock.advance(ACTION_SETTLE + 0.01)

    machine.update([detection("close")])

    assert actions.actions == [], (
        "fechou o painel no meio da animação de abrir"
    )

    assert machine.state == sm.UPGRADE, machine.state

    # Painel montado: agora o upgrade aparece e é ele que vence.
    clock.advance(espera)

    machine.update([
        detection("up_upgrade"),
        detection("close"),
    ])

    assert actions.actions == ["upgrade_item"], actions.actions

    assert machine.state == sm.UPGRADE, machine.state


def test_espera_de_entrada_cabe_no_timeout_do_estado():
    """A espera de entrada come o tempo que o bot tem para agir dentro do estado: mesmo relógio."""

    for estado, espera in STATE_ENTRY_SETTLE.items():

        limite = STATE_TIMEOUTS.get(estado)

        if limite is None:
            continue

        assert espera < limite / 2, (
            estado,
            espera,
            limite,
        )


def test_gray_coin_dispensa_como_o_gray_max():
    """
    `gray_coin` faz o que `gray_max` faz — dispensa e volta a NORMAL. O
    ponto tocado é assunto do ActionManager
    (test_pipeline::test_gray_coin_toca_no_ponto_proprio).
    """

    for categoria in ("gray_max", "gray_coin"):

        machine, actions, _ = build()

        machine.update([detection(categoria)])

        assert actions.actions == [categoria], (
            categoria,
            actions.actions,
        )

        # E em FOOD também: é lá que o painel esgotado aparece.
        machine, actions, _ = build()

        machine._enter(sm.FOOD)

        machine.update([detection(categoria)])

        assert actions.actions == [categoria], (
            categoria,
            actions.actions,
        )

        assert machine.state == sm.NORMAL, machine.state


def test_dispensa_em_food_acontece_antes_de_desistir():
    """
    Em FOOD, a dispensa tem de ser avaliada ANTES do _wait_or_give_up, que
    pode chamar _enter(NORMAL). Se a dispensa vier depois, a ação sai
    quando a máquina JÁ SE CONSIDERA em NORMAL — decidindo por regra de
    FOOD num estado que não é mais FOOD, rotulando a amostra errada.
    """

    machine, actions, clock = build()

    # Espia o estado NO MOMENTO em que a ação é despachada.
    estados = []

    original = actions.execute

    def espiao(action, deteccao):

        estados.append(machine.state)

        return original(action, deteccao)

    actions.execute = espiao

    machine._enter(sm.FOOD)

    # Timer de desistência JÁ VENCIDO: situação em que _wait_or_give_up agiria.
    machine.up_food_wait_start = clock.now

    clock.advance(UP_FOOD_WAIT + 0.1)

    machine.update([detection("gray_coin")])

    assert actions.actions == ["gray_coin"], actions.actions

    assert estados == [sm.FOOD], (
        f"dispensou já em {estados} — o _wait_or_give_up correu "
        f"antes e trocou o estado debaixo da ação"
    )


def test_food_desiste_depois_da_espera():

    machine, actions, clock = build()

    machine._enter(sm.FOOD)

    machine.update([detection("up_food")])

    assert actions.actions == ["upgrade_food"], actions.actions
    assert machine.state == sm.FOOD, machine.state

    # Nenhum up_food novo apareceu.
    clock.advance(UP_FOOD_WAIT + 0.1)

    machine.update([])

    assert machine.state == sm.NORMAL, machine.state


def test_food_sem_up_food_nao_adia_a_exploracao():

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0
    machine.swipe_count = 2

    machine.update([detection("food")])

    assert actions.actions == ["food"], actions.actions
    assert machine.state == sm.FOOD, machine.state
    assert machine.swipe_count == 2, machine.swipe_count
    assert not machine.explore_found

    clock.advance(STATE_TIMEOUTS["FOOD"] + 0.1)
    machine.update([])

    assert machine.state == sm.NORMAL, machine.state


def test_food_novamente_sem_up_food_preserva_ritmo_do_swipe():

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0
    machine.swipe_count = 2

    machine.update([detection("food")])
    clock.advance(STATE_TIMEOUTS["FOOD"] + 0.1)
    machine.update([])

    assert machine.state == sm.NORMAL, machine.state
    assert not machine.explore_found

    machine.update([detection("food")])

    assert actions.actions == ["food", "food"], actions.actions
    assert machine.state == sm.FOOD, machine.state
    assert machine.swipe_count == 2, machine.swipe_count
    assert not machine.explore_found

    clock.advance(STATE_TIMEOUTS["FOOD"] + 0.1)
    machine.update([])
    machine.update([])

    assert machine.swipe_count == 3, machine.swipe_count


def test_exploracao_faz_swipe_e_inverte():

    machine, actions, clock = build()

    for _ in range(machine.max_swipes):

        clock.advance(machine.exploration_delay + 0.1)

        machine.update([])

    swipes = [
        call for call in actions.calls
        if call[0] == "swipe"
    ]

    assert len(swipes) == machine.max_swipes, swipes

    # Não fixa QUAL direção começa (é ajuste de config):
    # verifica que todas foram na mesma.
    primeira = swipes[0][1]

    assert all(
        call[1] == primeira for call in swipes
    ), swipes

    # Passou do limite: inverte.
    clock.advance(machine.exploration_delay + 0.1)

    machine.update([])

    oposta = "up" if primeira == "down" else "down"

    assert actions.calls[-1] == ("swipe", oposta), (
        actions.calls[-1],
        oposta,
    )


def swipes_de(actions):

    return [
        call for call in actions.calls
        if call[0] == "swipe"
    ]


def test_achar_algo_adia_a_exploracao_sem_cancelar():
    """
    Achar algo ADIA o swipe, não o cancela.

    Antes, achar zerava a exploração, então o swipe só saía
    depois de 5 s de tela COMPLETAMENTE vazia. Agora o swipe é
    periódico e o que muda é o intervalo:

        não achou nada .... a cada EXPLORATION_DELAY        (5 s)
        achou algo ........ espera EXPLORATION_DELAY_AFTER_ACTION
                            (15 s) e volta ao ritmo de 5 s
    """

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    clock.advance(machine.exploration_delay + 0.1)

    machine.update([detection("open_store")])

    # No frame em que achou, não rola.
    assert not swipes_de(actions), actions.calls

    # O intervalo em vigor passou a ser o longo.
    assert machine.exploration_interval() == (
        EXPLORATION_DELAY_AFTER_ACTION
    )

    # Passado o intervalo CURTO, ainda não: está adiado.
    clock.advance(machine.exploration_delay + 0.1)

    machine.update([])

    assert not swipes_de(actions), (
        "rolou antes do intervalo longo — o adiamento não "
        "está valendo"
    )

    # Passado o LONGO, rola.
    clock.advance(
        EXPLORATION_DELAY_AFTER_ACTION
        - machine.exploration_delay
    )

    machine.update([])

    assert len(swipes_de(actions)) == 1, actions.calls

    # E volta ao ritmo curto.
    assert machine.exploration_interval() == (
        machine.exploration_delay
    )

    clock.advance(machine.exploration_delay + 0.1)

    machine.update([])

    assert len(swipes_de(actions)) == 2, actions.calls


def test_achar_algo_nao_zera_o_ciclo_de_varredura():
    """
    O ciclo é 5 para um lado e 5 para o outro, e achar algo no
    meio NÃO devolve a contagem para o começo.

    Zerava antes, e o efeito era o bot varrer sempre o mesmo
    pedaço da tela: como quase todo swipe revela algum alvo, a
    contagem voltava a zero antes de a volta fechar e a metade
    distante do restaurante nunca era visitada.
    """

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    # Dois swipes, para ficar no MEIO da volta.
    for _ in range(2):

        clock.advance(machine.exploration_delay + 0.1)

        machine.update([])

    assert machine.swipe_count == 2, machine.swipe_count

    contagem = machine.swipe_count
    direcao = machine.swipe_direction

    # Acha algo.
    #
    # `box` de propósito: a regra dela é (box, open_box, None),
    # ou seja PERMANECE em NORMAL. Com `food` a máquina sairia
    # para o estado FOOD, onde a exploração nem roda — e o teste
    # mediria outra coisa.
    clock.advance(0.1)

    machine.update([detection("box")])

    assert machine.swipe_count == contagem, (
        f"a contagem voltou para {machine.swipe_count} — o "
        f"ciclo zerou ao achar algo"
    )

    assert machine.swipe_direction == direcao

    # O próximo swipe CONTINUA a volta.
    clock.advance(EXPLORATION_DELAY_AFTER_ACTION + 0.1)

    machine.update([])

    assert machine.swipe_count == contagem + 1, (
        machine.swipe_count
    )

    assert machine.swipe_direction == direcao


def test_varredura_fecha_a_volta_e_inverte_indefinidamente():
    """
    5 para um lado, 5 para o outro, sem parar — inclusive
    quando o bot acha algo a cada volta, que é o caso real.
    """

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    for volta in range(12):

        # Acha algo entre um swipe e o outro, sempre.
        #
        # `box` permanece em NORMAL (ver o teste acima).
        clock.advance(0.1)

        machine.update([detection("box")])

        clock.advance(EXPLORATION_DELAY_AFTER_ACTION + 0.1)

        machine.update([])

    direcoes = [call[1] for call in swipes_de(actions)]

    assert len(direcoes) == 12, direcoes

    # Blocos de max_swipes na mesma direção, alternando.
    primeira = direcoes[0]

    esperado = [
        primeira
        if (i // machine.max_swipes) % 2 == 0
        else ("up" if primeira == "down" else "down")
        for i in range(12)
    ]

    assert direcoes == esperado, (direcoes, esperado)


def test_categorias_do_estado_sao_reduzidas():
    """Faz a passada do detector cair de ~264ms para ~16ms dentro da tela de upgrade."""

    machine, _, _ = build()

    normal = machine.wanted_categories()

    machine._enter(sm.UPGRADE)

    upgrade = machine.wanted_categories()

    assert "food" in normal
    assert "food" not in upgrade, upgrade
    assert list(upgrade) == ["up_upgrade", "close"], upgrade


def test_categorias_vem_na_ordem_da_prioridade():
    """
    A ORDEM é contrato: o detector para na primeira categoria que
    encontrar, então ordem embaralhada faria o bot agir na regra errada
    — e `food` (2/3 dos templates) deixaria de ser a última.
    """

    machine, _, _ = build()

    categorias = list(machine.wanted_categories())

    esperada = [
        category
        for category, _, _ in sm.NORMAL_RULES
    ]

    assert categorias == esperada, categorias

    # Sem repetição: categoria repetida seria template
    # procurado duas vezes na mesma passada.
    assert len(categorias) == len(set(categorias)), categorias

    # E `food` continua no fim, que é onde a economia vive.
    assert categorias[-1] == "food", categorias


def test_estado_nao_vaza_variavel():
    """up_food_wait_start era compartilhada entre FOOD e NEW_POINT."""

    machine, _, clock = build()

    machine._enter(sm.FOOD)

    machine.update([detection("up_food")])

    assert machine.up_food_wait_start is not None

    machine._enter(sm.NEW_POINT)

    assert machine.up_food_wait_start is None


def test_up_food_em_normal_evolui_a_comida():
    """
    up_food em NORMAL faz o long press de evolução, igual a FOOD. ESCOLHA
    DELIBERADA: não é o que parece "seguro" — GASTA MOEDA a cada painel
    aberto sem querer e trava o bot por UPGRADE_FOOD_PRESS segundos. A
    alternativa (dispensar em ponto neutro) é trocar para "dismiss" em
    NORMAL_RULES — ação já implementada e testada.
    """

    machine, actions, _ = build()

    machine.update([detection("up_food")])

    assert actions.actions == ["upgrade_food"], actions.actions

    # A regra não declara próximo estado: segue em NORMAL.
    assert machine.state == sm.NORMAL, machine.state


def test_up_food_faz_o_mesmo_em_normal_e_em_food():
    """Existe para o dia em que alguém "corrigir" um dos dois lados sem olhar o outro."""

    normal, acoes_normal, _ = build()

    normal.update([detection("up_food")])

    assert acoes_normal.actions == ["upgrade_food"]

    machine, actions, _ = build()

    machine._enter(sm.FOOD)

    machine.update([detection("up_food")])

    assert actions.actions == ["upgrade_food"], actions.actions


def test_up_food_perde_para_o_close():
    """O X fecha o painel sem gastar nada, então vem antes do long press de evolução."""

    machine, actions, _ = build()

    machine.update([
        detection("up_food"),
        detection("close"),
    ])

    assert actions.actions == ["close"], actions.actions


def test_up_food_vence_acao_de_jogo():
    """Fechar o que não deveria estar aberto vem antes de qualquer ação de jogo."""

    machine, actions, _ = build()

    machine.update([
        detection("food"),
        detection("up_food"),
        detection("box"),
    ])

    assert actions.actions == ["upgrade_food"], actions.actions


def test_acao_repetida_gera_aviso():
    """NORMAL não tem timeout: uma regra que dispara sem resolver repetiria para sempre em silêncio."""

    import logging

    machine, actions, clock = build()

    avisos = []

    class Coletor(logging.Handler):

        def emit(self, record):

            if record.levelno >= logging.WARNING:

                avisos.append(record.getMessage())

    handler = Coletor()

    sm.logger.addHandler(handler)

    # O runner roda em ERROR; sem baixar o nível aqui, o
    # warning é descartado antes de chegar ao handler.
    nivel_original = sm.logger.level

    sm.logger.setLevel(logging.WARNING)

    try:

        # O painel nunca fecha: up_food continua na tela.
        for _ in range(REPEATED_ACTION_WARNING):

            machine.update([detection("up_food")])

            clock.advance(ACTION_COOLDOWN + 0.01)

    finally:

        sm.logger.removeHandler(handler)

        sm.logger.setLevel(nivel_original)

    assert len(actions.calls) == REPEATED_ACTION_WARNING, (
        actions.calls
    )

    assert avisos, "nenhum aviso de ação repetida"

    assert "upgrade_food" in avisos[0], avisos


def test_contagem_de_repeticao_zera_ao_progredir():

    machine, actions, clock = build()

    machine.update([detection("up_food")])

    assert machine._repeat_count == 1

    clock.advance(ACTION_COOLDOWN + 0.01)

    # Ação diferente: a contagem reinicia.
    machine.update([detection("open_store")])

    assert machine._repeat_count == 1, machine._repeat_count

    assert machine._last_action == "open_store_click", (
        machine._last_action
    )


def test_max_preso_rola_a_tela_e_tenta_de_novo():
    """
    O cenário: o ponto fixo ABRE um painel, ele mostra "max", a regra do
    gray_max toca o mesmo ponto e reabre. A saída não é o BACK (neste
    jogo ele SAI DO JOGO): é rolar até o canto de baixo ficar vazio e
    tentar de novo.
    """

    machine, actions, clock = build()

    frames = (DISMISS_ATTEMPTS_BEFORE_SCROLL + 1) * 2

    for _ in range(frames):

        machine.update([detection("gray_max")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    esperado = (
        ["gray_max"] * DISMISS_ATTEMPTS_BEFORE_SCROLL
        + ["scroll_bottom"]
    ) * 2

    assert actions.actions == esperado, actions.actions

    # E não para no primeiro escape: o ciclo se repete.
    assert actions.actions.count("scroll_bottom") == 2, (
        actions.actions
    )


def test_nunca_usa_back():
    """Se alguém reintroduzir BACK, este teste avisa antes de o bot fechar o jogo sozinho."""

    from actions.manager import ACTION_TABLE

    assert "back" not in ACTION_TABLE, (
        "BACK sai do EatVenture — não pode entrar na tabela"
    )

    machine, actions, clock = build()

    for _ in range(12):

        machine.update([detection("gray_max")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert "back" not in actions.actions, actions.actions


def test_up_food_em_normal_nao_escala_para_rolagem():
    """
    A escada de fechamento vale só para DISMISS_ACTIONS, e "upgrade_food"
    não é uma delas — up_food preso em NORMAL cai no _trap_guard (3
    presses, 2 rodadas de "dismiss", press tenta de novo) em vez de
    rolar. O "dismiss" conta na escada mas nunca acumula o bastante
    para virar scroll_bottom, porque o press do meio zera a contagem.
    """

    machine, actions, clock = build()

    for _ in range(DISMISS_ATTEMPTS_BEFORE_SCROLL + 3):

        machine.update([detection("up_food")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert "scroll_bottom" not in actions.actions, actions.actions

    assert set(actions.actions) == {"upgrade_food", "dismiss"}, (
        actions.actions
    )

    # A escada nunca chega ao limiar da rolagem.
    assert machine._dismiss_attempts == 0


def test_painel_travado_suprime_up_food_e_libera_outros_upgrades():
    """
    O cenário da Wrap emperrada: up_food visível por N updates
    seguidos sem resolver (preço inalcançável) = popup travado.
    A supressão pula up_food para que regras de prioridade menor
    (upgrades de outras estações) tenham a vez; quando a janela
    esgota, up_food ganha outra chance (alterna).
    """

    machine, actions, clock = build()

    tela = [detection("up_food"), detection("upgrade")]

    for _ in range(UPGRADE_FOOD_STUCK_UPDATES + 5):

        machine.update(tela)

        clock.advance(ACTION_COOLDOWN + 0.01)

    # A regra de baixo ("upgrade") teve a vez no meio da janela.
    assert "upgrade" in actions.actions, actions.actions

    # Sair de NORMAL é progresso: a supressão zerou na troca.
    assert machine._up_food_suppressed == 0

    # Volta para NORMAL e confirma a alternância: sozinho na
    # tela, up_food é pausado pela supressão e depois volta
    # (alterna em vez de morrer para um lado só).
    machine.update([detection("close")])

    clock.advance(ACTION_COOLDOWN + 0.01)

    assert machine.state == sm.NORMAL, machine.state

    antes = actions.actions.count("upgrade_food")

    for _ in range(UPGRADE_FOOD_STUCK_UPDATES):

        machine.update([detection("up_food")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert machine._up_food_suppressed > 0

    for _ in range(UPGRADE_FOOD_SUPPRESS_UPDATES + 1):

        machine.update([detection("up_food")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert machine._up_food_suppressed == 0

    assert actions.actions.count("upgrade_food") > antes, (
        actions.actions
    )


def test_nao_age_duas_vezes_sobre_a_mesma_tela():
    """
    O bug relatado: fechava o "MAX" e tocava DE NOVO no mesmo ponto,
    REABRINDO o painel. Causa: o cooldown (0.5s) libera antes de existir
    frame que mostre o efeito, porque o detector tem ~0.535s de atraso —
    a detecção em mão vem de ANTES do toque. Números medidos no device.
    """

    COOLDOWN = 0.5
    ATRASO = 0.535

    machine, actions, clock = build()

    machine.action_cooldown = COOLDOWN

    # Primeiro toque, sobre uma tela recém-vista.
    machine.update([detection("gray_max")], 0.0)

    assert actions.actions == ["gray_max"], actions.actions

    # Passa o cooldown. A detecção continua sendo a MESMA tela
    # de antes do toque — é o que o atraso do detector entrega.
    clock.advance(COOLDOWN + 0.01)

    machine.update([detection("gray_max")], ATRASO)

    assert actions.actions == ["gray_max"], (
        "tocou duas vezes sobre a mesma tela: "
        f"{actions.actions}"
    )


def test_age_quando_o_frame_e_posterior_a_acao():
    """O outro lado: frame de DEPOIS da ação com o painel ainda lá tem de agir, senão viraria paralisia."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.5

    machine.update([detection("gray_max")], 0.0)

    clock.advance(0.6)

    # Atraso pequeno: o frame é de depois do toque.
    machine.update([detection("gray_max")], 0.05)

    assert actions.actions == ["gray_max", "gray_max"], (
        actions.actions
    )


def test_frame_antigo_nao_bloqueia_para_sempre():
    """O risco da correção seria travar o bot quando o detector está lento."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.5

    machine.update([detection("gray_max")], 0.0)

    # Vários frames velhos: nenhum age.
    for _ in range(5):

        clock.advance(0.6)

        machine.update([detection("gray_max")], 10.0)

    assert actions.actions == ["gray_max"], actions.actions

    # Frame fresco: age.
    machine.update([detection("gray_max")], 0.01)

    assert len(actions.actions) == 2, actions.actions


def test_primeira_acao_nao_precisa_esperar():
    """Sem esta guarda o bot não faria NADA no start, porque last_action_time = 0 é anterior a qualquer frame."""

    machine, actions, _ = build()

    machine.last_action_time = 0.0

    # Abaixo de MAX_DETECTION_AGE: acima dela cairia na guarda
    # de detecção velha, que é outra coisa.
    machine.update([detection("gray_max")], 1.0)

    assert actions.actions == ["gray_max"], actions.actions


def test_a_guarda_vale_para_todo_caminho_de_acao():
    """
    A checagem vive em _can_act, ponto único por onde passam TODOS os
    caminhos (regras, FOOD/NEW_POINT, swipe de exploração). Não dá para
    montar o caso pela exploração: exploration_delay = 5s de tela vazia
    já teria descartado o frame por MAX_DETECTION_AGE = 2s antes daqui.
    """

    machine, actions, clock = build()

    machine.action_cooldown = 0.0
    machine.action_settle = 0.4

    machine.last_action_time = clock.now

    # Frame capturado ANTES da ação: barrado.
    machine._frame_time = clock.now - 0.1

    assert not machine._can_act()

    # Frame do MESMO instante: barrado. Não pode mostrar o
    # efeito de algo que acabou de acontecer.
    machine._frame_time = clock.now

    assert not machine._can_act()

    # Frame posterior à ação MAS dentro da animação: barrado. É o caso
    # que a guarda causal sozinha deixava passar, causando o duplo toque.
    machine._frame_time = clock.now + 0.2

    assert not machine._can_act(), (
        "frame dentro da janela de animação não pode agir"
    )

    # Passado o settle: liberado.
    machine._frame_time = clock.now + 0.41

    assert machine._can_act()


def test_swipe_espera_a_vista_parar():
    """
    O bug relatado: depois de rolar, o bot detectava um alvo num frame
    capturado enquanto a vista ainda escorregava e tocava onde o alvo
    ESTAVA. Um swipe move a VISTA INTEIRA e o jogo desliza por inércia
    depois do dedo sair — daí a espera própria (SWIPE_WAITING_TIME),
    contada do FIM do gesto.
    """

    machine, actions, clock = build()

    # Tela vazia agora mesmo: dispara a exploração.
    machine.exploration_delay = 0.0

    machine.update([], lag=0.0)

    assert ("swipe", "down") in actions.calls, actions.calls

    momento_do_swipe = clock.now

    # O swipe LEVA TEMPO para executar (SWIPE_DURATION_MS) — ponto do
    # bug: medida da submissão, a espera seria consumida pelo próprio
    # gesto e não sobraria nada.
    clock.advance(0.5)

    actions.last_finished_at = clock.now

    fim_do_swipe = clock.now

    # Alvo visível, mas a vista ainda escorrega. PASSADA a espera de um
    # toque, e ainda barrado: é aí que a espera do swipe se distingue da
    # de toque — testar antes de ACTION_SETTLE não provaria nada.
    clock.advance(ACTION_SETTLE + 0.01)

    machine.update([detection("food")], lag=0.0)

    assert machine.action_counts["food"] == 0, (
        "tocou antes de a vista parar — é o clique que cai no "
        "lugar errado"
    )

    # Passado SWIPE_WAITING_TIME do FIM do swipe: liberado.
    clock.advance(SWIPE_WAITING_TIME)

    assert clock.now > fim_do_swipe + SWIPE_WAITING_TIME

    machine.update([detection("food")], lag=0.0)

    assert machine.action_counts["food"] == 1, (
        machine.action_counts
    )

    # E a espera foi contada do fim do gesto, não da submissão:
    # senão este teste passaria com o bug dentro.
    assert (
        clock.now - momento_do_swipe
        > SWIPE_WAITING_TIME + 0.5
    )


def test_piso_do_frame_concorda_com_o_can_act():
    """
    `frame_floor()` é o MESMO prazo que `_can_act` aplica, para o
    VisionWorker poder pular frame que a máquina descartaria de todo
    jeito. Se divergirem, o worker joga fora frame que serviria (bot
    cego) ou analisa frame que não serve (passada perdida).
    """

    # Máquina recém-nascida (build() adianta o relógio da última ação
    # para vencer o cooldown, não serve aqui): sem ação, um piso
    # qualquer cegaria o bot no arranque.
    assert sm.StateMachine(FakeActions()).frame_floor() == 0.0

    machine, actions, clock = build()

    machine.action_cooldown = 0.0

    assert machine._act("food", detection("food")) is True

    actions.last_finished_at = clock.now

    piso = machine.frame_floor()

    assert piso == clock.now + ACTION_SETTLE, piso

    # Exatamente no piso: barrado pelos dois.
    machine._frame_time = piso

    assert not machine._can_act()

    # Um fio acima: liberado pelos dois.
    machine._frame_time = piso + 0.001

    assert machine._can_act()

    # E o piso do swipe é o maior dos dois.
    machine._last_was_swipe = True

    assert machine.frame_floor() == clock.now + SWIPE_WAITING_TIME


def test_swipe_espera_mais_que_um_toque():
    """SWIPE_WAITING_TIME abaixo de ACTION_SETTLE não teria sentido: o swipe mexe MAIS na tela."""

    assert SWIPE_WAITING_TIME >= ACTION_SETTLE, (
        SWIPE_WAITING_TIME,
        ACTION_SETTLE,
    )

    # E não tanto que a exploração fique lenta: o bot faz
    # MAX_SWIPES seguidos procurando conteúdo.
    assert SWIPE_WAITING_TIME <= 2.0, SWIPE_WAITING_TIME


def test_scroll_bottom_usa_a_espera_do_swipe():
    """`scroll_bottom` é SEIS swipes seguidos; com espera curta, a escada voltaria a tocar em tela em movimento."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.0

    assert machine._act("scroll_bottom", None) is True

    assert machine._last_was_swipe is True, (
        "scroll_bottom tem de contar como movimento de vista"
    )

    actions.last_finished_at = clock.now

    # Dentro da espera de swipe (mas fora da de toque): barrado.
    machine._frame_time = clock.now + ACTION_SETTLE + 0.01

    assert not machine._can_act(), (
        "a espera curta de toque não serve para seis swipes"
    )

    machine._frame_time = clock.now + SWIPE_WAITING_TIME + 0.01

    assert machine._can_act()


def test_toque_comum_nao_herda_a_espera_do_swipe():
    """A bandeira tem de VOLTAR: um swipe não pode deixar toques seguintes pagando a espera longa."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.0

    machine._last_was_swipe = True

    assert machine._act("food", detection("food")) is True

    assert machine._last_was_swipe is False

    actions.last_finished_at = clock.now

    # Basta a espera de toque.
    machine._frame_time = clock.now + ACTION_SETTLE + 0.01

    assert machine._can_act()


def test_nao_toca_com_o_painel_ja_fechado():
    """
    O bug relatado, reproduzido no domínio do tempo. Simula a VERDADE do
    jogo vs. a visão ATRASADA da máquina: toque com painel aberto fecha
    (após animação); toque com painel fechado ABRE (o DISMISS_POINT abre
    algo — por isso o duplo toque dói); a máquina só vê a tela de
    `atraso` segundos atrás. O toque espúrio é o dado com painel JÁ
    fechado — sem a guarda saem vários, com ela, nenhum.
    """

    ATRASO = 0.535     # medido no device, 165 templates
    COOLDOWN = 0.5
    ANIMACAO = 0.3     # tempo do jogo para fechar

    def roda(settle):

        clock = Clock()

        sm.time.monotonic = clock.monotonic

        actions = FakeActions()

        machine = sm.StateMachine(actions)

        machine.action_cooldown = COOLDOWN
        machine.action_settle = settle
        machine.last_action_time = clock.now - 100

        # (instante em que passa a valer, aberto?)
        eventos = []

        def aberto(instante):

            for t0, valor in reversed(eventos):

                if instante >= t0:
                    return valor

            return True

        toques = espurios = 0

        fim = clock.now + 8.0

        while clock.now < fim:

            visto = aberto(clock.now - ATRASO)

            antes = actions.actions.count("gray_max")

            machine.update(
                [detection("gray_max")] if visto else [],
                ATRASO,
            )

            if actions.actions.count("gray_max") > antes:

                toques += 1

                estava = aberto(clock.now)

                if not estava:
                    espurios += 1

                eventos.append((clock.now + ANIMACAO, not estava))

            clock.advance(0.02)

        return toques, espurios

    # Sem settle (só cooldown + guarda causal): toca no vazio.
    _, espurios_sem = roda(0.0)

    assert espurios_sem > 0, (
        "a simulação deveria reproduzir o bug com settle=0"
    )

    # Com settle cobrindo a animação: um toque, nenhum espúrio.
    toques, espurios = roda(ACTION_SETTLE)

    assert espurios == 0, f"{espurios} toque(s) no painel fechado"

    assert toques == 1, f"{toques} toques onde 1 bastava"


def test_settle_cobre_a_animacao_configurada():
    """O settle tem de ser maior que a animação do jogo. Baixar ACTION_SETTLE para 0.1 traria o duplo toque de volta."""

    # Animação típica de painel de jogo.
    assert ACTION_SETTLE >= 0.3, ACTION_SETTLE

    # E não tão alto que o bot fique lento: acima de ~1s cada
    # decisão custaria settle + atraso do detector.
    assert ACTION_SETTLE <= 1.0, ACTION_SETTLE


def test_escada_zera_quando_algo_e_resolvido():
    """Se uma ação normal aconteceu, a contagem recomeça: não rola a tela por tentativas antigas."""

    machine, actions, clock = build()

    for _ in range(DISMISS_ATTEMPTS_BEFORE_SCROLL):

        machine.update([detection("gray_max")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert machine._dismiss_attempts == (
        DISMISS_ATTEMPTS_BEFORE_SCROLL
    )

    # Progresso: uma ação de jogo.
    machine.update([detection("open_store")])

    clock.advance(ACTION_COOLDOWN + 0.01)

    assert machine._dismiss_attempts == 0

    # Próximo gray_max começa do ponto, não da rolagem.
    machine.update([detection("gray_max")])

    assert actions.actions[-1] == "gray_max", actions.actions


def test_escada_zera_ao_trocar_de_estado():

    machine, actions, clock = build()

    machine.update([detection("gray_max")])

    assert machine._dismiss_attempts == 1

    machine._enter(sm.UPGRADE)

    assert machine._dismiss_attempts == 0
    assert machine._dismiss_rounds == 0


def test_apenas_acoes_de_fechar_escalam():
    """Ação de jogo não pode virar rolagem de tela por repetir."""

    machine, actions, clock = build()

    for _ in range(DISMISS_ATTEMPTS_BEFORE_SCROLL + 3):

        machine.update([detection("open_store")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert set(actions.actions) == {"open_store_click"}, actions.actions

    assert "gray_max" in DISMISS_ACTIONS
    assert "dismiss" in DISMISS_ACTIONS
    assert "open_store_click" not in DISMISS_ACTIONS


def main():

    log.setup("ERROR")

    real_monotonic = time.monotonic

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    failures = 0

    for test in tests:

        try:

            test()

            print(f"  ok    {test.__name__}")

        except AssertionError as error:

            failures += 1

            print(f"  FALHA {test.__name__}: {error}")

        except Exception as error:

            failures += 1

            print(
                f"  ERRO  {test.__name__}: "
                f"{type(error).__name__}: {error}"
            )

        finally:

            time.monotonic = real_monotonic

    print()

    if failures:

        print(f"{failures}/{len(tests)} falharam.")

        return 1

    print(f"{len(tests)}/{len(tests)} passaram.")

    return 0


if __name__ == "__main__":

    sys.exit(main())
