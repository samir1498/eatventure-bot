"""Máquina de estados: tabela de regras por estado, única função _act, timeout por estado."""

import logging
import time
from collections import Counter

from core import log
from core.metrics import formata_duracao
from core.config import (
    ACTION_COOLDOWN,
    ACTION_SETTLE,
    CYCLE_CATEGORIES,
    DISMISS_ACTIONS,
    DISMISS_ATTEMPTS_BEFORE_SCROLL,
    EXPLORATION_DELAY,
    EXPLORATION_DELAY_AFTER_ACTION,
    FOOD_IGNORE_IOU,
    FOOD_MAX_ATTEMPTS,
    MAX_DETECTION_AGE,
    MAX_SWIPES,
    REPEATED_ACTION_WARNING,
    STATE_ENTRY_SETTLE,
    STATE_TIMEOUTS,
    SWIPE_START_DIRECTION,
    SWIPE_WAIT_FOR_NO_ACTION,
    SWIPE_WAITING_TIME,
    UPGRADE_FOOD_DISMISS_ROUNDS,
    UPGRADE_FOOD_MAX_REPEATS,
    UPGRADE_FOOD_STUCK_UPDATES,
    UPGRADE_FOOD_SUPPRESS_UPDATES,
    UP_FOOD_WAIT,
    VIEW_MOVING_ACTIONS,
)

logger = log.get("state")


NORMAL = "NORMAL"
RENOVATE = "RENOVATE"
FOOD = "FOOD"
NEW_POINT = "NEW_POINT"
UPGRADE = "UPGRADE"
DELIVERY = "DELIVERY"
HELPER = "HELPER"


# Regras: (categoria, ação, próximo estado). Ordem = prioridade
# (a primeira categoria encontrada é a única executada no
# frame); next_state None mantém o estado atual.

# As quatro primeiras fecham o que não deveria estar aberto,
# antes de qualquer ação de jogo.
#
# up_food em NORMAL faz o MESMO que em FOOD (long press de
# UPGRADE_FOOD_PRESS s). Se o painel abrir sem querer e o press
# nunca resolver, _trap_guard desvia para "dismiss" (HOLD em
# ponto fixo, coberto por
# tests/test_pipeline.py::test_dismiss_toca_no_ponto_neutro)
# por UPGRADE_FOOD_DISMISS_ROUNDS rodadas antes de tentar o
# press de novo. "upgrade_food" segue fora de DISMISS_ACTIONS
# de propósito: a escada de fechamento não vale para ele.
NORMAL_RULES = [
    ("open_store", "open_store_click", None),
    ("close", "close", None),
    ("gray_max", "gray_max", None),
    ("gray_coin", "gray_coin", None),
    ("up_food", "upgrade_food", None),

    ("plane", "plane", RENOVATE),
    ("build", "click", RENOVATE),
    ("delivery", "delivery", DELIVERY),
    ("helper", "helper", HELPER),
    ("upgrade", "upgrade", UPGRADE),
    ("new_point", "new_point", NEW_POINT),
    # ("box", "open_box", None) disabled: user opens boxes manually.
    ("food", "food", FOOD),
]

RENOVATE_RULES = [
    ("renovate", "renovate_click", NORMAL),
    ("fly", "renovate_click", NORMAL),
]

DELIVERY_RULES = [
    ("accept_delivery", "accept_delivery", NORMAL),
    ("close", "close", NORMAL),
]

HELPER_RULES = [
    ("invite_helper", "invite_helper", NORMAL),
    ("close", "close", NORMAL),
]

# up_upgrade permanece em UPGRADE de propósito: no próximo
# frame procuramos outro item para evoluir. Só saímos da
# tela quando não houver mais nenhum.
UPGRADE_RULES = [
    ("up_upgrade", "upgrade_item", None),
    ("close", "close", NORMAL),
]

STATE_RULES = {
    NORMAL: NORMAL_RULES,
    RENOVATE: RENOVATE_RULES,
    DELIVERY: DELIVERY_RULES,
    HELPER: HELPER_RULES,
    UPGRADE: UPGRADE_RULES,

    # FOOD e NEW_POINT têm espera própria depois do press,
    # então são tratados em handler dedicado. As categorias
    # ficam aqui só para o filtro do detector.
    FOOD: [
        ("up_food", "upgrade_food", None),
        ("gray_max", "gray_max", None),
        ("gray_coin", "gray_coin", None)
    ],
    NEW_POINT: [
        ("up_food", "new_point_click", None),
    ],
}


class StateMachine:

    # Mantidos como atributos de classe por compatibilidade
    # com quem referenciava StateMachine.NORMAL.
    NORMAL = NORMAL
    RENOVATE = RENOVATE
    FOOD = FOOD
    NEW_POINT = NEW_POINT
    UPGRADE = UPGRADE
    DELIVERY = DELIVERY
    HELPER = HELPER

    def __init__(self, action_manager, recorder=None, detector=None):

        self.action_manager = action_manager

        # Gravador do dataset de treino. None = não grava.
        self.recorder = recorder
        self.detector = detector

        self.state = NORMAL
        self.state_entered = time.monotonic()

        self.last_action_time = 0.0
        self.action_cooldown = ACTION_COOLDOWN
        self.action_settle = ACTION_SETTLE

        # Espera própria do swipe, maior que a de um toque: um
        # toque mexe um painel, um swipe move a VISTA INTEIRA e
        # o jogo segue deslizando por inércia.
        self.swipe_settle = SWIPE_WAITING_TIME

        # A última ação foi um swipe?
        #
        # Guardado como BANDEIRA, e não como uma cópia do valor
        # do settle, de propósito: assim `machine.action_settle
        # = 0` continua desligando a espera de toque (é o que os
        # testes fazem) sem que exista um segundo número
        # desatualizado em algum lugar.
        self._last_was_swipe = False

        # Frame atual, guardado para o gravador do dataset.
        self._frame = None
        self._detections = []
        self._lag = 0.0

        # Quando o frame que gerou as detecções em mão foi
        # CAPTURADO (não quando chegou aqui).
        #
        # Sem isso o bot age duas vezes sobre a mesma tela: o
        # cooldown libera antes de existir um frame que mostre
        # o efeito da primeira ação. Ver _can_act.
        self._frame_time = None

        # Quantas vezes esperou frame novo, para não encher o
        # log com uma linha por quadro.
        self._waited_warned = 0.0

        # Espera depois do long press de comida.
        self.up_food_wait_start = None

        # Item de "food" atual, esperando up_food. Guardado aqui (e
        # não só na detecção local) porque _wait_or_give_up precisa
        # dele para saber QUAL item ignorar ao desistir.
        self._food_target = None

        # Itens de "food" que esgotaram UP_FOOD_WAIT em
        # FOOD_MAX_ATTEMPTS tentativas seguidas — ignorados até o
        # próximo swipe real, senão o bot tenta o mesmo item para
        # sempre (achar sempre adia o swipe).
        self._ignored_food = []

        # Última desistência (item + quantas vezes seguidas), para
        # saber se a PRÓXIMA é no mesmo item ou um recomeço.
        self._food_fail_target = None
        self._food_fail_count = 0

        # Conta do start, não do primeiro build: o primeiro
        # trecho também é informação.
        self.cycle_at = time.monotonic()
        self.cycle_last = None
        self.cycle_count = 0

        # Contado em _act (único ponto por onde ação sai);
        # contar em _apply_rules incluiria tentativa barrada por
        # cooldown/worker ocupado. Por categoria e por ação
        # porque são perguntas diferentes: `fly` é categoria
        # (voos), `upgrade_item` é ação (cliques de upgrade).
        self.category_counts = Counter()
        self.action_counts = Counter()

        self.last_action = None
        self.last_action_at = None

        # Ritmo normal: um swipe a cada tanto, enquanto não
        # aparecer nada.
        self.exploration_delay = EXPLORATION_DELAY

        # Depois de ACHAR algo, a exploração é ADIADA por este
        # tanto — não cancelada.
        self.exploration_action_delay = (
            EXPLORATION_DELAY_AFTER_ACTION
        )

        # De quando se conta a espera do próximo swipe: o
        # último swipe, ou a última vez que algo foi achado.
        self.explore_anchor = time.monotonic()

        # O âncora veio de um "achou"?
        #
        # Guardado como BANDEIRA, e o intervalo derivado dela na
        # hora — não uma cópia do número. É o que mantém
        # `machine.exploration_delay = 0` funcionando (os testes
        # usam isso) sem existir um segundo valor guardado que
        # possa ficar desatualizado. Mesmo padrão do
        # _last_was_swipe.
        self.explore_found = False

        # Ciclo de varredura: 5 swipes para um lado, 5 para o
        # outro, indefinidamente. A contagem NÃO zera quando o
        # bot acha algo — zerava antes, e o efeito era varrer
        # sempre o mesmo pedaço da tela (quase todo swipe revela
        # algum alvo, então a volta nunca fechava e a metade
        # distante do restaurante nunca era visitada).
        self.swipe_direction = SWIPE_START_DIRECTION
        self.swipe_count = 0
        self.max_swipes = MAX_SWIPES

        # Evita repetir o aviso de detecção velha a cada frame.
        self._stale_warned = 0.0

        # Detecta ação que repete sem resolver nada.
        self._last_action = None
        self._repeat_count = 0

        # Presses consecutivos de upgrade_food em NORMAL (lê
        # _trap_guard: painel que nunca resolve cai para dismiss).
        self._up_food_repeats = 0

        # Updates consecutivos com up_food visível sem o painel
        # resolver (lê update: popup travado, ex. preço
        # inalcançável) + supressão ativa de up_food.
        self._up_food_streak = 0
        self._up_food_suppressed = 0

        # Tentativas consecutivas de fechar algo. Sobe a escada
        # de escalonamento em vez de repetir o que não funcionou.
        self._dismiss_attempts = 0

        # Quantas vezes já precisou rolar para escapar.
        self._dismiss_rounds = 0

    def wanted_categories(self):
        """Categorias do estado atual, em ordem de prioridade — filtro do detector (~264ms → ~16ms por passada em UPGRADE)."""

        rules = STATE_RULES.get(self.state, NORMAL_RULES)

        categories = []

        for category, _, _ in rules:

            if category not in categories:

                categories.append(category)

        # Fora de NORMAL, ainda queremos ver o "X" (saída de
        # emergência de qualquer modal). No FIM da lista: é rede
        # de segurança, não prioridade — onde importa de verdade
        # ele já está nas regras, na posição certa.
        if "close" not in categories:

            categories.append("close")

        return categories

    def update(
        self,
        detections,
        lag=0.0,
        frame=None,
        frame_time=None,
    ):

        # Antes de qualquer return: é aqui que a ação anterior
        # recebe veredito, mesmo nos frames em que o bot não age.
        if self.recorder is not None:

            self._frame = frame

            self._detections = detections

            self.recorder.observe(
                frame,
                detections,
                frame_time,
                self.state,
            )

        # lag é a idade do frame; guardado antes de qualquer
        # coisa porque todo caminho de ação passa por _can_act.
        self._frame_time = time.monotonic() - lag

        self._lag = lag

        if self._timed_out():
            return

        # Detecção velha: clicar com base num frame antigo
        # acerta onde o objeto ESTAVA.
        if (
            MAX_DETECTION_AGE is not None
            and lag > MAX_DETECTION_AGE
        ):

            now = time.monotonic()

            if now - self._stale_warned > 5.0:

                logger.warning(
                    "Detections %.0f ms late "
                    "(max %.0f ms) — will not click. "
                    "Detector too slow?",
                    lag * 1000,
                    MAX_DETECTION_AGE * 1000,
                )

                self._stale_warned = now

            return

        if self.state == FOOD:

            self._food(detections)

        elif self.state == NEW_POINT:

            self._new_point(detections)

        else:

            # Painel sumiu sozinho: a contagem recomeça do zero
            # na próxima vez que o up_food aparecer.
            if self.state == NORMAL:
                if any(
                    d.get("category") == "up_food"
                    for d in detections or []
                ):
                    self._up_food_streak += 1
                    if (
                        self._up_food_streak >= UPGRADE_FOOD_STUCK_UPDATES
                        and self._up_food_suppressed <= 0
                    ):
                        self._up_food_suppressed = (
                            UPGRADE_FOOD_SUPPRESS_UPDATES
                        )
                        self._up_food_streak = 0
                        logger.warning(
                            "up_food stuck open for %d updates "
                            "in NORMAL (unresolvable panel?) — "
                            "suppressing up_food for %d updates "
                            "so other upgrades get a turn",
                            UPGRADE_FOOD_STUCK_UPDATES,
                            UPGRADE_FOOD_SUPPRESS_UPDATES,
                        )
                else:
                    self._up_food_repeats = 0
                    self._up_food_streak = 0
                    self._up_food_suppressed = 0
                if self._up_food_suppressed > 0:
                    self._up_food_suppressed -= 1
                    if self._up_food_suppressed == 0:
                        # Supressão esgotada: up_food ganha uma
                        # janela nova (streak zerado) antes de
                        # poder ser suprimido de novo — alterna
                        # entre o painel e os outros upgrades.
                        self._up_food_streak = 0

            acted = self._apply_rules(
                detections,
                STATE_RULES.get(self.state, NORMAL_RULES),
            )

            # Só NORMAL explora: nos outros estados estamos
            # dentro de um painel, e swipe atrapalharia.
            if self.state == NORMAL and (
                not SWIPE_WAIT_FOR_NO_ACTION or not acted
            ):

                self._explore_screen()

    def _apply_rules(self, detections, rules):
        """Executa a 1ª regra cuja categoria apareça; devolve True se achou algo (mesmo que a ação seja barrada por cooldown), pois achar já significa que não estamos perdidos."""

        for prioridade, (category, action, next_state) in enumerate(
            rules,
            start=1,
        ):

            # Supressão ativa: pula up_food para que regras de
            # prioridade menor (upgrades de outras estações)
            # tenham a vez. A detecção continua rodando — é o
            # que conta o streak e percebe quando o painel some.
            if (
                self.state == NORMAL
                and category == "up_food"
                and self._up_food_suppressed > 0
            ):
                continue

            detection = self._find(detections, category)

            if detection is None:
                continue

            # Em DEBUG, mostra qual prioridade venceu e o que
            # estava na tela junto. É o que responde "por que
            # clicou nisso e não naquilo".
            if logger.isEnabledFor(logging.DEBUG):

                na_tela = " ".join(
                    f"{d['category']}({d['confidence']:.2f})"
                    for d in detections
                )

                logger.debug(
                    "%s prio %d/%d: %s -> %s | on screen: %s",
                    self.state,
                    prioridade,
                    len(rules),
                    category,
                    action,
                    na_tela or "nothing",
                )

            if action != "food":

                self._delay_exploration()

            self._act(
                action,
                detection,
                next_state,
            )

            return True

        return False

    def _act(self, action, detection, next_state=None):
        """Único lugar do projeto que dispara ação e troca estado."""

        if not self._can_act():
            return False

        action, detection = self._trap_guard(action, detection)

        action = self._escalate(action)

        if not self.action_manager.execute(action, detection):

            # Worker ocupado: tenta no próximo frame.
            return False

        confidence = (
            detection["confidence"]
            if detection
            else 0.0
        )

        logger.info(
            "%s: %s (F=%.3f)",
            self.state,
            action,
            confidence,
        )

        self.last_action_time = time.monotonic()

        # `scroll_bottom` é seis swipes seguidos: move a vista
        # mais que qualquer swipe solto, e por isso merece a
        # espera longa igual. Um toque comum fica na curta.
        self._last_was_swipe = action in VIEW_MOVING_ACTIONS

        # Swipe real: itens ignorados podem estar em posição
        # diferente agora, ou terem saído de tela. Vale a pena
        # tentar de novo.
        if self._last_was_swipe:
            self._reset_food_tracking()

        # Mesmo raciocínio do ciclo, abaixo: só conta o que
        # realmente saiu.
        self.action_counts[action] += 1

        if detection:
            self.category_counts[detection["category"]] += 1

        self.last_action = action
        self.last_action_at = self.last_action_time

        # Aqui, e não em _apply_rules: este é o ponto em que a
        # ação SAIU de verdade. Marcar antes contaria ciclo em
        # tentativa barrada por cooldown ou worker ocupado.
        if detection and detection["category"] in CYCLE_CATEGORIES:

            self._mark_cycle(detection["category"])

        self._count_repeat(action)

        if self.detector is not None and detection is not None:
            self.detector.learn_action(self._frame, detection)

        self._record(action, detection)

        if next_state is not None and next_state != self.state:

            self._enter(next_state)

        # Depois do _enter (que não toca em _food_target): guarda
        # QUAL item entrou em FOOD, para _wait_or_give_up saber o
        # que ignorar se ele desistir.
        if action == "food":
            self._food_target = detection

        return True

    def _record(self, action, detection):
        """Entrega ao gravador a ação que ACABOU de sair; chamado de dentro de _act pois é o único ponto onde se sabe que ela não foi barrada por cooldown ou worker ocupado."""

        if self.recorder is None:
            return

        if self._frame is None:
            return

        self.recorder.on_action(
            action=action,

            # swipe_* não está em ACTION_TABLE: é a exploração,
            # que chama o ActionManager por outro caminho.
            action_kind=(
                "swipe"
                if action.startswith("swipe_")
                else self.action_manager.kind_of(action)
            ),
            detection=detection,

            # Ponto tocado no espaço da IMAGEM gravada.
            # None para o swipe: não é um toque pontual.
            click=(
                None
                if action.startswith("swipe_")
                else self.action_manager.target_frame(
                    action,
                    detection,
                )
            ),

            state=self.state,
            frame=self._frame,
            detections=self._detections,
            lag=self._lag,
            cycle_count=self.cycle_count,
        )

    def _mark_cycle(self, category):

        agora = time.monotonic()

        duracao = agora - self.cycle_at

        self.cycle_count += 1

        logger.info(
            "Cycle %d closed by '%s' in %s%s",
            self.cycle_count,
            category,
            formata_duracao(duracao),
            (
                f" (previous {formata_duracao(self.cycle_last)})"
                if self.cycle_last is not None
                else " (counted since start)"
            ),
        )

        self.cycle_last = duracao
        self.cycle_at = agora

    def cycle_stats(self):
        """(tempo_corrido, ultimo_ciclo, quantos) para o HUD; ultimo_ciclo é None até o primeiro build/plane."""

        return (
            time.monotonic() - self.cycle_at,
            self.cycle_last,
            self.cycle_count,
        )

    def summary(self):
        """O que interessa a quem olha a tela, num dict — nomes de negócio (voos, reformas), não de código."""

        return {
            "estado": self.state,
            "acao": self.last_action,
            "acao_ha": (
                None
                if self.last_action_at is None
                else time.monotonic() - self.last_action_at
            ),

            # `fly` é o botão do avião na tela de reforma; é ele
            # que efetivamente muda de restaurante.
            "voos": self.category_counts.get("fly", 0),

            # `build` e `plane` são as duas portas para RENOVATE
            # (CYCLE_CATEGORIES). Somadas, é quantas vezes o bot
            # entrou numa reforma; `cycle_count` conta o mesmo
            # evento, e serve de conferência.
            "reformas": sum(
                self.category_counts.get(categoria, 0)
                for categoria in CYCLE_CATEGORIES
            ),

            "acoes": sum(self.action_counts.values()),
        }

    def _trap_guard(self, action, detection):
        """NORMAL só: up_food que sobrevive a UPGRADE_FOOD_MAX_REPEATS presses cai para "dismiss" (HOLD em ponto fixo, sem detecção) por UPGRADE_FOOD_DISMISS_ROUNDS rodadas; depois o ciclo recomeça e o press ganha outra chance. Qualquer outra ação zera a contagem."""

        if self.state != NORMAL or action != "upgrade_food":
            self._up_food_repeats = 0

            return action, detection

        self._up_food_repeats += 1

        if self._up_food_repeats <= UPGRADE_FOOD_MAX_REPEATS:
            return action, detection

        if self._up_food_repeats <= (
            UPGRADE_FOOD_MAX_REPEATS + UPGRADE_FOOD_DISMISS_ROUNDS
        ):
            logger.warning(
                "upgrade_food (up_food) survived %d presses in NORMAL — "
                "dismissing instead (round %d/%d)",
                UPGRADE_FOOD_MAX_REPEATS,
                self._up_food_repeats - UPGRADE_FOOD_MAX_REPEATS,
                UPGRADE_FOOD_DISMISS_ROUNDS,
            )

            return "dismiss", None

        self._up_food_repeats = 0

        return action, detection

    def _escalate(self, action):
        """Escada de fechamento: N tentativas no ponto fixo, depois rola até o fim e tenta de novo — o ponto pode ser o que ABRE o painel, então tocar nele às vezes reabre em vez de fechar; não há BACK neste jogo."""

        if action not in DISMISS_ACTIONS:

            # Ação normal: progresso, zera a escada.
            self._dismiss_attempts = 0
            self._dismiss_rounds = 0

            return action

        self._dismiss_attempts += 1

        if (
            self._dismiss_attempts
            <= DISMISS_ATTEMPTS_BEFORE_SCROLL
        ):
            return action

        # Estourou as tentativas no ponto: rola e reinicia a
        # contagem, para tentar o ponto de novo depois.
        self._dismiss_attempts = 0
        self._dismiss_rounds += 1

        logger.warning(
            "'%s' did not close in %d attempts (round %d) — "
            "scrolling the screen to the end to find a position "
            "where the point is safe",
            action,
            DISMISS_ATTEMPTS_BEFORE_SCROLL,
            self._dismiss_rounds,
        )

        return "scroll_bottom"

    def _count_repeat(self, action):
        """Avisa quando a mesma ação repete sem levar a nada (sintoma típico: ponto de dispensa que não fecha o painel)."""

        if action != self._last_action:

            self._last_action = action
            self._repeat_count = 1

            return

        self._repeat_count += 1

        if self._repeat_count % REPEATED_ACTION_WARNING:
            return

        logger.warning(
            "'%s' repeated %d times in a row in %s without "
            "resolving. Wrong dismiss point, or missing "
            "template for what is on screen?",
            action,
            self._repeat_count,
            self.state,
        )

    def frame_floor(self):
        """Instante de CAPTURA mínimo para um frame autorizar ação. Definição única, compartilhada com o VisionWorker (evita divergir e agir sobre tela velha); conta do FIM da ação, não da submissão, pois ações assíncronas variam de duração (swipe 500ms, long press de comida 4s); piso final é o maior entre a espera de entrada no estado e a de settle pós-ação."""

        # Espera de entrada independe da última ação: uma tela
        # que abriu com animação não fica pronta só porque o
        # toque assentou (sem isto, upgrade era fechado ainda na
        # animação, quando só o "X" já casava). O `if` evita que
        # um estado SEM espera de entrada barre por empate o
        # primeiro frame de idade zero (piso == state_entered).
        entrada = STATE_ENTRY_SETTLE.get(self.state, 0.0)

        piso = (
            self.state_entered + entrada
            if entrada
            else 0.0
        )

        if not self.last_action_time:
            return piso

        base = max(
            self.last_action_time,
            getattr(
                self.action_manager,
                "last_finished_at",
                0.0,
            ),
        )

        settle = (
            self.swipe_settle
            if self._last_was_swipe
            else self.action_settle
        )

        return max(piso, base + settle)

    def _can_act(self):

        # Ação em andamento (ex.: long press de 4 s) precisa
        # terminar antes da próxima.
        if self.action_manager.is_busy():
            return False

        agora = time.monotonic()

        if agora - self.last_action_time < self.action_cooldown:
            return False

        # O cooldown não basta: com o detector atrasado, a
        # detecção em mão pode vir de frame ANTERIOR à última
        # ação (sintoma: duplo toque reabrindo painel — o ponto
        # de dispensa costuma ser também o que abre). A condição
        # certa é causal (frame posterior ao FIM da ação, ver
        # frame_floor), não temporal — um cooldown maior só
        # atrasaria o mesmo problema com um detector mais lento.
        piso = self.frame_floor()

        if (
            self._frame_time is not None
            and piso
            and self._frame_time <= piso
        ):

            if (
                logger.isEnabledFor(logging.DEBUG)
                and agora - self._waited_warned > 1.0
            ):

                logger.debug(
                    "%s: waiting for frame showing the effect "
                    "of %s (missing %.0f ms)",
                    self.state,
                    (
                        "a swipe"
                        if self._last_was_swipe
                        else "action"
                    ),
                    (piso - self._frame_time) * 1000,
                )

                self._waited_warned = agora

            return False

        return True

    def _enter(self, state):

        if state == self.state:
            return

        previous_state = self.state

        logger.info(
            "%s -> %s",
            self.state,
            state,
        )

        self.state = state
        self.state_entered = time.monotonic()

        # Variáveis locais do estado anterior não vazam
        # para o próximo.
        self.up_food_wait_start = None

        # Trocar de estado é progresso: as contagens zeram.
        self._last_action = None
        self._repeat_count = 0
        self._up_food_repeats = 0
        self._up_food_streak = 0
        self._up_food_suppressed = 0
        self._dismiss_attempts = 0
        self._dismiss_rounds = 0

        if state != FOOD and previous_state != FOOD:

            self._delay_exploration()

    def _timed_out(self):

        timeout = STATE_TIMEOUTS.get(self.state)

        if timeout is None:
            return False

        elapsed = time.monotonic() - self.state_entered

        if elapsed < timeout:
            return False

        logger.warning(
            "%s stuck for %.1fs — returning to NORMAL",
            self.state,
            elapsed,
        )

        self._enter(NORMAL)

        return True

    def _food(self, detections):
        """Segura o botão de upgrade de comida e espera o próximo aparecer; sem próximo em UP_FOOD_WAIT, volta para NORMAL."""

        up_food = self._find(detections, "up_food")

        if up_food is not None:

            if self._act("upgrade_food", up_food):

                self._delay_exploration()

                self.up_food_wait_start = time.monotonic()

            return

        # Painel esgotado: dispensa (DISMISS_POINT ou
        # GRAY_COIN_POINT). ANTES do _wait_or_give_up, não
        # depois: ele pode chamar _enter(NORMAL), e um _act
        # depois disso agiria fora do estado FOOD.
        for categoria, acao in (
            ("gray_max", "gray_max"),
            ("gray_coin", "gray_coin"),
        ):

            deteccao = self._find(detections, categoria)

            if deteccao is not None:

                # self._delay_exploration()

                self._act(acao, deteccao, NORMAL)

                return

        self._wait_or_give_up()

    def _new_point(self, detections):
        """Igual ao FOOD, mas basta um clique para liberar — não é long press."""

        unlock_food = self._find(detections, "up_food")

        if unlock_food is not None:

            self._delay_exploration()

            if self._act("new_point_click", unlock_food):

                self.up_food_wait_start = time.monotonic()

            return

        self._wait_or_give_up()

    def _wait_or_give_up(self):
        """Tela pode demorar para atualizar após agir; espera UP_FOOD_WAIT antes de desistir."""

        if self.up_food_wait_start is None:
            return

        elapsed = time.monotonic() - self.up_food_wait_start

        if elapsed < UP_FOOD_WAIT:
            return

        logger.info(
            "No up_food in %.1fs",
            UP_FOOD_WAIT,
        )

        if self.state == FOOD and self._food_target is not None:

            self._register_food_failure(self._food_target)
            self._food_target = None

        self._enter(NORMAL)

    def _register_food_failure(self, target):
        """Conta desistências SEGUIDAS no mesmo item (por IoU); ao chegar em FOOD_MAX_ATTEMPTS, ignora até o próximo swipe — antes disso, volta a NORMAL e deixa tentar de novo (pode ter sido só o up_food demorando a aparecer)."""

        if (
            self._food_fail_target is not None
            and self._iou(target, self._food_fail_target)
            >= FOOD_IGNORE_IOU
        ):
            self._food_fail_count += 1

        else:
            self._food_fail_count = 1

        self._food_fail_target = target

        if self._food_fail_count < FOOD_MAX_ATTEMPTS:
            return

        logger.warning(
            "'food' without up_food in %d consecutive attempts — "
            "ignoring until the next swipe",
            FOOD_MAX_ATTEMPTS,
        )

        self._ignored_food.append(target)

        self._food_fail_target = None
        self._food_fail_count = 0

    def _reset_food_tracking(self):
        """Esquece itens ignorados e a contagem de tentativas — chamado a cada swipe real, que muda o que está na tela."""

        self._ignored_food = []

        self._food_fail_target = None
        self._food_fail_count = 0

    def _find(self, detections, category):
        """Primeira detecção da categoria. Sem refiltragem por threshold aqui de propósito: já existiu um segundo threshold neste método, divergente do detector, e o escolhido era o primeiro em ordem alfabética de arquivo, não o melhor."""

        for detection in detections:

            if detection["category"] != category:
                continue

            if category == "food" and self._is_ignored_food(
                detection
            ):
                continue

            return detection

        return None

    def _is_ignored_food(self, detection):
        """Mesmo item de um 'food' que já esgotou UP_FOOD_WAIT sem up_food (mesma posição na tela, comparada por IoU já que a tela não se move fora de um swipe)."""

        return any(
            self._iou(detection, ignored) >= FOOD_IGNORE_IOU
            for ignored in self._ignored_food
        )

    @staticmethod
    def _iou(a, b):

        ax2 = a["x"] + a["width"]
        ay2 = a["y"] + a["height"]

        bx2 = b["x"] + b["width"]
        by2 = b["y"] + b["height"]

        inter_width = min(ax2, bx2) - max(a["x"], b["x"])
        inter_height = min(ay2, by2) - max(a["y"], b["y"])

        if inter_width <= 0 or inter_height <= 0:
            return 0.0

        intersection = inter_width * inter_height

        union = (
            a["width"] * a["height"]
            + b["width"] * b["height"]
            - intersection
        )

        return intersection / union

    def reset(self):

        self._enter(NORMAL)

        self.last_action_time = 0.0

        self._food_target = None
        self._reset_food_tracking()

    def _delay_exploration(self):
        """Algo foi encontrado: ADIA o próximo swipe, sem resetar o ciclo de varredura (renomeado de `_reset_exploration` — resetar a contagem aqui fazia o bot varrer sempre o mesmo pedaço da tela, já que quase todo swipe acha algo)."""

        if not SWIPE_WAIT_FOR_NO_ACTION:
            return

        self.explore_anchor = time.monotonic()

        self.explore_found = True

    def exploration_interval(self):
        """Quanto esperar pelo próximo swipe: exploration_delay se nada foi achado, exploration_action_delay (maior) se sim — derivado de explore_found em vez de guardado."""

        return (
            self.exploration_action_delay
            if self.explore_found
            else self.exploration_delay
        )

    def _explore_screen(self):

        now = time.monotonic()

        if (
            now - self.explore_anchor
            < self.exploration_interval()
        ):
            return

        if not self._can_act():
            return

        logger.info(
            "explore: swipe %s %d/%d",
            self.swipe_direction,
            self.swipe_count + 1,
            self.max_swipes,
        )

        if not self.action_manager.swipe(
            self.swipe_direction
        ):
            return

        self.last_action_time = now

        # Volta ao RITMO normal: a partir daqui o próximo swipe
        # é em exploration_delay. Achar algo é o que troca isso
        # pelos 15 s (_delay_exploration).
        self.explore_anchor = now

        self.explore_found = False

        # A partir daqui vale SWIPE_WAITING_TIME, não
        # ACTION_SETTLE: a vista acabou de se mover inteira, e
        # nenhuma detecção de antes dela parar vale um toque.
        self._last_was_swipe = True

        # Swipe real: dá nova chance aos itens ignorados.
        self._reset_food_tracking()

        # A exploração não passa por _act, então precisa gravar
        # aqui. É decisão do bot como qualquer outra: "não achei
        # nada, rolo a tela" — e para o treino subir e descer são
        # rótulos diferentes, daí a direção ir no NOME.
        self._record(
            f"swipe_{self.swipe_direction}",
            None,
        )

        self.swipe_count += 1

        if self.swipe_count >= self.max_swipes:

            self.swipe_direction = (
                "down"
                if self.swipe_direction == "up"
                else "up"
            )

            self.swipe_count = 0
