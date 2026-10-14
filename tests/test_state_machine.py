"""
State-machine tests. No device needed: the ActionManager is
fake and the clock is controlled, so cooldown, state
timeouts, and priority can be tested without waiting in real time.
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
    CITY_STUCK_RESTART_AFTER,
    CITY_STUCK_SWEEP_AFTER,
    DISMISS_ACTIONS,
    DISMISS_ATTEMPTS_BEFORE_SCROLL,
    EXPLORATION_DELAY,
    EXPLORATION_DELAY_AFTER_ACTION,
    GAME_PACKAGE,
    GRAY_STUCK_UPDATES,
    GRAY_SUPPRESS_UPDATES,
    MAX_DETECTION_AGE,
    REPEATED_ACTION_WARNING,
    STATE_ENTRY_SETTLE,
    STATE_TIMEOUTS,
    SWIPE_WAITING_TIME,
    UPGRADE_FOOD_STUCK_UPDATES,
    UPGRADE_FOOD_SUPPRESS_UPDATES,
    UPGRADE_EMPTY_VISITS,
    UPGRADE_IGNORE_SECONDS,
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

    # Test convenience
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
    """Returns (machine, actions, clock) with cooldown already expired, so the first update can act."""

    clock = Clock()

    time.monotonic = clock.monotonic

    actions = FakeActions()

    machine = sm.StateMachine(actions)

    # Leaves cooldown expired.
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
    """close/open_store/gray_max close what shouldn't be open: priority over plane and food."""

    machine, actions, _ = build()

    machine.update([
        detection("plane"),
        detection("close"),
    ])

    assert actions.actions == ["close"], actions.actions

    # Closing a tab doesn't change state.
    assert machine.state == sm.NORMAL, machine.state


def test_uma_acao_por_cooldown():

    machine, actions, clock = build()

    detections = [detection("box")]

    machine.update(detections)

    # Second frame right after: cooldown blocks.
    machine.update(detections)

    assert len(actions.calls) == 1, actions.calls

    clock.advance(ACTION_COOLDOWN + 0.01)

    machine.update(detections)

    assert len(actions.calls) == 2, actions.calls


def test_acao_em_andamento_bloqueia():
    """The food long press lasts 4s: while it runs, no other click may be queued."""

    machine, actions, _ = build()

    actions.busy = True

    machine.update([detection("food")])

    assert actions.calls == [], actions.calls
    assert machine.state == sm.NORMAL, machine.state


def test_timeout_do_estado_volta_para_normal():
    """Before, a modal without a template stalled the bot forever: RENOVATE only exited on finding the coin."""

    machine, actions, clock = build()

    machine._enter(sm.RENOVATE)

    # Nothing on screen: without a timeout, it would be stuck here.
    machine.update([])

    assert machine.state == sm.RENOVATE, machine.state

    clock.advance(STATE_TIMEOUTS["RENOVATE"] + 0.1)

    machine.update([])

    assert machine.state == sm.NORMAL, machine.state


def test_delivery_e_helper_abrem_painel_e_usam_botao_interno():
    # delivery from NORMAL opens DELIVERY and uses accept_delivery
    machine, actions, clock = build()
    assert "delivery" in machine.wanted_categories()
    machine.update([detection("delivery")])
    assert actions.actions == ["delivery"]
    assert machine.state == sm.DELIVERY
    assert "accept_delivery" in machine.wanted_categories()

    clock.advance(STATE_ENTRY_SETTLE[sm.DELIVERY] + 0.01)
    machine.update([detection("accept_delivery"), detection("close")])
    assert actions.actions == ["delivery", "accept_delivery"]
    assert machine.state == sm.NORMAL

    # helper from DELIVERY opens HELPER and uses invite_helper
    machine, actions, clock = build()
    machine._enter(sm.DELIVERY)
    clock.advance(STATE_ENTRY_SETTLE[sm.DELIVERY] + 0.01)
    assert "helper" in machine.wanted_categories()
    machine.update([detection("helper")])
    assert actions.actions == ["helper"]
    assert machine.state == sm.HELPER
    assert "invite_helper" in machine.wanted_categories()

    clock.advance(STATE_ENTRY_SETTLE[sm.HELPER] + 0.01)
    machine.update([detection("invite_helper"), detection("close")])
    assert actions.actions == ["helper", "invite_helper"]
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

    # Panel opens with an animation: STATE_ENTRY_SETTLE holds back before
    # acting. See test_upgrade_nao_fecha_no_meio_da_animacao.
    clock.advance(STATE_ENTRY_SETTLE[sm.UPGRADE] + 0.01)

    machine.update([detection("up_upgrade")])

    assert actions.actions == ["upgrade_item"], actions.actions
    assert machine.state == sm.UPGRADE, machine.state

    # No more items: closes and returns.
    clock.advance(ACTION_COOLDOWN + 0.01)

    machine.update([detection("close")])

    assert actions.actions[-1] == "close", actions.actions
    assert machine.state == sm.NORMAL, machine.state


def test_upgrade_nao_fecha_no_meio_da_animacao():
    """
    The reported bug: on opening the upgrade screen, the bot closed it right away.
    While the panel slides in, the "X" already matches the template and the
    upgrade buttons don't yet — rule 1 finds nothing, rule 2 finds it, and the bot
    undoes what it just did. ACTION_SETTLE doesn't cover this: it's counted
    from the TAP that opened it, sized for the CLOSE animation.
    """

    machine, actions, clock = build()

    espera = STATE_ENTRY_SETTLE[sm.UPGRADE]

    assert espera > ACTION_SETTLE, (
        "uma espera de entrada menor que a de ação não teria "
        "efeito nenhum"
    )

    machine._enter(sm.UPGRADE)

    # Mid-animation, the "X" is the only thing matching. Past
    # ACTION_SETTLE already, to prove it's the ENTRY wait holding.
    clock.advance(ACTION_SETTLE + 0.01)

    machine.update([detection("close")])

    assert actions.actions == [], (
        "fechou o painel no meio da animação de abrir"
    )

    assert machine.state == sm.UPGRADE, machine.state

    # Panel built: now the upgrade shows and it wins.
    clock.advance(espera)

    machine.update([
        detection("up_upgrade"),
        detection("close"),
    ])

    assert actions.actions == ["upgrade_item"], actions.actions

    assert machine.state == sm.UPGRADE, machine.state


def test_espera_de_entrada_cabe_no_timeout_do_estado():
    """The entry wait eats the time the bot has to act inside the state: same clock."""

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
    `gray_coin` does what `gray_max` does — dismisses and returns to NORMAL. The
    tapped point is the ActionManager's business
    (test_pipeline::test_gray_coin_toca_no_ponto_proprio).
    """

    for categoria in ("gray_max", "gray_coin"):

        machine, actions, _ = build()

        machine.update([detection(categoria)])

        assert actions.actions == [categoria], (
            categoria,
            actions.actions,
        )

        # And in FOOD too: that's where the depleted panel shows.
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
    In FOOD, the dismiss must be evaluated BEFORE _wait_or_give_up, which
    may call _enter(NORMAL). If the dismiss comes after, the action fires
    when the machine ALREADY CONSIDERS itself in NORMAL — deciding by a
    FOOD rule in a state that's no longer FOOD, mislabeling the sample.
    """

    machine, actions, clock = build()

    # Spies the state AT THE MOMENT the action is dispatched.
    estados = []

    original = actions.execute

    def espiao(action, deteccao):

        estados.append(machine.state)

        return original(action, deteccao)

    actions.execute = espiao

    machine._enter(sm.FOOD)

    # Give-up timer ALREADY EXPIRED: the situation where _wait_or_give_up would act.
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

    # No new up_food showed up.
    clock.advance(UP_FOOD_WAIT + 0.1)

    machine.update([])

    assert machine.state == sm.NORMAL, machine.state


def test_food_nao_desiste_no_meio_do_press():
    """Regression for the overnight stall: UPGRADE_FOOD_PRESS (4s) >
    UP_FOOD_WAIT (2s). The async worker runs the press on its own
    thread, so without a busy guard _wait_or_give_up fired 2s into
    the 4s press and bailed to NORMAL mid-upgrade."""

    machine, actions, clock = build()

    machine._enter(sm.FOOD)

    machine.update([detection("up_food")])

    assert actions.actions == ["upgrade_food"], actions.actions

    # Worker still holding the 4s press: the wait clock freezes.
    # (Advance past UP_FOOD_WAIT but under the FOOD state timeout,
    # so only _wait_or_give_up is under test — not _timed_out.)
    actions.busy = True
    clock.advance(UP_FOOD_WAIT + 0.5)

    machine.update([])

    assert machine.state == sm.FOOD, machine.state

    # Press finished, still no next up_food: now it may give up.
    actions.busy = False
    clock.advance(UP_FOOD_WAIT + 0.1)

    machine.update([])

    assert machine.state == sm.NORMAL, machine.state


def test_food_sem_badge_ainda_toca_mas_desiste_e_ignora():
    """Prefer-with-fallback badge gate: food WITHOUT a badge is still
    tapped (badge templates miss on some views — a strict gate blinded
    the bot entirely). The tap MUST arm the give-up timer on entry and
    the timeout path MUST register a failure, so after
    GIVE_UP_MAX_ATTEMPTS the dish is ignored instead of tapped forever
    (the "FOOD stuck" infinite loop)."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    dish = detection("food")

    # 1. Entry taps the dish AND arms the give-up timer.
    machine.update([dish])
    assert machine.state == sm.FOOD, machine.state
    assert machine.up_food_wait_start is not None
    assert "food" in actions.actions, actions.actions

    # 2. up_food never shows up: give-up fires UP_FOOD_WAIT after the
    # tap (not the 5s state timeout) and counts the first failure.
    clock.advance(UP_FOOD_WAIT + 0.1)
    machine.update([])
    assert machine.state == sm.NORMAL, machine.state
    assert machine._give_up["food"]["fail_count"] == 1, machine._give_up

    # 3. Two more identical cycles -> dish ignored, no 4th tap.
    for _ in range(2):
        machine.update([dish])
        assert machine.state == sm.FOOD, machine.state
        clock.advance(UP_FOOD_WAIT + 0.1)
        machine.update([])

    taps = [a for a in actions.actions if a == "food"]

    assert len(taps) == 3, actions.actions

    machine.update([dish])

    assert machine.state == sm.NORMAL, machine.state
    assert [a for a in actions.actions if a == "food"] == taps


def test_red_arrow_fallback_taps_badge_when_food_missing():
    """Badge with no dish (occlusion, e.g. pet over it): taps the
    badge center as a dish proxy, arms the give-up timer, and after
    GIVE_UP_MAX_ATTEMPTS the badge is ignored instead of tapped forever."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    badge = detection("red_arrow", x=300, y=400)

    machine.update([badge])
    assert machine.state == sm.FOOD, machine.state
    assert actions.actions == ["food"], actions.actions
    assert machine._panel_target is not None
    assert machine._panel_target.get("fallback") == "red_arrow"
    assert (machine._panel_target["x"], machine._panel_target["y"]) == (300, 400)

    clock.advance(UP_FOOD_WAIT + 0.1)
    machine.update([])
    assert machine.state == sm.NORMAL, machine.state

    for _ in range(2):
        machine.update([badge])
        assert machine.state == sm.FOOD, machine.state
        clock.advance(UP_FOOD_WAIT + 0.1)
        machine.update([])

    taps = [a for a in actions.actions if a == "food"]
    assert len(taps) == 3, actions.actions

    # Food gave up on the badge: the new_point row now claims it
    # (plot-unlock handoff), so no 4th food tap happens.
    machine.update([badge])
    assert machine.state == sm.NEW_POINT, machine.state
    assert len([a for a in actions.actions if a == "food"]) == 3


def test_red_arrow_fallback_skipped_when_food_present():
    """Real dish wins over the badge proxy: no synthetic detection."""

    machine, actions, _ = build()

    machine.update([detection("food"), detection("red_arrow")])

    assert actions.actions == ["food"], actions.actions
    assert machine._panel_target.get("fallback") is None


def test_red_arrow_fallback_disabled_taps_nothing():
    """Flag off: badge alone fires no food tap."""

    machine, actions, _ = build()

    previous = sm.RED_ARROW_FALLBACK_ENABLED
    sm.RED_ARROW_FALLBACK_ENABLED = False

    try:
        machine.update([detection("red_arrow")])
    finally:
        sm.RED_ARROW_FALLBACK_ENABLED = previous

    assert actions.actions == [], actions.actions
    assert machine.state == sm.NORMAL, machine.state


def test_red_arrow_fallback_retries_ignored_food_as_new_point():
    """Badge food gave up on (3 fails): the new_point row claims it
    as a station proxy instead of stalling on the blocked plot."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    badge = detection("red_arrow", x=300, y=400)

    for _ in range(3):
        machine.update([badge])
        assert machine.state == sm.FOOD, machine.state
        clock.advance(UP_FOOD_WAIT + 0.1)
        machine.update([])
        assert machine.state == sm.NORMAL, machine.state

    assert len([a for a in actions.actions if a == "food"]) == 3

    machine.update([badge])

    assert actions.actions[-1] == "new_point", actions.actions
    assert machine.state == sm.NEW_POINT, machine.state
    assert machine._panel_target["category"] == "new_point"
    assert machine._panel_target.get("fallback") == "red_arrow"


def test_red_arrow_fallback_new_point_ignores_after_failures():
    """Badge that fails as new_point too is ignored after
    GIVE_UP_MAX_ATTEMPTS instead of tapped forever."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    badge = detection("red_arrow", x=300, y=400)

    for _ in range(3):
        machine.update([badge])
        clock.advance(UP_FOOD_WAIT + 0.1)
        machine.update([])

    # Fallback taps arm the give-up timer at tap time, so one
    # empty window per cycle is enough (same as food).
    for _ in range(3):
        machine.update([badge])
        assert machine.state == sm.NEW_POINT, machine.state
        clock.advance(UP_FOOD_WAIT + 0.1)
        machine.update([])
        assert machine.state == sm.NORMAL, machine.state

    taps = [a for a in actions.actions if a == "new_point"]
    assert len(taps) == 3, actions.actions

    machine.update([badge])

    assert machine.state == sm.NORMAL, machine.state
    assert len([a for a in actions.actions if a == "new_point"]) == 3


def test_real_new_point_wins_over_fallback():
    """Visible slot beats the badge proxy: no synthetic detection."""

    machine, actions, _ = build()

    machine.update([detection("new_point"), detection("red_arrow")])

    assert actions.actions == ["new_point"], actions.actions
    assert machine._panel_target.get("fallback") is None


def test_find_skips_ignored_new_point():
    """An ignored slot is not returned even when directly visible."""

    machine, _, clock = build()

    slot = detection("new_point", x=500, y=600)

    assert machine._find([slot], "new_point") is slot

    machine._give_up["new_point"]["ignored"].append((slot, clock.now))

    assert machine._find([slot], "new_point") is None


def test_upgrade_vazio_tres_vezes_ignora_botao():
    """Empty Upgrades panel ("You have all upgrades!") 3 visits in a
    row: the button is permanent nav chrome, so reopening it is pure
    spam. The 4th sighting is skipped without entering UPGRADE."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    for _ in range(UPGRADE_EMPTY_VISITS):
        machine.update([detection("upgrade")])
        assert machine.state == sm.UPGRADE, machine.state
        # Past the UPGRADE entry settle (real panels animate open;
        # same-instant frames predate the entry tap anyway).
        clock.advance(1.1)
        machine.update([detection("close")])
        assert machine.state == sm.NORMAL, machine.state
        clock.advance(0.1)

    assert (
        actions.actions == ["upgrade", "close"] * UPGRADE_EMPTY_VISITS
    ), actions.actions

    machine.update([detection("upgrade")])

    assert machine.state == sm.NORMAL, machine.state
    assert (
        actions.actions == ["upgrade", "close"] * UPGRADE_EMPTY_VISITS
    ), actions.actions


def test_upgrade_com_item_zera_visitas_vazias():
    """A visit that actually buys (upgrade_item tap) resets the empty
    counter: 1 buying visit + 2 empty ones must NOT ignore the button."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    # Buying visit.
    machine.update([detection("upgrade")])
    clock.advance(1.1)
    machine.update([detection("up_upgrade")])
    assert "upgrade_item" in actions.actions, actions.actions
    clock.advance(0.1)
    machine.update([detection("close")])
    assert machine.state == sm.NORMAL, machine.state

    # Two empty visits: not enough to ignore.
    for _ in range(UPGRADE_EMPTY_VISITS - 1):
        clock.advance(0.1)
        machine.update([detection("upgrade")])
        assert machine.state == sm.UPGRADE, machine.state
        clock.advance(1.1)
        machine.update([detection("close")])
        assert machine.state == sm.NORMAL, machine.state
        clock.advance(0.1)

    before = list(actions.actions)
    machine.update([detection("upgrade")])

    assert machine.state == sm.UPGRADE, machine.state
    assert actions.actions == before + ["upgrade"], actions.actions


def test_upgrade_ignore_expira():
    """Ignoring is time-boxed: after UPGRADE_IGNORE_SECONDS the button
    gets another chance (new content may have arrived)."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    for _ in range(UPGRADE_EMPTY_VISITS):
        machine.update([detection("upgrade")])
        clock.advance(1.1)
        machine.update([detection("close")])
        clock.advance(0.1)

    machine.update([detection("upgrade")])
    assert machine.state == sm.NORMAL, machine.state

    # Time-box expired (set directly: advancing the clock 30 min
    # would trip the city-stuck recovery instead).
    machine._upgrade_ignored_until = clock.now - 1.0
    clock.advance(0.1)
    machine.update([detection("upgrade")])

    assert machine.state == sm.UPGRADE, machine.state


def test_cycle_limpa_upgrade_ignore():
    """Renovation door (new city coming) clears the ignore: a new city
    means new general upgrades to buy."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    for _ in range(UPGRADE_EMPTY_VISITS):
        machine.update([detection("upgrade")])
        clock.advance(1.1)
        machine.update([detection("close")])
        clock.advance(0.1)

    machine.update([detection("upgrade")])
    assert machine.state == sm.NORMAL, machine.state

    machine._mark_cycle("plane")

    clock.advance(0.1)
    machine.update([detection("upgrade")])
    assert machine.state == sm.UPGRADE, machine.state


def test_food_sem_up_food_nao_adia_a_exploracao():

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0
    machine.swipe_count = 2

    machine.update([detection("food"), detection("red_arrow")])

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

    machine.update([detection("food"), detection("red_arrow")])
    clock.advance(STATE_TIMEOUTS["FOOD"] + 0.1)
    machine.update([])

    assert machine.state == sm.NORMAL, machine.state
    assert not machine.explore_found

    machine.update([detection("food"), detection("red_arrow")])

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

    # Doesn't pin WHICH direction starts (it's a config tweak):
    # verifies all went the same way.
    primeira = swipes[0][1]

    assert all(
        call[1] == primeira for call in swipes
    ), swipes

    # Past the limit: flips.
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
    Finding something POSTPONES the swipe, doesn't cancel it.

    Before, finding reset exploration, so the swipe only fired
    after 5 s of a COMPLETELY empty screen. Now the swipe is
    periodic and what changes is the interval:

        found nothing ..... every EXPLORATION_DELAY                (5 s)
        found something ... waits EXPLORATION_DELAY_AFTER_ACTION
                            (15 s) then back to the 5 s rhythm
    """

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    clock.advance(machine.exploration_delay + 0.1)

    machine.update([detection("box")])

    # On the frame where it found, no scroll.
    assert not swipes_de(actions), actions.calls

    # The interval in force became the long one.
    assert machine.exploration_interval() == (
        EXPLORATION_DELAY_AFTER_ACTION
    )

    # Past the SHORT interval, still not: it's postponed.
    clock.advance(machine.exploration_delay + 0.1)

    machine.update([])

    assert not swipes_de(actions), (
        "rolou antes do intervalo longo — o adiamento não "
        "está valendo"
    )

    # Past the LONG one, scrolls.
    clock.advance(
        EXPLORATION_DELAY_AFTER_ACTION
        - machine.exploration_delay
    )

    machine.update([])

    assert len(swipes_de(actions)) == 1, actions.calls

    # And back to the short rhythm.
    assert machine.exploration_interval() == (
        machine.exploration_delay
    )

    clock.advance(machine.exploration_delay + 0.1)

    machine.update([])

    assert len(swipes_de(actions)) == 2, actions.calls


def test_achar_algo_nao_zera_o_ciclo_de_varredura():
    """
    The cycle is 5 one way and 5 the other, and finding something in
    the middle does NOT reset the count.

    It used to reset, and the effect was the bot always sweeping the same
    screen slice: since almost every swipe reveals some target, the
    count went back to zero before the return leg closed and the far
    half of the restaurant was never visited.
    """

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    # Two swipes, to sit in the MIDDLE of the leg.
    for _ in range(2):

        clock.advance(machine.exploration_delay + 0.1)

        machine.update([])

    assert machine.swipe_count == 2, machine.swipe_count

    contagem = machine.swipe_count
    direcao = machine.swipe_direction

    # Finds something.
    #
    # `box` on purpose: its rule is (box, open_box, None),
    # i.e. it STAYS in NORMAL. With `food` the machine would leave
    # for the FOOD state, where exploration doesn't even run — and the test
    # would measure something else.
    clock.advance(0.1)

    machine.update([detection("box")])

    assert machine.swipe_count == contagem, (
        f"a contagem voltou para {machine.swipe_count} — o "
        f"ciclo zerou ao achar algo"
    )

    assert machine.swipe_direction == direcao

    # The next swipe CONTINUES the leg.
    clock.advance(EXPLORATION_DELAY_AFTER_ACTION + 0.1)

    machine.update([])

    assert machine.swipe_count == contagem + 1, (
        machine.swipe_count
    )

    assert machine.swipe_direction == direcao


def test_varredura_fecha_a_volta_e_inverte_indefinidamente():
    """
    5 one way, 5 the other, nonstop — even
    when the bot finds something every leg, which is the real case.
    """

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    for volta in range(12):

        # Finds something between one swipe and the next, always.
        #
        # `box` stays in NORMAL (see the test above).
        clock.advance(0.1)

        machine.update([detection("box")])

        clock.advance(EXPLORATION_DELAY_AFTER_ACTION + 0.1)

        machine.update([])

    direcoes = [call[1] for call in swipes_de(actions)]

    assert len(direcoes) == 12, direcoes

    # Blocks of max_swipes in the same direction, alternating.
    primeira = direcoes[0]

    esperado = [
        primeira
        if (i // machine.max_swipes) % 2 == 0
        else ("up" if primeira == "down" else "down")
        for i in range(12)
    ]

    assert direcoes == esperado, (direcoes, esperado)


def test_categorias_do_estado_sao_reduzidas():
    """Makes the detector pass drop from ~264ms to ~16ms inside the upgrade screen."""

    machine, _, _ = build()

    normal = machine.wanted_categories()

    machine._enter(sm.UPGRADE)

    upgrade = machine.wanted_categories()

    assert "food" in normal
    assert "food" not in upgrade, upgrade
    assert list(upgrade) == ["up_upgrade", "close"], upgrade


def test_categorias_vem_na_ordem_da_prioridade():
    """
    ORDER is contract: the detector stops at the first category it
    finds, so a shuffled order would make the bot act on the wrong rule
    — and `food` (2/3 of templates) would stop being last.
    """

    machine, _, _ = build()

    categorias = list(machine.wanted_categories())

    esperada = [
        # Badge first: the detector stops at the first matching
        # category, so anything after `food` would never be searched
        # on a food screen — and the food gate would block every tap.
        "red_arrow",
        *(
            category
            for category, _, _ in sm.NORMAL_RULES
        ),
    ]

    assert categorias == esperada, categorias

    # No repeats: a repeated category would mean a template
    # searched twice in the same pass.
    assert len(categorias) == len(set(categorias)), categorias

    # And the gray pair sits dead last: dimmed price chips are main-view
    # chrome a tap never resolves, so above the real actions they would
    # starve the city — while `up_food` stays last of the ACTIONABLE
    # categories (its template also matches the main-view cost button,
    # so at high priority it would hijack NORMAL).
    assert categorias[-2:] == ["gray_max", "gray_coin"], categorias
    assert categorias[-3] == "up_food", categorias
    assert categorias[-4] == "food", categorias


def test_estado_nao_vaza_variavel():
    """up_food_wait_start was shared between FOOD and NEW_POINT."""

    machine, _, clock = build()

    machine._enter(sm.FOOD)

    machine.update([detection("up_food")])

    assert machine.up_food_wait_start is not None

    machine._enter(sm.NEW_POINT)

    assert machine.up_food_wait_start is None


def test_up_food_em_normal_evolui_a_comida():
    """
    up_food alone in NORMAL still fires the evolve long press (fallback
    for an accidentally opened panel), but only as LAST resort — any
    real game action (box/food/...) wins first.
    """

    machine, actions, _ = build()

    machine.update([detection("up_food")])

    assert actions.actions == ["upgrade_food"], actions.actions

    # The rule declares no next state: stays in NORMAL.
    assert machine.state == sm.NORMAL, machine.state


def test_up_food_faz_o_mesmo_em_normal_e_em_food():
    """up_food alone fires upgrade_food in both states (fallback parity). With company in NORMAL, real actions win — see test_up_food_vence_acao_de_jogo."""

    normal, acoes_normal, _ = build()

    normal.update([detection("up_food")])

    assert acoes_normal.actions == ["upgrade_food"]

    machine, actions, _ = build()

    machine._enter(sm.FOOD)

    machine.update([detection("up_food")])

    assert actions.actions == ["upgrade_food"], actions.actions


def test_up_food_perde_para_o_close():
    """The X closes the panel for free, so it comes before the evolve long press."""

    machine, actions, _ = build()

    machine.update([
        detection("up_food"),
        detection("close"),
    ])

    assert actions.actions == ["close"], actions.actions


def test_up_food_vence_acao_de_jogo():
    """Real game actions beat the demoted up_food: the up_food template also matches the main-view cost button, so box/food win in NORMAL (FOOD state still handles the panel)."""

    machine, actions, _ = build()

    machine.update([
        detection("food"),
        detection("up_food"),
        detection("box"),
    ])

    assert actions.actions == ["open_box"], actions.actions


def test_acao_repetida_gera_aviso():
    """NORMAL has no timeout: a rule firing without resolving would repeat forever in silence."""

    import logging

    machine, actions, clock = build()

    avisos = []

    class Coletor(logging.Handler):

        def emit(self, record):

            if record.levelno >= logging.WARNING:

                avisos.append(record.getMessage())

    handler = Coletor()

    sm.logger.addHandler(handler)

    # The runner runs at ERROR; without lowering the level here, the
    # warning is dropped before reaching the handler.
    nivel_original = sm.logger.level

    sm.logger.setLevel(logging.WARNING)

    try:

        # The panel never closes: up_food stays on screen.
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

    # Different action: the count restarts.
    machine.update([detection("box")])

    assert machine._repeat_count == 1, machine._repeat_count

    assert machine._last_action == "open_box", (
        machine._last_action
    )


def test_max_preso_rola_a_tela_e_tenta_de_novo():
    """
    The scenario: the fixed point OPENS a panel, it shows "max", the
    gray_max rule taps the same point and reopens. The way out isn't BACK (in
    this game it QUITS THE GAME): it's scrolling until the bottom corner is empty and
    trying again.
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

    # And it doesn't stop at the first escape: the cycle repeats.
    assert actions.actions.count("scroll_bottom") == 2, (
        actions.actions
    )


def test_nunca_usa_back():
    """If someone reintroduces BACK, this test warns before the bot closes the game on its own."""

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
    The close-down ladder only applies to DISMISS_ACTIONS, and "upgrade_food"
    isn't one of them — up_food stuck in NORMAL falls into _trap_guard (3
    presses, 2 "dismiss" rounds, press tries again) instead of
    scrolling. "dismiss" counts on the ladder but never accumulates enough
    to become scroll_bottom, because the press in the middle zeroes the count.
    """

    machine, actions, clock = build()

    for _ in range(DISMISS_ATTEMPTS_BEFORE_SCROLL + 3):

        machine.update([detection("up_food")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert "scroll_bottom" not in actions.actions, actions.actions

    assert set(actions.actions) == {"upgrade_food", "dismiss"}, (
        actions.actions
    )

    # The ladder never reaches the scroll threshold.
    assert machine._dismiss_attempts == 0


def test_painel_travado_suprime_up_food_e_libera_outros_upgrades():
    """
    The stuck-Wrap scenario: up_food visible for N straight updates
    without resolving (unreachable price) = stuck popup.
    Suppression skips up_food so lower-priority rules
    (other stations' upgrades) get a turn; when the window
    runs out, up_food gets another chance (alternates).
    """

    machine, actions, clock = build()

    tela = [detection("up_food"), detection("upgrade")]

    for _ in range(UPGRADE_FOOD_STUCK_UPDATES + 5):

        machine.update(tela)

        clock.advance(ACTION_COOLDOWN + 0.01)

    # The lower rule ("upgrade") got its turn mid-window.
    assert "upgrade" in actions.actions, actions.actions

    # Leaving NORMAL is progress: suppression zeroed on the switch.
    assert machine._up_food_suppressed == 0

    # Back to NORMAL, confirming the alternation: alone on
    # screen, up_food is paused by suppression then returns
    # (alternates instead of dying to one side).
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


def test_cinzento_travado_suprime_gray_e_libera_regras_de_baixo():
    """
    Mirror of the up_food test one level up: gray_coin (#4)
    outranks up_food (#5), so a grayed-out price
    (unreachable) starves the whole panel the same way. The
    suppression skips the gray pair so up_food and the upgrades
    below get a turn; then the pair gets another chance.
    """

    machine, actions, clock = build()

    # gray and up_food visible together stall both (double
    # suppression); the lower rule ("upgrade", e.g. stations with
    # reachable prices) is the one served in the window.
    tela = [
        detection("gray_coin"),
        detection("up_food"),
        detection("upgrade"),
    ]

    for _ in range(GRAY_STUCK_UPDATES + 5):

        machine.update(tela)

        clock.advance(ACTION_COOLDOWN + 0.01)

    # The lower rule ("upgrade") got its turn mid-window.
    assert "upgrade" in actions.actions, actions.actions

    # Alone on screen, gray is paused by suppression
    # then returns (alternates instead of dying to one side).
    machine2, actions2, clock2 = build()

    for _ in range(GRAY_STUCK_UPDATES):

        machine2.update([detection("gray_coin")])

        clock2.advance(ACTION_COOLDOWN + 0.01)

    assert machine2._gray_suppressed > 0

    antes = actions2.actions.count("gray_coin")

    for _ in range(GRAY_SUPPRESS_UPDATES + 1):

        machine2.update([detection("gray_coin")])

        clock2.advance(ACTION_COOLDOWN + 0.01)

    assert machine2._gray_suppressed == 0

    assert actions2.actions.count("gray_coin") > antes, (
        actions2.actions
    )

    # Swaps state zeroing the pair, as with up_food.
    machine2._gray_streak = 7
    machine2._gray_suppressed = 5
    machine2._enter(sm.UPGRADE)

    assert machine2._gray_streak == 0
    assert machine2._gray_suppressed == 0


def test_nao_age_duas_vezes_sobre_a_mesma_tela():
    """
    The reported bug: closed "MAX" then tapped AGAIN on the same spot,
    REOPENING the panel. Cause: the cooldown (0.5s) releases before any
    frame showing the effect exists, because the detector lags ~0.535s —
    the detection in hand comes from BEFORE the tap. Numbers measured on-device.
    """

    COOLDOWN = 0.5
    ATRASO = 0.535

    machine, actions, clock = build()

    machine.action_cooldown = COOLDOWN

    # First tap, on a freshly seen screen.
    machine.update([detection("gray_max")], 0.0)

    assert actions.actions == ["gray_max"], actions.actions

    # Past cooldown. The detection is still the SAME screen
    # from before the tap — that's what detector lag delivers.
    clock.advance(COOLDOWN + 0.01)

    machine.update([detection("gray_max")], ATRASO)

    assert actions.actions == ["gray_max"], (
        "tocou duas vezes sobre a mesma tela: "
        f"{actions.actions}"
    )


def test_age_quando_o_frame_e_posterior_a_acao():
    """The flip side: a post-action frame with the panel still there must act, or it'd become paralysis."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.5

    machine.update([detection("gray_max")], 0.0)

    clock.advance(0.6)

    # Small lag: the frame is from after the tap.
    machine.update([detection("gray_max")], 0.05)

    assert actions.actions == ["gray_max", "gray_max"], (
        actions.actions
    )


def test_frame_antigo_nao_bloqueia_para_sempre():
    """The risk of the fix would be stalling the bot when the detector is slow."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.5

    machine.update([detection("gray_max")], 0.0)

    # Several old frames: none acts.
    for _ in range(5):

        clock.advance(0.6)

        machine.update([detection("gray_max")], 10.0)

    assert actions.actions == ["gray_max"], actions.actions

    # Fresh frame: acts.
    machine.update([detection("gray_max")], 0.01)

    assert len(actions.actions) == 2, actions.actions


def test_primeira_acao_nao_precisa_esperar():
    """Without this guard the bot would do NOTHING at start, because last_action_time = 0 predates any frame."""

    machine, actions, _ = build()

    machine.last_action_time = 0.0

    # Below MAX_DETECTION_AGE: above it would hit the
    # stale-detection guard, which is another thing.
    machine.update([detection("gray_max")], 1.0)

    assert actions.actions == ["gray_max"], actions.actions


def test_a_guarda_vale_para_todo_caminho_de_acao():
    """
    The check lives in _can_act, the single point every path goes
    through (rules, FOOD/NEW_POINT, exploration swipe). Can't build
    the case via exploration: exploration_delay = 5s of empty screen
    would already have dropped the frame via MAX_DETECTION_AGE = 2s before here.
    """

    machine, actions, clock = build()

    machine.action_cooldown = 0.0
    machine.action_settle = 0.4

    machine.last_action_time = clock.now

    # Frame captured BEFORE the action: blocked.
    machine._frame_time = clock.now - 0.1

    assert not machine._can_act()

    # SAME-instant frame: blocked. Can't show the
    # effect of something that just happened.
    machine._frame_time = clock.now

    assert not machine._can_act()

    # Post-action frame BUT inside the animation: blocked. The case
    # the causal guard alone let through, causing the double tap.
    machine._frame_time = clock.now + 0.2

    assert not machine._can_act(), (
        "frame dentro da janela de animação não pode agir"
    )

    # Past settle: released.
    machine._frame_time = clock.now + 0.41

    assert machine._can_act()


def test_swipe_espera_a_vista_parar():
    """
    The reported bug: after scrolling, the bot spotted a target in a frame
    captured while the view was still sliding and tapped where the target
    WAS. A swipe moves the WHOLE VIEW and the game glides on inertia
    after the finger lifts — hence its own wait (SWIPE_WAITING_TIME),
    counted from the END of the gesture.
    """

    machine, actions, clock = build()

    # Empty screen right now: fires exploration.
    machine.exploration_delay = 0.0

    machine.update([], lag=0.0)

    assert ("swipe", "down") in actions.calls, actions.calls

    momento_do_swipe = clock.now

    # The swipe TAKES TIME to run (SWIPE_DURATION_MS) — the point of
    # the bug: measured from submission, the wait would be eaten by the
    # gesture itself with nothing left.
    clock.advance(0.5)

    actions.last_finished_at = clock.now

    fim_do_swipe = clock.now

    # Target visible, but the view still slides. PAST one tap's wait,
    # and still blocked: that's where the swipe wait differs from the
    # tap one — testing before ACTION_SETTLE would prove nothing.
    clock.advance(ACTION_SETTLE + 0.01)

    machine.update([detection("food"), detection("red_arrow")], lag=0.0)

    assert machine.action_counts["food"] == 0, (
        "tocou antes de a vista parar — é o clique que cai no "
        "lugar errado"
    )

    # Past SWIPE_WAITING_TIME from the swipe END: released.
    clock.advance(SWIPE_WAITING_TIME)

    assert clock.now > fim_do_swipe + SWIPE_WAITING_TIME

    machine.update([detection("food"), detection("red_arrow")], lag=0.0)

    assert machine.action_counts["food"] == 1, (
        machine.action_counts
    )

    # And the wait was counted from the gesture end, not submission:
    # otherwise this test would pass with the bug inside.
    assert (
        clock.now - momento_do_swipe
        > SWIPE_WAITING_TIME + 0.5
    )


def test_piso_do_frame_concorda_com_o_can_act():
    """
    `frame_floor()` is the SAME deadline `_can_act` enforces, so the
    VisionWorker can skip frames the machine would drop anyway.
    If they diverge, the worker throws away frames that would serve (blind
    bot) or analyzes frames that don't serve (wasted pass).
    """

    # Freshly born machine (build() fast-forwards the last-action clock
    # to beat cooldown, no good here): with no action, any floor
    # would blind the bot at startup.
    assert sm.StateMachine(FakeActions()).frame_floor() == 0.0

    machine, actions, clock = build()

    machine.action_cooldown = 0.0

    assert machine._act("food", detection("food")) is True

    actions.last_finished_at = clock.now

    piso = machine.frame_floor()

    assert piso == clock.now + ACTION_SETTLE, piso

    # Exactly at the floor: blocked by both.
    machine._frame_time = piso

    assert not machine._can_act()

    # A hair above: released by both.
    machine._frame_time = piso + 0.001

    assert machine._can_act()

    # And the swipe floor is the larger of the two.
    machine._last_was_swipe = True

    assert machine.frame_floor() == clock.now + SWIPE_WAITING_TIME


def test_swipe_espera_mais_que_um_toque():
    """SWIPE_WAITING_TIME below ACTION_SETTLE would make no sense: the swipe moves MORE of the screen."""

    assert SWIPE_WAITING_TIME >= ACTION_SETTLE, (
        SWIPE_WAITING_TIME,
        ACTION_SETTLE,
    )

    # And not so long that exploration slows: the bot does
    # MAX_SWIPES in a row looking for content.
    assert SWIPE_WAITING_TIME <= 2.0, SWIPE_WAITING_TIME


def test_scroll_bottom_usa_a_espera_do_swipe():
    """`scroll_bottom` is SIX swipes in a row; with a short wait, the ladder would tap on a moving screen again."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.0

    assert machine._act("scroll_bottom", None) is True

    assert machine._last_was_swipe is True, (
        "scroll_bottom tem de contar como movimento de vista"
    )

    actions.last_finished_at = clock.now

    # Inside the swipe wait (but outside the tap one): blocked.
    machine._frame_time = clock.now + ACTION_SETTLE + 0.01

    assert not machine._can_act(), (
        "a espera curta de toque não serve para seis swipes"
    )

    machine._frame_time = clock.now + SWIPE_WAITING_TIME + 0.01

    assert machine._can_act()


def test_toque_comum_nao_herda_a_espera_do_swipe():
    """The flag must FLIP BACK: one swipe can't leave later taps paying the long wait."""

    machine, actions, clock = build()

    machine.action_cooldown = 0.0

    machine._last_was_swipe = True

    assert machine._act("food", detection("food")) is True

    assert machine._last_was_swipe is False

    actions.last_finished_at = clock.now

    # Just the tap wait.
    machine._frame_time = clock.now + ACTION_SETTLE + 0.01

    assert machine._can_act()


def test_nao_toca_com_o_painel_ja_fechado():
    """
    The reported bug, replayed in the time domain. Simulates the game TRUTH
    vs. the machine's LAGGED view: tap with panel open closes
    (after animation); tap with panel closed OPENS (DISMISS_POINT opens
    something — that's why the double tap hurts); the machine only sees the
    screen from `lag` seconds ago. The spurious tap is the one made with the panel
    ALREADY closed — without the guard several fire, with it, none.
    """

    ATRASO = 0.535     # measured on-device, 165 templates
    COOLDOWN = 0.5
    ANIMACAO = 0.3     # game time to close

    def roda(settle):

        clock = Clock()

        sm.time.monotonic = clock.monotonic

        actions = FakeActions()

        machine = sm.StateMachine(actions)

        machine.action_cooldown = COOLDOWN
        machine.action_settle = settle
        machine.last_action_time = clock.now - 100

        # (instant it takes effect, open?)
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

    # Isolates the causal/settle guard under test: gray suppression
    # (25 updates) would also mute the taps and mask the bug
    # this test must reproduce. Patched on core.failures, where the
    # suppression now lives (split from state_machine).
    from core import failures as _failures

    limiar = _failures.GRAY_STUCK_UPDATES
    _failures.GRAY_STUCK_UPDATES = 10 ** 9

    try:
        # No settle (cooldown + causal guard only): taps into the void.
        _, espurios_sem = roda(0.0)

        assert espurios_sem > 0, (
            "a simulação deveria reproduzir o bug com settle=0"
        )

        # With settle covering the animation: one tap, no spurious.
        toques, espurios = roda(ACTION_SETTLE)

        assert espurios == 0, f"{espurios} toque(s) no painel fechado"

        assert toques == 1, f"{toques} toques onde 1 bastava"
    finally:
        _failures.GRAY_STUCK_UPDATES = limiar


def test_settle_cobre_a_animacao_configurada():
    """Settle must exceed the game animation. Lowering ACTION_SETTLE to 0.1 would bring the double tap back."""

    # Typical game-panel animation.
    assert ACTION_SETTLE >= 0.3, ACTION_SETTLE

    # And not so high the bot slows: above ~1s every
    # decision would cost settle + detector lag.
    assert ACTION_SETTLE <= 1.0, ACTION_SETTLE


def test_escada_zera_quando_algo_e_resolvido():
    """If a normal action happened, the count restarts: no screen scroll over old attempts."""

    machine, actions, clock = build()

    for _ in range(DISMISS_ATTEMPTS_BEFORE_SCROLL):

        machine.update([detection("gray_max")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert machine._dismiss_attempts == (
        DISMISS_ATTEMPTS_BEFORE_SCROLL
    )

    # Progress: a game action.
    machine.update([detection("box")])

    clock.advance(ACTION_COOLDOWN + 0.01)

    assert machine._dismiss_attempts == 0

    # Next gray_max starts from scratch, not from scroll.
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
    """A game action can't become a screen scroll by repeating."""

    machine, actions, clock = build()

    for _ in range(DISMISS_ATTEMPTS_BEFORE_SCROLL + 3):

        machine.update([detection("box")])

        clock.advance(ACTION_COOLDOWN + 0.01)

    assert set(actions.actions) == {"open_box"}, actions.actions

    assert "gray_max" in DISMISS_ACTIONS
    assert "dismiss" in DISMISS_ACTIONS
    assert "open_box" not in DISMISS_ACTIONS


def test_city_stuck_sweep_clears_give_up_and_swipes_soon():
    """10 min without progress: forced sweep clears the give-up
    lists and swipes immediately instead of waiting out the
    exploration interval."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    machine._give_up["food"]["ignored"].append(
        (detection("food"), clock.now)
    )

    machine._last_progress = (
        clock.now - CITY_STUCK_SWEEP_AFTER - 1.0
    )

    machine.update([])

    assert machine._recovery_stage == 1, machine._recovery_stage
    assert machine._give_up["food"]["ignored"] == []
    assert ("swipe", machine.swipe_direction) in actions.calls


def test_reset_clears_wedged_state():
    """A city restart must not inherit wedged bookkeeping:
    suppression, dismiss ladder, give-up lists, and the
    time-boxed upgrade ignore are all zeroed."""

    machine, actions, clock = build()

    machine._up_food_suppressed = 5
    machine._dismiss_attempts = 3
    machine._upgrade_ignored_until = clock.now + 50.0
    machine._give_up["food"]["ignored"].append(
        (detection("food"), clock.now)
    )

    machine.reset()

    assert machine._up_food_suppressed == 0
    assert machine._dismiss_attempts == 0
    assert machine._upgrade_ignored_until == 0.0
    assert machine._give_up["food"]["ignored"] == []


def test_city_stuck_restart_relaunches_and_resets():
    """15 min without progress: the game package is restarted and
    the wedged counters are cleared with it."""

    machine, actions, clock = build()

    machine.action_settle = 0.0
    machine.action_cooldown = 0.0

    class FakeAndroid:

        def __init__(self):
            self.restarts = []

        def restart_package(self, package):
            self.restarts.append(package)
            return True

    actions.android = FakeAndroid()

    machine._up_food_suppressed = 5
    machine._last_progress = (
        clock.now - CITY_STUCK_RESTART_AFTER - 1.0
    )

    machine.update([])

    assert actions.android.restarts == [GAME_PACKAGE]
    assert machine._up_food_suppressed == 0
    assert machine._recovery_stage == 0, machine._recovery_stage


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
