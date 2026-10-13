"""State machine: rule table per state, single _act function, per-state timeout."""

import logging
import time
from collections import Counter

from core import log
from core.metrics import formata_duracao
from core.config import (
    ACTION_COOLDOWN,
    ACTION_SETTLE,
    BOX_STUCK_UPDATES,
    BOX_SUPPRESS_UPDATES,
    CITY_STUCK_RESTART_AFTER,
    CITY_STUCK_SWEEP_AFTER,
    CYCLE_CATEGORIES,
    DISMISS_ACTIONS,
    DISMISS_ATTEMPTS_BEFORE_SCROLL,
    EXPLORATION_DELAY,
    EXPLORATION_DELAY_AFTER_ACTION,
    FOOD_IGNORE_IOU,
    FOOD_IGNORE_TTL,
    FOOD_MAX_ATTEMPTS,
    GAME_PACKAGE,
    GRAY_STUCK_UPDATES,
    GRAY_SUPPRESS_UPDATES,
    MAX_DETECTION_AGE,
    MAX_SWIPES,
    PROGRESS_ACTIONS,
    RED_ARROW_GATE_MARGIN,
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
    UPGRADE_EMPTY_VISITS,
    UPGRADE_IGNORE_SECONDS,
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


# Gray dismiss pair: an unreachable price shows dimmed and
# never resolves on tap — same suppression treatment as up_food.
GRAY_DISMISS_CATEGORIES = ("gray_coin", "gray_max")


# Rules: (category, action, next state). Order = priority
# (the first category found is the only one executed in the
# frame); next_state None keeps the current state.

# The first four close whatever should not be open,
# before any game action.
#
# up_food sits LAST in NORMAL on purpose: the up_food template (blue
# button w/ coin) also matches the main-view cost buttons at ~0.97
# confidence. At high priority it hijacks NORMAL and fires 4s
# upgrade_food presses on the wrong screen, overlapping the FOOD
# panel cycle. Demoted below food/box so real station taps win; the
# FOOD state still handles the panel.
#
# food itself is gated by the red upgrade badge (see _handle_food):
# dishes WITHOUT a badge are not upgradeable, and tapping them just
# opens a station view the bot then dismisses (open-then-dismiss
# loop). Strict gate = badge or nothing.
# The gray pair (dimmed price chips) usually sits on the MAIN view as
# permanent chrome for unaffordable/maxed stations — a tap never
# resolves them. So they sit BELOW the real game actions: if they
# outranked plane/build/delivery/new_point/food, an unaffordable chip
# would starve the whole city (315 gray_max taps in one 5.5h run).
# The GRAY_STUCK suppression still protects panels where they truly
# block.
NORMAL_RULES = [
    ("open_store", "open_store_click", None),
    ("close", "close", None),
    ("box", "open_box", None),

    ("plane", "plane", RENOVATE),
    ("build", "click", RENOVATE),
    ("upgrade", "upgrade", UPGRADE),
    ("delivery", "delivery", DELIVERY),
    ("new_point", "new_point", NEW_POINT),
    ("food", "food", FOOD),
    ("up_food", "upgrade_food", None),

    ("gray_max", "gray_max", None),
    ("gray_coin", "gray_coin", None),
]

RENOVATE_RULES = [
    ("renovate", "renovate_click", NORMAL),
    ("fly", "renovate_click", NORMAL),
]

DELIVERY_RULES = [
    ("accept_delivery", "accept_delivery", NORMAL),
    ("helper", "helper", HELPER),
    ("close", "close", NORMAL),
]

HELPER_RULES = [
    ("invite_helper", "invite_helper", NORMAL),
    ("delivery", "delivery", DELIVERY),
    ("close", "close", NORMAL),
]

# up_upgrade stays in UPGRADE on purpose: on the next
# frame we look for another item to upgrade. We only leave the
# screen when there is nothing left.
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

    # FOOD and NEW_POINT have their own wait after the press,
    # so they are handled in a dedicated handler. The categories
    # stay here only for the detector filter.
    FOOD: [
        ("up_food", "upgrade_food", None),
        ("gray_max", "gray_max", None),
        ("gray_coin", "gray_coin", None)
    ],
    NEW_POINT: [
        ("up_food", "new_point_click", None),
        ("gray_max", "gray_max", None),
        ("gray_coin", "gray_coin", None),
    ],
}


class StateMachine:

    # Kept as class attributes for compatibility
    # with code referencing StateMachine.NORMAL.
    NORMAL = NORMAL
    RENOVATE = RENOVATE
    FOOD = FOOD
    NEW_POINT = NEW_POINT
    UPGRADE = UPGRADE
    DELIVERY = DELIVERY
    HELPER = HELPER

    def __init__(self, action_manager, recorder=None, detector=None):

        self.action_manager = action_manager

        # Training dataset recorder. None = do not record.
        self.recorder = recorder
        self.detector = detector

        self.state = NORMAL
        self.state_entered = time.monotonic()

        self.last_action_time = 0.0
        self.action_cooldown = ACTION_COOLDOWN
        self.action_settle = ACTION_SETTLE

        # Swipe's own wait, longer than a tap's: a
        # tap moves a panel, a swipe moves the WHOLE VIEW and
        # the game keeps sliding by inertia.
        self.swipe_settle = SWIPE_WAITING_TIME

        # Was the last action a swipe?
        #
        # Stored as a FLAG, not as a copy of the
        # settle value, on purpose: so `machine.action_settle
        # = 0` keeps disabling the tap wait (which is what the
        # tests do) without a second stale
        # number living somewhere else.
        self._last_was_swipe = False

        # Current frame, stored for the dataset recorder.
        self._frame = None
        self._detections = []
        self._lag = 0.0

        # When the frame that produced the detections in hand was
        # CAPTURED (not when it arrived here).
        #
        # Without this the bot acts twice on the same screen: the
        # cooldown releases before any frame shows
        # the first action's effect. See _can_act.
        self._frame_time = None

        # How many times it waited for a new frame, to avoid filling the
        # log with one line per frame.
        self._waited_warned = 0.0

        # Wait after the food long press.
        self.up_food_wait_start = None

        # Current "food" item, waiting for up_food. Stored here (and
        # not only in the local detection) because _wait_or_give_up needs
        # it to know WHICH item to ignore when giving up.
        self._food_target = None

        # "food" items that exhausted UP_FOOD_WAIT in
        # FOOD_MAX_ATTEMPTS consecutive tries — ignored until the
        # next real swipe, otherwise the bot tries the same item
        # forever (finding it keeps postponing the swipe).
        self._ignored_food = []

        # Last give-up (item + how many times in a row), to know
        # whether the NEXT one is on the same item or a restart.
        self._food_fail_target = None
        self._food_fail_count = 0

        # Counted from start, not from the first build: the first
        # stretch is information too.
        self.cycle_at = time.monotonic()
        self.cycle_last = None
        self.cycle_count = 0

        # Counted in _act (the single point actions leave through);
        # counting in _apply_rules would include attempts blocked by
        # cooldown/busy worker. Per category and per action
        # because they are different questions: `fly` is a category
        # (flights), `upgrade_item` is an action (upgrade clicks).
        self.category_counts = Counter()
        self.action_counts = Counter()

        self.last_action = None
        self.last_action_at = None

        # Normal pace: a swipe every so often, while
        # nothing shows up.
        self.exploration_delay = EXPLORATION_DELAY

        # After FINDING something, exploration is POSTPONED by this
        # much — not cancelled.
        self.exploration_action_delay = (
            EXPLORATION_DELAY_AFTER_ACTION
        )

        # What the next-swipe wait counts from: the
        # last swipe, or the last time something was found.
        self.explore_anchor = time.monotonic()

        # Did the anchor come from a "find"?
        #
        # Stored as a FLAG, with the interval derived from it at
        # call time — not a copy of the number. That is what keeps
        # `machine.exploration_delay = 0` working (the tests
        # use that) with no second stored value that
        # could go stale. Same pattern as
        # _last_was_swipe.
        self.explore_found = False

        # Sweep cycle: 5 swipes one way, 5 the other,
        # indefinitely. The count does NOT reset when the
        # bot finds something — it used to reset, and the effect was always sweeping
        # the same slice of the screen (almost every swipe reveals
        # some target, so the return trip never closed and the far
        # half of the restaurant was never visited).
        self.swipe_direction = SWIPE_START_DIRECTION
        self.swipe_count = 0
        self.max_swipes = MAX_SWIPES

        # Avoid repeating the stale-detection warning on every frame.
        self._stale_warned = 0.0

        # Detects an action repeating without resolving anything.
        self._last_action = None
        self._repeat_count = 0

        # Consecutive upgrade_food presses in NORMAL (see
        # _trap_guard: a panel that never resolves falls back to dismiss).
        self._up_food_repeats = 0

        # Consecutive updates with up_food visible without the panel
        # resolving (see update: stuck popup, e.g. unreachable
        # price) + active up_food suppression.
        self._up_food_streak = 0
        self._up_food_suppressed = 0

        # Same one level up: dimmed gray_coin/gray_max
        # (unreachable price) outrank up_food and
        # lock the panel the same way — suppression of the pair.
        self._gray_streak = 0
        self._gray_suppressed = 0

        # Same one step down: a chest that never resolves is a
        # paid offer, not loot — suppression of open_box.
        self._box_streak = 0
        self._box_suppressed = 0

        # Foreground gate: while the game is not focused (settings,
        # Gmail, Play sheet) no action may fire. Set from main.py's
        # game-ensure check; defaults to True so tests and the first
        # frames behave as before.
        self._game_in_foreground = True
        self._foreground_warned = 0.0

        # Consecutive attempts at closing something. Climbs the
        # escalation ladder instead of repeating what did not work.
        self._dismiss_attempts = 0

        # City-stuck recovery: last time an action proved the city
        # is moving forward (PROGRESS_ACTIONS or a renovation
        # door). Recovery stage 0 = none, 1 = forced sweep done,
        # 2 = app restarted. Reset by real progress.
        self._last_progress = time.monotonic()
        self._recovery_stage = 0

        # How many times it already had to scroll to escape.
        self._dismiss_rounds = 0

        # Empty-Upgrades-panel guard: consecutive UPGRADE visits with
        # no buy, and until when the `upgrade` button stays ignored.
        # Same idea as _ignored_food, but time-boxed instead of
        # swipe-reset (the button is fixed nav chrome — swipes never
        # move it, so a swipe reset would never stick).
        self._upgrade_empty_visits = 0
        self._upgrade_ignored_until = 0.0

        # Whether the current UPGRADE visit bought anything yet.
        self._upgrade_did_item = False

    def set_game_foreground(self, in_foreground):
        """Called from main.py's game-ensure check. None = unknown, keep previous."""
        if in_foreground is None:
            return
        self._game_in_foreground = bool(in_foreground)

    def wanted_categories(self):
        """Current-state categories, in priority order — detector filter (~264ms → ~16ms per pass in UPGRADE)."""

        rules = STATE_RULES.get(self.state, NORMAL_RULES)

        categories = []

        for category, _, _ in rules:

            if category not in categories:

                categories.append(category)

        # Outside NORMAL, we still want to see the "X" (emergency
        # exit from any modal). At the END of the list: it is a
        # safety net, not a priority — where it truly matters
        # it is already in the rules, in the right position.
        if "close" not in categories:

            categories.append("close")

        # In NORMAL the red-arrow badge must be visible so the
        # food rule can gate on it. FIRST, not last: the detector
        # stops at the first matching category, so anything after
        # `food` would never be searched on a food screen (and the
        # gate would block every food tap). One template, searched
        # every NORMAL pass; never acted on (no rule row fires it).
        # Same pattern as "close" below, opposite end.
        if self.state == NORMAL and "red_arrow" not in categories:

            categories.insert(0, "red_arrow")

        # While `upgrade` is on cooldown the rule would be skipped
        # anyway, but asking for it still costs a match that stops the
        # pass before delivery/new_point/food — so the bot would keep
        # swiping while "ignoring upgrades". Better to not ask.
        if self.state == NORMAL and self._upgrade_ignored():

            if "upgrade" in categories:

                categories.remove("upgrade")

        return categories

    def update(
        self,
        detections,
        lag=0.0,
        frame=None,
        frame_time=None,
    ):

        # Before any return: this is where the previous action
        # gets its verdict, even on frames where the bot does not act.
        if self.recorder is not None:

            self._frame = frame

            self._detections = detections

            self.recorder.observe(
                frame,
                detections,
                frame_time,
                self.state,
            )

        # lag is the frame age; stored before anything
        # else because every action path goes through _can_act.
        self._frame_time = time.monotonic() - lag

        self._lag = lag

        if self._timed_out():
            return

        # Stale detection: clicking from an old frame
        # hits where the object WAS.
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

            # Panel went away on its own: the count restarts from zero
            # the next time up_food shows up.
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
                        # Suppression exhausted: up_food gets a
                        # fresh window (zeroed streak) before it
                        # can be suppressed again — alternates
                        # between the panel and the other upgrades.
                        self._up_food_streak = 0
                # Same cycle for the gray pair: visible for N
                # updates without resolving = stuck popup; suppress
                # so up_food and the upgrades below get
                # their turn, then the pair gets another chance.
                if any(
                    d.get("category") in GRAY_DISMISS_CATEGORIES
                    for d in detections or []
                ):
                    self._gray_streak += 1
                    if (
                        self._gray_streak >= GRAY_STUCK_UPDATES
                        and self._gray_suppressed <= 0
                    ):
                        self._gray_suppressed = (
                            GRAY_SUPPRESS_UPDATES
                        )
                        self._gray_streak = 0
                        logger.warning(
                            "gray dismiss (gray_coin/gray_max) "
                            "stuck open for %d updates in "
                            "NORMAL (unresolvable panel?) — "
                            "suppressing gray dismiss for %d "
                            "updates so lower rules get a turn",
                            GRAY_STUCK_UPDATES,
                            GRAY_SUPPRESS_UPDATES,
                        )
                else:
                    self._gray_streak = 0
                    self._gray_suppressed = 0
                if self._gray_suppressed > 0:
                    self._gray_suppressed -= 1
                    if self._gray_suppressed == 0:
                        self._gray_streak = 0
                # Same cycle for the reward chest: visible for N
                # updates without resolving = paid offer (e.g.
                # starter pack), not free loot; suppress open_box
                # so the bot backs off instead of tapping the paywall.
                if any(
                    d.get("category") == "box"
                    for d in detections or []
                ):
                    self._box_streak += 1
                    if (
                        self._box_streak >= BOX_STUCK_UPDATES
                        and self._box_suppressed <= 0
                    ):
                        self._box_suppressed = (
                            BOX_SUPPRESS_UPDATES
                        )
                        self._box_streak = 0
                        logger.warning(
                            "box stuck open for %d updates in "
                            "NORMAL (paid offer?) — "
                            "suppressing open_box for %d "
                            "updates",
                            BOX_STUCK_UPDATES,
                            BOX_SUPPRESS_UPDATES,
                        )
                else:
                    self._box_streak = 0
                    self._box_suppressed = 0
                if self._box_suppressed > 0:
                    self._box_suppressed -= 1
                    if self._box_suppressed == 0:
                        self._box_streak = 0

            self._city_stuck_check()

            acted = self._apply_rules(
                detections,
                STATE_RULES.get(self.state, NORMAL_RULES),
            )

            # Only NORMAL explores: in the other states we are
            # inside a panel, and swiping would get in the way.
            if self.state == NORMAL and (
                not SWIPE_WAIT_FOR_NO_ACTION or not acted
            ):

                self._explore_screen()

    def _apply_rules(self, detections, rules):
        """Runs the 1st rule whose category shows up; returns True if it found something (even when the action is blocked by cooldown), because finding already means we are not lost."""

        for prioridade, (category, action, next_state) in enumerate(
            rules,
            start=1,
        ):

            # Active suppression: skips up_food so lower-
            # priority rules (other stations' upgrades)
            # get their turn. Detection keeps running — that is
            # what counts the streak and notices when the panel is gone.
            # Same for the gray pair (even higher priority):
            # without this it would starve up_food forever.
            if self.state == NORMAL and (
                (
                    category == "up_food"
                    and self._up_food_suppressed > 0
                )
                or (
                    category in GRAY_DISMISS_CATEGORIES
                    and self._gray_suppressed > 0
                )
                or (
                    category == "box"
                    and self._box_suppressed > 0
                )
                or (
                    category == "upgrade"
                    and self._upgrade_ignored()
                )
            ):
                continue

            detection = self._find(detections, category)

            if detection is None:
                continue

            # Red-badge preference: dishes WITH a red upgrade badge are
            # upgradeable — prefer them, but fall back to any visible
            # dish when no badge is detected (badge templates miss on
            # some views; a strict gate blinded the bot entirely).
            if category == "food" and self.state == NORMAL:

                badged = self._find_badged_food(detections)

                if badged is not None:

                    detection = badged

            # In DEBUG, shows which priority won and what
            # was on screen with it. That is what answers "why
            # did it click this and not that".
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
        """Single place in the project that fires an action and switches state."""

        # Foreground gate: never tap when another app (settings,
        # Gmail, Play sheet) is in front — a paper-plane match on
        # Gmail's send arrow must not become a sent email.
        if not self._game_in_foreground:
            now = time.monotonic()
            if now - self._foreground_warned > 5.0:
                logger.warning(
                    "Game not in foreground — holding all actions."
                )
                self._foreground_warned = now
            return False

        if not self._can_act():
            return False

        action, detection = self._trap_guard(action, detection)

        action = self._escalate(action)

        if not self.action_manager.execute(action, detection):

            # Worker busy: try on the next frame.
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

        # `scroll_bottom` is six swipes in a row: it moves the view
        # more than any single swipe, so it deserves the
        # same long wait. A plain tap keeps the short one.
        self._last_was_swipe = action in VIEW_MOVING_ACTIONS

        # Real swipe: ignored items may now be in a different
        # position, or off screen. Worth trying again.
        if self._last_was_swipe:
            self._reset_food_tracking()

        # Same reasoning as the cycle below: only counts what
        # actually went out.
        self.action_counts[action] += 1

        if detection:
            self.category_counts[detection["category"]] += 1

        self.last_action = action
        self.last_action_at = self.last_action_time

        # Here, not in _apply_rules: this is the point where the
        # action REALLY went out. Marking earlier would count cycles on
        # attempts blocked by cooldown or a busy worker.
        if detection and detection["category"] in CYCLE_CATEGORIES:

            self._mark_cycle(detection["category"])

        # Real forward motion resets the city-stuck clock (and the
        # recovery ladder with it).
        if action in PROGRESS_ACTIONS or (
            detection and detection["category"] in CYCLE_CATEGORIES
        ):

            self._last_progress = time.monotonic()
            self._recovery_stage = 0

        # A buy inside UPGRADE: this visit was not empty.
        if action == "upgrade_item":

            self._upgrade_did_item = True

        self._count_repeat(action)

        if self.detector is not None and detection is not None:
            self.detector.learn_action(self._frame, detection)

        self._record(action, detection)

        if next_state is not None and next_state != self.state:

            self._enter(next_state)

        # After _enter (which resets up_food_wait_start): arms the
        # FOOD give-up timer AND stores WHICH item entered FOOD, so
        # _wait_or_give_up fires UP_FOOD_WAIT after the tap even when
        # up_food never shows up — without this the timer stays None,
        # _wait_or_give_up no-ops, and the only exit is the 5s state
        # timeout (which registers no failure, so the dish is tapped
        # forever).
        if action == "food":
            self._food_target = detection
            self.up_food_wait_start = time.monotonic()

        return True

    def _record(self, action, detection):
        """Hands the recorder the action that JUST went out; called from inside _act because that is the only point where we know it was not blocked by cooldown or a busy worker."""

        if self.recorder is None:
            return

        if self._frame is None:
            return

        self.recorder.on_action(
            action=action,

            # swipe_* is not in ACTION_TABLE: it is exploration,
            # which calls the ActionManager through another path.
            action_kind=(
                "swipe"
                if action.startswith("swipe_")
                else self.action_manager.kind_of(action)
            ),
            detection=detection,

            # Tap point in the RECORDED image space.
            # None for swipes: not a point tap.
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

        # Renovation door = new city coming = new general upgrades:
        # the empty-panel ignore must not survive into it.
        self._upgrade_empty_visits = 0
        self._upgrade_ignored_until = 0.0

    def cycle_stats(self):
        """(elapsed time, last cycle, count) for the HUD; last cycle is None until the first build/plane."""

        return (
            time.monotonic() - self.cycle_at,
            self.cycle_last,
            self.cycle_count,
        )

    def summary(self):
        """What matters to whoever watches the screen, as a dict — business names (flights, renovations), not code names."""

        return {
            "estado": self.state,
            "acao": self.last_action,
            "acao_ha": (
                None
                if self.last_action_at is None
                else time.monotonic() - self.last_action_at
            ),

            # `fly` is the plane button on the renovation screen; it is
            # what effectively changes restaurants.
            "voos": self.category_counts.get("fly", 0),

            # `build` and `plane` are the two doors into RENOVATE
            # (CYCLE_CATEGORIES). Summed, it is how many times the bot
            # entered a renovation; `cycle_count` counts the same
            # event, and serves as a cross-check.
            "reformas": sum(
                self.category_counts.get(categoria, 0)
                for categoria in CYCLE_CATEGORIES
            ),

            "acoes": sum(self.action_counts.values()),
        }

    def _trap_guard(self, action, detection):
        """NORMAL only: up_food surviving UPGRADE_FOOD_MAX_REPEATS presses falls back to "dismiss" (HOLD on a fixed point, no detection) for UPGRADE_FOOD_DISMISS_ROUNDS rounds; then the cycle restarts and the press gets another chance. Any other action resets the count."""

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
        """Closing ladder: N attempts on the fixed point, then scroll to the end and try again — the point may be what OPENS the panel, so tapping it sometimes reopens instead of closing; there is no BACK in this game."""

        if action not in DISMISS_ACTIONS:

            # Normal action: progress, reset the ladder.
            self._dismiss_attempts = 0
            self._dismiss_rounds = 0

            return action

        self._dismiss_attempts += 1

        if (
            self._dismiss_attempts
            <= DISMISS_ATTEMPTS_BEFORE_SCROLL
        ):
            return action

        # Out of point attempts: scroll and restart the
        # count, to try the point again later.
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
        """Warns when the same action repeats without leading anywhere (typical symptom: a dismiss point that does not close the panel)."""

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
        """Minimum CAPTURE instant for a frame to authorize an action. Single definition, shared with the VisionWorker (avoids diverging and acting on a stale screen); counted from the END of the action, not submission, since async actions vary in duration (500ms swipe, 4s food long press); the final floor is the largest of the state-entry wait and the post-action settle."""

        # Entry wait does not depend on the last action: a screen
        # that opened with an animation is not ready just because the
        # tap settled (without this, upgrade was closed mid-
        # animation, when only the "X" matched yet). The `if` keeps a
        # state WITHOUT entry wait from blocking on a tie the
        # first zero-age frame (floor == state_entered).
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

        # An in-flight action (e.g. 4 s long press) must
        # finish before the next one.
        if self.action_manager.is_busy():
            return False

        agora = time.monotonic()

        if agora - self.last_action_time < self.action_cooldown:
            return False

        # Cooldown is not enough: with a lagging detector, the
        # detection in hand may come from a frame BEFORE the last
        # action (symptom: double taps reopening panels — the
        # dismiss point is often also what opens them). The right
        # condition is causal (frame after the END of the action, see
        # frame_floor), not temporal — a longer cooldown would only
        # delay the same problem under a slower detector.
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

        # UPGRADE visit bookkeeping: entering arms the per-visit buy
        # flag; leaving for NORMAL accounts the visit (empty ones
        # pile up toward ignoring the button — see
        # _account_upgrade_visit).
        if state == UPGRADE:

            self._upgrade_did_item = False

        elif previous_state == UPGRADE and state == NORMAL:

            self._account_upgrade_visit()

        # Previous-state locals do not leak
        # into the next one.
        self.up_food_wait_start = None

        # Switching state is progress: counts reset.
        self._last_action = None
        self._repeat_count = 0
        self._up_food_repeats = 0
        self._up_food_streak = 0
        self._up_food_suppressed = 0
        self._gray_streak = 0
        self._gray_suppressed = 0
        self._box_streak = 0
        self._box_suppressed = 0
        self._dismiss_attempts = 0
        self._dismiss_rounds = 0

        if state != FOOD and previous_state != FOOD:

            self._delay_exploration()

    def _timed_out(self):

        timeout = STATE_TIMEOUTS.get(self.state)

        if timeout is None:
            return False

        # Same async overlap as _wait_or_give_up: the worker holds the
        # press on its own thread, so a state timeout must not fire
        # mid-press either — freeze the clock while busy.
        if self.action_manager.is_busy():
            return False

        elapsed = time.monotonic() - self.state_entered

        if elapsed < timeout:
            return False

        logger.warning(
            "%s stuck for %.1fs — returning to NORMAL",
            self.state,
            elapsed,
        )

        # Timeout exit for FOOD: without this the give-up path in
        # _wait_or_give_up is skipped entirely (e.g. panel opened with
        # no up_food AND no gray, or tap landed on a dead frame), so
        # the dish would be tapped again on the next NORMAL pass —
        # the "FOOD stuck" infinite loop. Registering here feeds the
        # FOOD_MAX_ATTEMPTS ignore instead.
        if self.state == FOOD and self._food_target is not None:

            self._register_food_failure(self._food_target)
            self._food_target = None

        self._enter(NORMAL)

        return True

    def _food(self, detections):
        """Holds the food upgrade button and waits for the next one to appear; with no next one in UP_FOOD_WAIT, returns to NORMAL."""

        up_food = self._find(detections, "up_food")

        if up_food is not None:

            if self._act("upgrade_food", up_food):

                self._delay_exploration()

                self.up_food_wait_start = time.monotonic()

            return

        # Exhausted panel: dismiss (DISMISS_POINT or
        # GRAY_COIN_POINT). BEFORE _wait_or_give_up, not
        # after: it may call _enter(NORMAL), and an _act
        # after that would act outside the FOOD state.
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
        """Same as FOOD, but a single click releases it — not a long press."""

        unlock_food = self._find(detections, "up_food")

        if unlock_food is not None:

            self._delay_exploration()

            if self._act("new_point_click", unlock_food):

                self.up_food_wait_start = time.monotonic()

            return

        # Unaffordable new point: dismiss immediately
        for categoria, acao in (
            ("gray_max", "gray_max"),
            ("gray_coin", "gray_coin"),
        ):

            deteccao = self._find(detections, categoria)

            if deteccao is not None:

                self._act(acao, deteccao, NORMAL)

                return

        self._wait_or_give_up()

    def _wait_or_give_up(self):
        """The screen may take a while to update after acting; waits UP_FOOD_WAIT before giving up."""

        if self.up_food_wait_start is None:
            return

        # Async overlap guard: the action worker runs the press on its
        # own thread (UPGRADE_FOOD_PRESS=4s > UP_FOOD_WAIT=2s). While the
        # worker is busy the screen still shows the pre-press state, so
        # don't let the timer expire mid-press — freeze the clock.
        if self.action_manager.is_busy():
            return

        # Timer runs from the last FINISHED action, not from submit:
        # last_finished_at is only set when the worker thread completes
        # the press, so a slow/queued press can't be cut off early.
        # getattr: the test fake has no worker thread.
        last_finished = getattr(self.action_manager, "last_finished_at", 0.0)
        elapsed = time.monotonic() - max(
            self.up_food_wait_start,
            last_finished,
        )

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
        """Counts CONSECUTIVE give-ups on the same item (by IoU); on reaching FOOD_MAX_ATTEMPTS, ignores it until the next swipe or FOOD_IGNORE_TTL passing, whichever comes first — before that, returns to NORMAL and lets it try again (it may just be up_food slow to appear)."""

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

        self._ignored_food.append((target, time.monotonic()))

        self._food_fail_target = None
        self._food_fail_count = 0

    def _reset_food_tracking(self):
        """Forgets ignored items and the attempt count — called on every real swipe, which changes what is on screen."""

        self._ignored_food = []

        self._food_fail_target = None
        self._food_fail_count = 0

    def _find(self, detections, category):
        """First detection of the category. No threshold refiltering here on purpose: a second threshold once lived in this method, diverging from the detector's, and the winner was the first in filename alphabetical order, not the best."""

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
        """Same item as a 'food' that already exhausted UP_FOOD_WAIT without up_food (same on-screen position, compared by IoU since the screen does not move outside a swipe). Entries expire after FOOD_IGNORE_TTL — a bot kept busy by other chrome never swipes, and swipe-only expiry would ignore the dish forever."""

        now = time.monotonic()

        alive = [
            (ignored, ignored_at)
            for ignored, ignored_at in self._ignored_food
            if now - ignored_at < FOOD_IGNORE_TTL
        ]

        self._ignored_food = alive

        return any(
            self._iou(detection, ignored) >= FOOD_IGNORE_IOU
            for ignored, _ in alive
        )

    def _upgrade_ignored(self):
        """`upgrade` button on cooldown after UPGRADE_EMPTY_VISITS
        consecutive empty visits ("You have all upgrades!")."""

        return time.monotonic() < self._upgrade_ignored_until

    def _account_upgrade_visit(self):
        """Called when leaving UPGRADE for NORMAL (close tap or
        timeout — _enter is the single funnel, so both land here). A
        visit with no buy is empty; UPGRADE_EMPTY_VISITS in a row
        ignores the button for UPGRADE_IGNORE_SECONDS. A visit with
        a buy resets the count."""

        if self._upgrade_did_item:

            self._upgrade_empty_visits = 0

            return

        self._upgrade_empty_visits += 1

        if self._upgrade_empty_visits >= UPGRADE_EMPTY_VISITS:

            self._upgrade_empty_visits = 0

            self._upgrade_ignored_until = (
                time.monotonic() + UPGRADE_IGNORE_SECONDS
            )

            logger.warning(
                "UPGRADE empty %d times in a row — "
                "ignoring `upgrade` for %.0fs",
                UPGRADE_EMPTY_VISITS,
                UPGRADE_IGNORE_SECONDS,
            )

    def _find_badged_food(self, detections):
        """First 'food' detection with a 'red_arrow' badge on it (Karahan-style gate). The badge floats above the dish, so the test is badge-center inside the dish box inflated by RED_ARROW_GATE_MARGIN — not IoU (badge and dish barely overlap)."""

        badges = [
            d
            for d in detections
            if d.get("category") == "red_arrow"
        ]

        if not badges:
            return None

        for detection in detections:

            if detection.get("category") != "food":
                continue

            if self._is_ignored_food(detection):
                continue

            if any(
                self._badge_near(badge, detection)
                for badge in badges
            ):
                return detection

        return None

    @staticmethod
    def _badge_near(badge, food):

        margin = RED_ARROW_GATE_MARGIN

        center_x = badge["x"] + badge["width"] / 2
        center_y = badge["y"] + badge["height"] / 2

        return (
            food["x"] - margin
            <= center_x
            <= food["x"] + food["width"] + margin
            and food["y"] - margin
            <= center_y
            <= food["y"] + food["height"] + margin
        )

    def _city_stuck_check(self):
        """Recovery ladder for a city going nowhere (shikhar-style escalation): 10 min without progress forces a full sweep; 15 min force-stops + relaunches the game. Only runs in NORMAL — inside a panel we are mid-task, not stuck."""

        if self.state != NORMAL:
            return

        # Game not in front (crashed, or the user opened something):
        # main.py's game-ensure owns that case. Firing here would
        # swipe and restart into the wrong app.
        if not self._game_in_foreground:
            return

        idle = time.monotonic() - self._last_progress

        if (
            idle >= CITY_STUCK_RESTART_AFTER
            and self._recovery_stage < 2
        ):

            self._recovery_stage = 2

            logger.warning(
                "no progress for %.0fs — restarting %s",
                idle,
                GAME_PACKAGE,
            )

            restarted = self.action_manager.android.restart_package(
                GAME_PACKAGE
            )

            # Fresh process, fresh eyes: whatever was wedged is
            # gone, and the relaunch needs settle time before taps.
            # The ladder rearms (stage back to 0): still stuck in
            # 10 more minutes means sweeping again, not silence.
            # The _last_progress reset above is what stops an
            # immediate refire, not the stage.
            self.reset()
            self.last_action_time = time.monotonic()
            self._last_progress = time.monotonic()
            self._recovery_stage = 0

            if not restarted:

                logger.warning(
                    "restart failed — carrying on without it"
                )

        elif (
            idle >= CITY_STUCK_SWEEP_AFTER
            and self._recovery_stage < 1
        ):

            self._recovery_stage = 1

            logger.warning(
                "no progress for %.0fs — forcing full sweep",
                idle,
            )

            # Ignored dishes get a new chance, and the next
            # update swipes immediately instead of waiting out
            # the exploration interval.
            self._reset_food_tracking()
            self.explore_anchor = 0.0
            self.explore_found = False

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
        """Something was found: POSTPONES the next swipe without resetting the sweep cycle (renamed from `_reset_exploration` — resetting the count here made the bot always sweep the same slice of the screen, since almost every swipe finds something)."""

        if not SWIPE_WAIT_FOR_NO_ACTION:
            return

        if not self.explore_found:
            self.explore_anchor = time.monotonic()
            self.explore_found = True

    def exploration_interval(self):
        """How long to wait for the next swipe: exploration_delay when nothing was found, exploration_action_delay (longer) when it was — derived from explore_found instead of stored."""

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
            return False

        if not self._can_act():
            return False

        logger.info(
            "explore: swipe %s %d/%d",
            self.swipe_direction,
            self.swipe_count + 1,
            self.max_swipes,
        )

        if not self.action_manager.swipe(
            self.swipe_direction
        ):
            return False

        self.last_action_time = now

        # Back to the NORMAL pace: from here the next swipe
        # is in exploration_delay. Finding something is what switches that
        # to the 15 s (_delay_exploration).
        self.explore_anchor = now

        self.explore_found = False

        # From here on SWIPE_WAITING_TIME applies, not
        # ACTION_SETTLE: the view just moved as a whole, and
        # no detection from before it settles is worth a tap.
        self._last_was_swipe = True

        # Real swipe: ignored items get a new chance.
        self._reset_food_tracking()

        # Exploration does not go through _act, so it must record
        # here. It is a bot decision like any other: "found
        # nothing, scrolling the screen" — and for training up and down are
        # different labels, hence the direction in the NAME.
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

        return True
