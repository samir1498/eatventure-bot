"""State machine: rule table per state, single _act function, per-state timeout."""

import logging
import time
from collections import Counter

from core import log
from core.metrics import formata_duracao
from core.config import (
    ACTION_COOLDOWN,
    ACTION_SETTLE,
    CYCLE_CATEGORIES,
    MAX_DETECTION_AGE,
    PROGRESS_ACTIONS,
    RED_ARROW_FALLBACK_CATEGORIES,
    RED_ARROW_FALLBACK_ENABLED,
    RED_ARROW_GATE_MARGIN,
    STATE_ENTRY_SETTLE,
    STATE_TIMEOUTS,
    SWIPE_WAIT_FOR_NO_ACTION,
    VIEW_MOVING_ACTIONS,
)

logger = log.get("state")


from core.failures import FailureMixin
from core.exploration import ExplorationMixin
from core.rules import (
    DELIVERY,
    FOOD,
    GRAY_DISMISS_CATEGORIES,
    HELPER,
    NEW_POINT,
    NORMAL,
    NORMAL_RULES,
    RENOVATE,
    STATE_RULES,
    UPGRADE,
)


class StateMachine(FailureMixin, ExplorationMixin):

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

        self._init_failure_tracking()

        self._init_exploration_tracking()

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

        # Avoid repeating the stale-detection warning on every frame.
        self._stale_warned = 0.0

        # Foreground gate: while the game is not focused (settings,
        # Gmail, Play sheet) no action may fire. Set from main.py's
        # game-ensure check; defaults to True so tests and the first
        # frames behave as before.
        self._game_in_foreground = True
        self._foreground_warned = 0.0

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

            self._track_stuck_popups(detections)

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

                # Orphan-badge fallback: the upgrade badge shows but
                # its station does not (panda over the plot, empty
                # dashed outline) — tap the badge center as a proxy
                # instead of stalling. Food keeps priority: the food
                # proxy is resolved once here, so a new_point proxy
                # never steals a badge the food row would claim, and
                # fires once food gave up on it.
                if (
                    category in RED_ARROW_FALLBACK_CATEGORIES
                    and self.state == NORMAL
                ):

                    food_proxy = (
                        self._find_fallback_proxy(detections, "food")
                        if category == "new_point"
                        else None
                    )

                    if food_proxy is None:

                        detection = self._find_fallback_proxy(
                            detections,
                            category,
                        )

                if detection is None:
                    continue

            # Red-badge preference: dishes WITH a red upgrade badge are
            # upgradeable — prefer them, but fall back to any visible
            # dish when no badge is detected (badge templates miss on
            # some views; a strict gate blinded the bot entirely).
            if (
                category == "food"
                and self.state == NORMAL
                and not detection.get("fallback")
            ):

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
            self._panel_target = detection
            self.up_food_wait_start = time.monotonic()

        # Panel target only: the timer arms lazily on the first
        # NEW_POINT frame (panel-open grace), see _new_point —
        # except badge-fallback taps, which never open a panel
        # (same city view), so they arm at tap time like food.
        if action == "new_point":
            self._panel_target = detection
            if detection is not None and detection.get("fallback"):
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
        if self._panel_target is not None:

            if self.state == FOOD:
                self._register_give_up(self._panel_target, "food")

            elif self.state == NEW_POINT:
                self._register_give_up(self._panel_target, "new_point")

            self._panel_target = None

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

        # Entry grace: the panel opens after the tap, so the give-up
        # clock starts on the first panel frame, not on the tap.
        if self.up_food_wait_start is None:
            self.up_food_wait_start = time.monotonic()

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

    def _find(self, detections, category):
        """First detection of the category. No threshold refiltering here on purpose: a second threshold once lived in this method, diverging from the detector's, and the winner was the first in filename alphabetical order, not the best."""

        for detection in detections:

            if detection["category"] != category:
                continue

            if category in ("food", "new_point") and self._is_ignored(
                detection, category
            ):
                continue

            return detection

        return None

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

            if self._is_ignored(detection, "food"):
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

    def _find_fallback_proxy(self, detections, category):
        """Badge-as-station proxy: red_arrow with no matching station.

        Returns a synthetic detection of `category` on the badge box,
        so the normal action (and its give-up/ignore guard) applies
        unchanged. Skips badges already ignored for that action."""

        if not RED_ARROW_FALLBACK_ENABLED:
            return None

        if category not in RED_ARROW_FALLBACK_CATEGORIES:
            return None

        if not detections:
            return None

        for badge in detections:

            if badge.get("category") != "red_arrow":
                continue

            proxy = self._badge_proxy(badge, category)

            if self._is_ignored(proxy, category):
                continue

            return proxy

        return None

    @staticmethod
    def _badge_proxy(badge, category):
        """Synthetic station detection centered on the badge."""

        return {
            "category": category,
            "name": "red_arrow_fallback",
            "confidence": badge.get("confidence", 0.0),
            "color_similarity": badge.get("color_similarity", 0.0),
            "x": badge["x"],
            "y": badge["y"],
            "width": badge["width"],
            "height": badge["height"],
            "fallback": "red_arrow",
        }
