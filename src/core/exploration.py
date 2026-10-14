"""Exploration and recovery: swipe pacing, sweep cycle, city-stuck ladder."""

import time

from core import log
from core.config import (
    CITY_STUCK_RESTART_AFTER,
    CITY_STUCK_SWEEP_AFTER,
    EXPLORATION_DELAY,
    EXPLORATION_DELAY_AFTER_ACTION,
    GAME_PACKAGE,
    MAX_SWIPES,
    SWIPE_START_DIRECTION,
    SWIPE_WAIT_FOR_NO_ACTION,
    SWIPE_WAITING_TIME,
)
from core.rules import NORMAL

logger = log.get("state")


class ExplorationMixin:
    """Owns nothing at rest — all state lives on the machine (shared
    self namespace, same as FailureMixin). Moved verbatim."""

    def _init_exploration_tracking(self):

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

        # City-stuck recovery: last time an action proved the city
        # is moving forward (PROGRESS_ACTIONS or a renovation
        # door). Recovery stage 0 = none, 1 = forced sweep done,
        # 2 = app restarted. Reset by real progress.
        self._last_progress = time.monotonic()
        self._recovery_stage = 0

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

    def reset(self):

        self._enter(NORMAL)

        self.last_action_time = 0.0

        self._panel_target = None
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
