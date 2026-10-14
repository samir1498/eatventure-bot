"""Failure tracking: give-up timers, ignore lists, stuck-popup suppression.

Mixed into StateMachine — same object, same fields, only the source
file changes. Everything here answers "this did not work, what now":
counting consecutive give-ups, ignoring dead items until a swipe,
suppressing panels that never resolve, and the dismiss escalation
ladder.
"""

import time

from core import log
from core.config import (
    BOX_STUCK_UPDATES,
    BOX_SUPPRESS_UPDATES,
    DISMISS_ACTIONS,
    DISMISS_ATTEMPTS_BEFORE_SCROLL,
    FOOD_IGNORE_IOU,
    FOOD_IGNORE_TTL,
    FOOD_MAX_ATTEMPTS,
    GRAY_STUCK_UPDATES,
    GRAY_SUPPRESS_UPDATES,
    REPEATED_ACTION_WARNING,
    UP_FOOD_WAIT,
    UPGRADE_EMPTY_VISITS,
    UPGRADE_FOOD_DISMISS_ROUNDS,
    UPGRADE_FOOD_MAX_REPEATS,
    UPGRADE_FOOD_STUCK_UPDATES,
    UPGRADE_FOOD_SUPPRESS_UPDATES,
    UPGRADE_IGNORE_SECONDS,
)
from core.rules import (
    FOOD,
    GRAY_DISMISS_CATEGORIES,
    NEW_POINT,
    NORMAL,
)

logger = log.get("state")


class FailureMixin:
    """Give-up/ignore/suppress state and logic. See module docstring."""

    def _init_failure_tracking(self):

        # Wait after the food long press.
        self.up_food_wait_start = None

        # Current panel target ("food" dish or "new_point" slot),
        # waiting for up_food. Stored here (and not only in the local
        # detection) because _wait_or_give_up needs it to know WHICH
        # item to ignore when giving up.
        self._panel_target = None

        # Consecutive-give-up accounting per action ("food" dish,
        # "new_point" slot): consecutive misses on the same item,
        # ignored entries, each with its own counter — a badge food
        # gave up on is retried as new_point, and vice versa must
        # not double-count. Ignored entries last until the next real
        # swipe, otherwise the bot tries the same item forever
        # (finding it keeps postponing the swipe).
        self._give_up = {
            kind: {"fail_target": None, "fail_count": 0, "ignored": []}
            for kind in ("food", "new_point")
        }

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

        # Consecutive attempts at closing something. Climbs the
        # escalation ladder instead of repeating what did not work.
        self._dismiss_attempts = 0

        # How many times it already had to scroll to escape.
        self._dismiss_rounds = 0

        # Empty-Upgrades-panel guard: consecutive UPGRADE visits with
        # no buy, and until when the `upgrade` button stays ignored.
        # Same idea as the _give_up ignore lists, but time-boxed instead of
        # swipe-reset (the button is fixed nav chrome — swipes never
        # move it, so a swipe reset would never stick).
        self._upgrade_empty_visits = 0
        self._upgrade_ignored_until = 0.0

        # Whether the current UPGRADE visit bought anything yet.
        self._upgrade_did_item = False

    def _track_stuck_popups(self, detections):
        """Stuck-popup suppression counters (up_food / gray pair /
        reward chest). Extracted verbatim from update(): visible N
        updates without resolving = unresolvable panel or paywall,
        so the category is suppressed and lower rules get a turn."""

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

        if self._panel_target is not None:

            if self.state == FOOD:
                self._register_give_up(self._panel_target, "food")

            elif self.state == NEW_POINT:
                self._register_give_up(self._panel_target, "new_point")

            self._panel_target = None

        self._enter(NORMAL)

    def _register_give_up(self, target, kind):
        """Counts CONSECUTIVE give-ups on the same item (by IoU); on reaching FOOD_MAX_ATTEMPTS, ignores it until the next swipe or FOOD_IGNORE_TTL passing, whichever comes first — before that, returns to NORMAL and lets it try again (it may just be up_food slow to appear). Per-kind counters: a badge food gave up on gets a full new_point chance, not a shared one."""

        tracked = self._give_up[kind]

        if (
            tracked["fail_target"] is not None
            and self._iou(target, tracked["fail_target"])
            >= FOOD_IGNORE_IOU
        ):
            tracked["fail_count"] += 1

        else:
            tracked["fail_count"] = 1

        tracked["fail_target"] = target

        if tracked["fail_count"] < FOOD_MAX_ATTEMPTS:
            return

        logger.warning(
            "'%s' without up_food in %d consecutive attempts — "
            "ignoring until the next swipe",
            kind,
            FOOD_MAX_ATTEMPTS,
        )

        tracked["ignored"].append((target, time.monotonic()))

        tracked["fail_target"] = None
        tracked["fail_count"] = 0

    def _reset_food_tracking(self):
        """Forgets ignored items and the attempt count — called on every real swipe, which changes what is on screen."""

        for tracked in self._give_up.values():
            tracked["ignored"] = []
            tracked["fail_target"] = None
            tracked["fail_count"] = 0

    def _is_ignored(self, detection, kind):
        """Same item as one that already exhausted UP_FOOD_WAIT without up_food (same on-screen position, compared by IoU since the screen does not move outside a swipe). Entries expire after FOOD_IGNORE_TTL — a bot kept busy by other chrome never swipes, and swipe-only expiry would ignore the dish forever. Per-kind lists: a badge food gave up on is retried as new_point."""

        now = time.monotonic()

        tracked = self._give_up[kind]

        alive = [
            (ignored, ignored_at)
            for ignored, ignored_at in tracked["ignored"]
            if now - ignored_at < FOOD_IGNORE_TTL
        ]

        tracked["ignored"] = alive

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
