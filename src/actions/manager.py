"""ActionManager: table-driven dispatch, async execution on its own thread."""

import queue
import threading
import time

from actions.android import AndroidActions
from core import log
from core.config import (
    ACTION_POINTS,
    DISMISS_HOLD_DURATION,
    DISMISS_POINT,
    SCROLL_BOTTOM_DIRECTION,
    SCROLL_BOTTOM_SWIPES,
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    SWIPE_DISTANCE,
    SWIPE_DURATION_MS,
    SWIPE_X,
    SWIPE_Y,
    UPGRADE_FOOD_PRESS,
    UPGRADE_ITEM_CLICKS,
)

logger = log.get("action")


CLICK = "click"        # tap at the detection center
REPEAT = "repeat"      # several taps
PRESS = "press"        # long tap
DISMISS = "dismiss"    # tap at a fixed point
HOLD = "hold"          # held tap at a fixed point
SCROLL = "scroll"      # scroll screen to the end

SEM_DETECCAO = {DISMISS, HOLD, SCROLL}  # Actions without detection


# Action table: name -> behavior (handler deduplication)
ACTION_TABLE = {
    "click": CLICK,
    "food": CLICK,
    "plane": CLICK,
    "delivery": CLICK,
    "accept_delivery": CLICK,
    "helper": CLICK,
    "invite_helper": CLICK,
    "upgrade": CLICK,
    "close": CLICK,
    "open_box": CLICK,
    "new_point": CLICK,
    "new_point_click": CLICK,
    "open_renovate": CLICK,
    "renovate_click": CLICK,
    "open_store_click": CLICK,

    "upgrade_item": REPEAT,

    "upgrade_food": PRESS,

    "gray_max": DISMISS,  # Close unwanted panel with a tap
    "dismiss": HOLD,      # Same but with a held tap

    "gray_coin": DISMISS,  # Tap at a different fixed point (ACTION_POINTS)

    "scroll_bottom": SCROLL,  # Scroll to empty (closes nothing)
}


def ponto_fixo(action):
    """Point in REFERENCE coordinates. Module-level function to keep dataset and dispatch in sync."""

    return ACTION_POINTS.get(action, DISMISS_POINT)


class ActionManager:

    def __init__(self, serial=None):

        self.android = AndroidActions(serial)

        # Detection in frame coords, tap in device coords (may differ if stream is scaled down)
        self.device_size = None
        self.frame_size = None

        self.queue = queue.Queue(maxsize=1)

        self.running = False
        self.thread = None

        self.busy = False
        self.busy_lock = threading.Lock()

        # When the last action finished (settle counts from here, not from submission)
        self.last_finished_at = 0.0

    def start(self):

        if self.running:
            return

        self.device_size = self.android.get_screen_size()

        if self.device_size:

            logger.info(
                "Device: %dx%d",
                self.device_size[0],
                self.device_size[1],
            )

        else:

            self.device_size = (
                REFERENCE_WIDTH,
                REFERENCE_HEIGHT,
            )

            logger.warning(
                "Unknown device resolution, "
                "assuming %dx%d",
                REFERENCE_WIDTH,
                REFERENCE_HEIGHT,
            )

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            name="action-worker",
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

        # Unblock the worker if it is waiting on the queue.
        try:
            self.queue.put_nowait(None)

        except queue.Full:
            pass

        if self.thread:

            self.thread.join(
                timeout=UPGRADE_FOOD_PRESS + 3.0
            )

            self.thread = None

    def set_frame_size(self, width, height):
        """Resolution the detections are computed at."""

        size = (width, height)

        if size == self.frame_size:
            return

        self.frame_size = size

        if self.device_size and size != self.device_size:

            logger.warning(
                "Frame %dx%d != device %dx%d — "
                "converting tap coordinates.",
                width,
                height,
                self.device_size[0],
                self.device_size[1],
            )

    def _from_frame(self, x, y):
        """Frame coords -> device coords."""

        if not self.frame_size or not self.device_size:
            return int(x), int(y)

        frame_width, frame_height = self.frame_size
        device_width, device_height = self.device_size

        if not frame_width or not frame_height:
            return int(x), int(y)

        return (
            int(round(x * device_width / frame_width)),
            int(round(y * device_height / frame_height)),
        )

    def _from_reference(self, x, y):
        """Reference resolution (where the fixed points were annotated) -> device."""

        if not self.device_size:
            return int(x), int(y)

        device_width, device_height = self.device_size

        return (
            int(round(x * device_width / REFERENCE_WIDTH)),
            int(round(y * device_height / REFERENCE_HEIGHT)),
        )

    def _reference_to_frame(self, x, y):
        """Reference -> FRAME. Only used for the dataset label, not for tapping."""

        if not self.frame_size:
            return int(x), int(y)

        frame_width, frame_height = self.frame_size

        return (
            int(round(x * frame_width / REFERENCE_WIDTH)),
            int(round(y * frame_height / REFERENCE_HEIGHT)),
        )

    def target_frame(self, action, detection):
        """Where the action taps, in FRAME coords (for the dataset). None when there is no point target (scroll).

        Separate from _dispatch: the real tap converts directly from reference
        to device, without going through the frame (avoids rounding on real money).
        """

        kind = ACTION_TABLE.get(action)

        if kind is None or kind == SCROLL:
            return None

        if kind in (DISMISS, HOLD):

            return self._reference_to_frame(
                *ponto_fixo(action)
            )

        if detection is None:
            return None

        return (
            detection["x"] + detection["width"] // 2,
            detection["y"] + detection["height"] // 2,
        )

    def kind_of(self, action):
        return ACTION_TABLE.get(action)

    def is_busy(self):

        with self.busy_lock:
            return self.busy

    def execute(self, action, detection):
        """Enqueues an action (non-blocking). False if an action is already in progress."""

        kind = ACTION_TABLE.get(action)

        if kind is None:

            logger.warning(
                "Unknown action: %s",
                action,
            )

            return False

        if kind not in SEM_DETECCAO and detection is None:
            return False

        return self._submit(
            ("action", action, kind, detection)
        )

    def swipe(self, direction):

        return self._submit(
            ("swipe", direction, None, None)
        )

    def _submit(self, job):

        with self.busy_lock:

            if self.busy:
                return False

            self.busy = True

        try:

            self.queue.put_nowait(job)

        except queue.Full:

            with self.busy_lock:
                self.busy = False

            return False

        return True

    def _run(self):

        while self.running:

            try:

                job = self.queue.get(timeout=0.2)

            except queue.Empty:
                continue

            if job is None:
                break

            try:

                self._dispatch(job)

            except Exception as error:

                logger.exception(
                    "Error executing %s: %s",
                    job,
                    error,
                )

            finally:

                with self.busy_lock:

                    # Before releasing busy: whoever wakes up
                    # seeing "free" must also see WHEN
                    # it became free.
                    self.last_finished_at = time.monotonic()

                    self.busy = False

    def _dispatch(self, job):

        job_type, name, kind, detection = job

        if job_type == "swipe":

            self._swipe(name)

            return

        if detection is not None:

            center_x = (
                detection["x"]
                + detection["width"] // 2
            )

            center_y = (
                detection["y"]
                + detection["height"] // 2
            )

            x, y = self._from_frame(center_x, center_y)

        else:

            x, y = 0, 0

        if kind == CLICK:

            self._click(name, x, y)

        elif kind == REPEAT:

            self._repeat(name, x, y)

        elif kind == PRESS:

            self._press(name, x, y)

        elif kind == DISMISS:

            x, y = self._from_reference(*ponto_fixo(name))

            self._click(name, x, y)

        elif kind == HOLD:

            x, y = self._from_reference(*ponto_fixo(name))

            self._hold(name, x, y)

        elif kind == SCROLL:

            self._scroll_bottom(name)

    def _click(self, name, x, y):

        logger.info(
            "%s -> click (%d, %d)",
            name,
            x,
            y,
        )

        self.android.click(x, y)

    def _repeat(self, name, x, y):
        """N clicks in a single adb command (avoids N-process overhead)."""

        logger.info(
            "%s -> %d clicks (%d, %d)",
            name,
            UPGRADE_ITEM_CLICKS,
            x,
            y,
        )

        self.android.tap_many(x, y, UPGRADE_ITEM_CLICKS)

    def _hold(self, name, x, y):
        """Short held tap at a fixed point (unlike _press, which holds at the detection center)."""

        logger.info(
            "%s -> hold (%d, %d) for %.2fs",
            name,
            x,
            y,
            DISMISS_HOLD_DURATION,
        )

        self.android.press(
            x,
            y,
            duration=DISMISS_HOLD_DURATION,
        )

    def _press(self, name, x, y):

        logger.info(
            "%s -> press (%d, %d) for %.1fs",
            name,
            x,
            y,
            UPGRADE_FOOD_PRESS,
        )

        self.android.press(
            x,
            y,
            duration=UPGRADE_FOOD_PRESS,
        )

    def _scroll_bottom(self, name):
        """Swipe sequence to the end of the screen. Runs on the action thread, does not block the main loop."""

        logger.info(
            "%s -> %d swipes '%s' (scrolling to end)",
            name,
            SCROLL_BOTTOM_SWIPES,
            SCROLL_BOTTOM_DIRECTION,
        )

        if SCROLL_BOTTOM_DIRECTION == "up":
            end_y = SWIPE_Y - SWIPE_DISTANCE

        else:
            end_y = SWIPE_Y + SWIPE_DISTANCE

        start_x, start_y = self._from_reference(
            SWIPE_X,
            SWIPE_Y,
        )

        _, target_y = self._from_reference(SWIPE_X, end_y)

        self.android.swipe_many(
            start_x,
            start_y,
            start_x,
            target_y,
            SCROLL_BOTTOM_SWIPES,
            SWIPE_DURATION_MS,
        )

    def _swipe(self, direction):

        if direction == "up":

            end_y = SWIPE_Y - SWIPE_DISTANCE

        elif direction == "down":

            end_y = SWIPE_Y + SWIPE_DISTANCE

        else:

            logger.warning(
                "Invalid swipe direction: %s",
                direction,
            )

            return

        start_x, start_y = self._from_reference(
            SWIPE_X,
            SWIPE_Y,
        )

        _, target_y = self._from_reference(
            SWIPE_X,
            end_y,
        )

        logger.info(
            "swipe %s (%d -> %d)",
            direction,
            start_y,
            target_y,
        )

        self.android.swipe(
            start_x,
            start_y,
            start_x,
            target_y,
            SWIPE_DURATION_MS,
        )
