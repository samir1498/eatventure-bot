"""Central EatVenture AI configuration — every tunable value lives here, no other module should have magic numbers."""

import os
import platform
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_IS_WINDOWS = platform.system() == "Windows"
_EXE_SUFFIX = ".exe" if _IS_WINDOWS else ""


# ==== DEVICE ====

# None = detect at runtime; pin it to avoid repeated prompts when connected via USB+WiFi simultaneously.
DEVICE_SERIAL = os.environ.get("DEVICE_SERIAL")


# ==== GAME ====

# Game package: the bot keeps it in the foreground and
# relaunches it alone if an ad or the system takes it away.
GAME_PACKAGE = "com.hwqgrhhjfd.idlefastfood"

# How often to check the focused app (dumpsys costs ~50ms,
# so it cannot run on every cycle). Kept tight on purpose:
# between checks the bot would otherwise keep tapping
# whatever is on screen (settings, Gmail, Play sheet).
GAME_ENSURE_INTERVAL = 5.0

# Minimum wait between two relaunches (a game still opening does not
# count as "out of foreground" again).
GAME_RELAUNCH_COOLDOWN = 30.0


# ==== SCRCPY ====

_scrcpy_dir = PROJECT_ROOT / "tools" / "scrcpy"
_SERVER_NAMES = ("scrcpy-server*.jar", "scrcpy-server")


def _find_server(base: Path):
    for pattern in _SERVER_NAMES:
        try:
            hit = next((p for p in base.rglob(pattern) if p.is_file()), None)
        except Exception:
            hit = None
        if hit is not None:
            return hit
    return None


_scrcpy_path = shutil.which("scrcpy")
if _scrcpy_path:
    SCRCPY_PATH = _scrcpy_path
    try:
        _jar = _find_server(Path(_scrcpy_path).parent)
    except Exception:
        _jar = None
    if _jar is None and _scrcpy_dir.exists():
        _jar = _find_server(_scrcpy_dir)
    SCRCPY_SERVER_PATH = str(_jar) if _jar is not None else r""
else:
    SCRCPY_PATH = r""
    SCRCPY_SERVER_PATH = r""
    if _scrcpy_dir.exists():
        try:
            _exe = next(_scrcpy_dir.rglob(f"scrcpy{_EXE_SUFFIX}"), None)
            _jar = _find_server(_scrcpy_dir)
            SCRCPY_PATH = str(_exe) if _exe is not None else r""
            SCRCPY_SERVER_PATH = str(_jar) if _jar is not None else r""
        except Exception:
            pass

SCRCPY_SERVER_VERSION = "5.0.1"
SCRCPY_PORT = 27283
SCRCPY_DEVICE_JAR = "/data/local/tmp/eatventure-server.jar"


# ==== ADB ====

_adb_system = shutil.which("adb")
if _adb_system:
    ADB_PATH = _adb_system
else:
    _adb_from_scrcpy = None
    try:
        if SCRCPY_PATH:
            scrcpy_p = Path(SCRCPY_PATH)
            candidate = scrcpy_p.parent / f'adb{_EXE_SUFFIX}'
            if candidate.exists():
                _adb_from_scrcpy = str(candidate)
            else:
                sibling = scrcpy_p.parent / 'platform-tools' / f'adb{_EXE_SUFFIX}'
                if sibling.exists():
                    _adb_from_scrcpy = str(sibling)
    except Exception:
        _adb_from_scrcpy = None

    if not _adb_from_scrcpy:
        _scrcpy_dir_adb = PROJECT_ROOT / 'tools' / 'scrcpy'
        if _scrcpy_dir_adb.exists():
            _adb_candidate = next(_scrcpy_dir_adb.rglob(f'adb{_EXE_SUFFIX}'), None)
            if _adb_candidate:
                _adb_from_scrcpy = str(_adb_candidate)

    ADB_PATH = _adb_from_scrcpy if _adb_from_scrcpy else "adb"


# ==== CAPTURE ====

CAPTURE_CONNECT_TIMEOUT = 12.0
CAPTURE_START_TIMEOUT = 6.0
CAPTURE_MAX_FPS = 10


# ==== UI ====

SHOW_AI_VISION = False
SHOW_SCRCPY = False
SHOW_DETECTION_LABELS = True
SHOW_DETECTION_LAG = True
SHOW_CYCLE_TIME = True
SHOW_BATTERY = True
SHOW_FPS = True

STATUS_PANEL = True
STATUS_PANEL_INTERVAL = 0.5

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 900
AI_WINDOW_NAME = "EatVenture - AI"
AI_WINDOW_POSITION = (600, 50)

SCRCPY_EXTRA_ARGS = [
    "--capture-orientation", "90",
    "--no-audio",
    "--video-bit-rate", "2M",
    "--max-size", "1024",
]

SELECTOR_HEIGHT_FRACTION = 0.80
SELECTOR_WIDTH_FRACTION = 0.95
SELECTOR_MAX_WIDTH = None
SELECTOR_MAX_HEIGHT = None
SELECTOR_FALLBACK_WIDTH = 1600
SELECTOR_FALLBACK_HEIGHT = 1000


# ==== STATUS & METRICS ====

CYCLE_CATEGORIES = {"build", "plane"}
CYCLE_STALL_FACTOR = 3.0
BATTERY_POLL_INTERVAL = 30.0
BATTERY_WARNING_LEVEL = 20


# ==== DATASET ====

DATASET_SAVE = False
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_IMAGE_FORMAT = "jpg"
DATASET_JPEG_QUALITY = 92

DATASET_ACTIONS = {
    "click",
    "close",
    "food",
    "new_point",
    "new_point_click",
    "open_box",
    "open_renovate",
    "open_store_click",
    "plane",
    "renovate_click",
    "upgrade",
    "upgrade_item",
    "upgrade_food",
    "gray_max",
    "gray_coin",
    "dismiss",
    "scroll_bottom",
    "swipe_up",
    "swipe_down",
}

DATASET_NEGATIVE_INTERVAL = 20.0
DATASET_HASH_SIZE = 8
DATASET_DEDUPE_MEMORY = 200
DATASET_MAX_UNCHANGED_STREAK = 3
DATASET_SCREEN_CHANGE = 8.0
DATASET_QUEUE_SIZE = 8
DATASET_OUTCOME_TIMEOUT = 3.0
DATASET_MAX_SAMPLES = 0
DATASET_MAX_DISK_MB = 20_000

DATASET_DB_ENABLED = os.environ.get(
    "EATVENTURE_DB_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
DATASET_DB_DSN = os.environ.get("EATVENTURE_DB_DSN", "")
DATASET_DB_SCHEMA = "public"
DATASET_DB_BATCH = 20


# ==== LOGGING ====

LOG_LEVEL = "INFO"


# ==== TEMPLATES ====

REFERENCE_WIDTH = 1080
REFERENCE_HEIGHT = 2400

TEMPLATE_SCALE_BASIS = "short_side"
TEMPLATE_SCALE_STEPS = (0.92, 1.0, 1.08)

SHAPE_THRESHOLD = 0.80
COLOR_THRESHOLD = 0.80

# How many scales of TEMPLATE_SCALE_STEPS each pass covers per template.
# Once a template hits on one scale, the others are waste: the
# device scale does not change on its own. The pass uses only the winning scale and
# reopens the comb when the category goes ESCALA_LIBERA_SECAS passes
# without finding anything (game camera zoom changed, or never calibrated).
ESCALA_UNICA_POR_TEMPLATE = True
ESCALA_LIBERA_SECAS = 6

# Cap of approved detections per category in one pass: the StateMachine acts
# on ONE detection per frame (_apply_rules), so scanning all 86
# food templates after already holding 6 dishes changes no decision. Only cuts
# AFTER finding something — a partial pass never becomes an "empty screen", which would trigger
# _explore_screen. Leaves margin for _is_ignored to discard a few.
# A category outside the dict is scanned in full.
CATEGORY_MATCH_BUDGET = {
    "food": 6,
    "new_point": 3,
}

# How many templates of the category the pass looks at before stopping, and ONLY if it
# already found something. Without this, the cap above does not protect the case where the
# first templates in the queue miss: the pass would scan all 86 food templates
# even with 6 dishes on screen. Nothing found = full scan, always
# (cutting there would blind the bot).
CATEGORY_SCAN_QUOTA = {
    "food": 20,
    "new_point": 12,
}

# How long a template that hit stays at the front of its category queue.
# A restaurant has ~6-10 possible dishes among the 86 food templates;
# looking for recently seen ones before the others makes the cap
# above be reached on the first templates, not the fiftieth.
QUENTES_TTL = 30.0

CATEGORY_THRESHOLDS = {
    "build": 0.95,
    "new_point": 0.95,
    "food": 0.90,
    "red_arrow": 0.90,
    "upgrade": 0.90,
    "up_food": 0.90,
    "up_upgrade": 0.90,
    "close": 0.85,
    "plane": 0.98,
    "box": 0.90,
    "delivery": 0.95,
    "helper": 0.95,
}

CATEGORY_COLOR_THRESHOLDS = {
    "delivery": 0.99
}


CATEGORY_ROIS = {}

# When detecting via default template, saves the crop at the exact
# device resolution (even on the reference) — later passes use that override
# instead of the default.
LEARN_RESOLUTION_TEMPLATES = False

# How often to check whether the template folder changed (new,
# edited, or removed template) and reload without restarting the bot. 0 disables.
TEMPLATES_WATCH_INTERVAL = 0


# ==== DETECTOR ====

DETECTOR_DEBUG_MISSES = False
DETECTOR_DEBUG_INTERVAL = 3.0

# Scale cap of the coarse stage. _coarse_scale_for returns
# min(COARSE_SCALE, grade), so this only binds SMALL templates: with
# cap 0.40, every template with side < 30px would fall out of the coarse stage
# (12/0.40) and go to full-resolution matchTemplate on the WHOLE SCREEN, the
# most expensive detector path. Large templates are unaffected (100px side -> 0.15).
COARSE_SCALE = 0.70
COARSE_SCALE_MIN = 0.10
COARSE_SCALE_STEP = 0.05
COARSE_MARGIN = 0.18

REFINE_SLACK = 8
MAX_MATCHES_PER_TEMPLATE = 12
NMS_IOU = 0.35


# ==== ACTIONS ====

UPGRADE_ITEM_CLICKS = 10
UPGRADE_FOOD_PRESS = 4.0

DISMISS_POINT = (10, 2200)
# Left-edge middle: deliberately off the top-right corner, where
# the settings gear lives — a gray_coin misfire must never open it.
GRAY_COIN_POINT = (10, 1200)
ACTION_POINTS = {
    "gray_coin": GRAY_COIN_POINT,
}

DISMISS_ACTIONS = {"dismiss", "gray_max", "gray_coin"}
DISMISS_ATTEMPTS_BEFORE_SCROLL = 5
DISMISS_HOLD_DURATION = 0.4

# Food-trap guard (NORMAL only): consecutive upgrade_food
# presses that never resolve fall back to fixed-point dismiss.
UPGRADE_FOOD_MAX_REPEATS = 3
UPGRADE_FOOD_DISMISS_ROUNDS = 2

# Suppression (NORMAL only): up_food visible this many
# consecutive updates without resolving = stuck popup (e.g.
# unaffordable price) -> suppress up_food so lower-priority
# upgrade rules get a turn at other stations; then up_food
# gets another chance. Presence-based (not action-based) so
# interleaved side-actions can't hide the stuck panel.
UPGRADE_FOOD_STUCK_UPDATES = 25
UPGRADE_FOOD_SUPPRESS_UPDATES = 40

# Same treatment one priority level up: gray_coin/gray_max (#4)
# outrank up_food (#5), so a greyed-out price can starve the
# whole panel the same way. Suppress the gray dismiss pair so
# lower rules (up_food, station upgrades) get a turn.
GRAY_STUCK_UPDATES = 25
GRAY_SUPPRESS_UPDATES = 40

# Same treatment for the reward chest: a "box" that stays visible
# for N updates without resolving is a paid offer (e.g. starter
# pack), not free loot — suppress open_box so the bot backs off
# instead of tapping inside the paywall, then tries again later.
BOX_STUCK_UPDATES = 100
BOX_SUPPRESS_UPDATES = 20

SCROLL_BOTTOM_DIRECTION = "up"
SCROLL_BOTTOM_SWIPES = 6

SWIPE_X = 540
SWIPE_Y = 1200
SWIPE_DISTANCE = 700
SWIPE_DURATION_MS = 500


# ==== VISION ====

VISION_INTERVAL = 0.2
VISION_FILTER_BY_STATE = True
VISION_PRIORITY_STOP = True
MAX_DETECTION_AGE = 10.0

# Categories that never TRIGGER the priority stop: searched even
# after a higher-priority category matched, and matching them does
# not stop the pass either. For gating-only categories (no rule row
# ever fires them): stopping at one would hide the actionable
# categories below it, and stopping before one would blind the gate.
#
# `upgrade` is different: it HAS a rule row (it opens the Upgrades
# panel), but the button is permanent chrome that matches on every
# frame. Letting it stop the pass hid `delivery`, `new_point` and
# `food` behind it, so the bot saw one detection and swiped forever.
CATEGORY_NO_PRIORITY_STOP = {"red_arrow", "upgrade", "delivery"}


# ==== TIMING ====

ACTION_COOLDOWN = 0.5
ACTION_SETTLE = 0.4
SWIPE_WAITING_TIME = 0.5
VIEW_MOVING_ACTIONS = {"scroll_bottom"}


# ==== EXPLORATION ====

SWIPE_WAIT_FOR_NO_ACTION = True
EXPLORATION_DELAY = 5.0
EXPLORATION_DELAY_AFTER_ACTION = 15.0
MAX_SWIPES = 5
SWIPE_START_DIRECTION = "down"


# ==== FOOD ====

UP_FOOD_WAIT = 2.0
GIVE_UP_IGNORE_IOU = 0.5
GIVE_UP_MAX_ATTEMPTS = 3

# Ignored dishes get a second chance after this long even with no
# swipe: a bot kept busy by other chrome postpones exploration
# forever, and swipe-only expiry would blind it to the dish
# permanently.
GIVE_UP_IGNORE_TTL = 300.0


# ==== UPGRADE ====

# Consecutive UPGRADE visits with zero upgrade_item taps before the
# `upgrade` button is ignored: the panel was empty ("You have all
# upgrades!") and the button is permanent nav chrome — without this
# the bot reopens the empty panel forever.
#
# Trip FAST. One visit cycle is ~2.8s (open panel, no buy, close), so
# every extra visit before the trip is ~3s of the bot opening an empty
# panel instead of unlocking stations. 8 visits cost 26s of startup.
# 3 is the minimum that can be called "conclusively empty".
UPGRADE_EMPTY_VISITS = 3

# How long the button stays ignored. Cleared early by a renovation
# door (new city = new general upgrades to buy).
#
# Short on purpose: while ignored, `upgrade` is dropped from the
# detector request (see wanted_categories), so the window costs one
# panel open and buys back a full sweep of real work. 60s = one
# re-check per minute, which is not spam; 1800s = half an hour of
# missing a newly affordable global upgrade.
UPGRADE_IGNORE_SECONDS = 60.0

# Red-arrow badge gate: a "food" detection in NORMAL only becomes a
# tap when a "red_arrow" badge overlaps it (badge center inside the
# dish box inflated by this margin, px at reference resolution).
# Without the badge the dish is not upgradeable — tapping it anyway
# is what made the bot click the same dish like crazy.
RED_ARROW_GATE_MARGIN = 100

# Badge-only fallback: red_arrow visible but its station is not
# detected (occlusion, e.g. pet standing over the plot; empty dashed
# plots read as nothing). Taps the badge center as a station proxy
# through the normal food/new_point actions, so the give-up timer and
# GIVE_UP_MAX_ATTEMPTS ignore still guard loops. Food proxies fire
# first; a badge food gave up on is retried as new_point.
RED_ARROW_FALLBACK_ENABLED = True
RED_ARROW_FALLBACK_CATEGORIES = ("food", "new_point")


# ==== CITY-STUCK RECOVERY ====

# Actions that prove the city is moving forward (money spent or a
# new item unlocked). Anything else (taps, dismisses, swipes) does
# not reset the stuck clock.
PROGRESS_ACTIONS = {
    "upgrade_food",
    "upgrade_item",
    "new_point_click",
    "renovate_click",
}

# No progress for this long in NORMAL: force a full sweep (clears
# ignored dishes, resets the exploration anchor so the next update
# swipes immediately).
CITY_STUCK_SWEEP_AFTER = 600.0

# Still nothing after this long: force-stop + relaunch the game
# (bigger hammer, same idea as shikhar's stale restart).
CITY_STUCK_RESTART_AFTER = 900.0


# ==== STATE MACHINE ====

REPEATED_ACTION_WARNING = 8

STATE_ENTRY_SETTLE = {
    "UPGRADE": 1,
    "DELIVERY": 1,
    "HELPER": 1,
    "GRAY_MAX": 1,
}

STATE_TIMEOUTS = {
    "RENOVATE": 15.0,
    "UPGRADE": 5.0,
    "DELIVERY": 5.0,
    "HELPER": 5.0,
    "NEW_POINT": 5.0,
    "FOOD": 5.0,
}
