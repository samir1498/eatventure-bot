"""Configuração central do EatVenture AI — todo valor ajustável mora aqui, nenhum outro módulo deve ter número mágico."""

import os
import platform
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_IS_WINDOWS = platform.system() == "Windows"
_EXE_SUFFIX = ".exe" if _IS_WINDOWS else ""


# ==== DEVICE ====

# None = detecta em runtime; fixe para evitar prompt repetido quando conectado via USB+WiFi simultaneamente.
DEVICE_SERIAL = None


# ==== GAME ====

# Pacote do jogo: o bot garante que ele está em primeiro plano e o
# relança sozinho se um anúncio ou o sistema o tirar da frente.
GAME_PACKAGE = "com.hwqgrhhjfd.idlefastfood"

# De quanto em quanto tempo checar o app em foco (dumpsys custa ~50ms,
# não pode entrar em todo ciclo).
GAME_ENSURE_INTERVAL = 15.0

# Espera mínima entre dois relançamentos (jogo abrindo não conta como
# "fora da frente" de novo).
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

SCRCPY_SERVER_VERSION = "4.1"
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
TEMPLATE_SCALE_STEPS = (0.85, 0.92, 1.0, 1.08, 1.15)

SHAPE_THRESHOLD = 0.80
COLOR_THRESHOLD = 0.80

# Quantas escalas de TEMPLATE_SCALE_STEPS a passada percorre por template.
# Uma vez que um template bateu numa escala, as outras são desperdício: a
# escala do device não muda sozinha. A passada usa só a escala vencedora e
# volta a abrir o pente quando a categoria fica ESCALA_LIBERA_SECAS passadas
# sem achar nada (zoom da câmera do jogo mudou, ou nunca foi calibrada).
ESCALA_UNICA_POR_TEMPLATE = True
ESCALA_LIBERA_SECAS = 6

# Teto de detecções aprovadas por categoria numa passada: a StateMachine age
# em UMA detecção por frame (_apply_rules), então varrer os 86 templates de
# food depois de já ter 6 pratos na mão não muda nenhuma decisão. Só corta
# DEPOIS de achar — passada parcial nunca vira "tela vazia", que dispararia
# _explore_screen. Sobra margem para _is_ignored_food descartar alguns.
# Categoria fora do dict é varrida inteira.
CATEGORY_MATCH_BUDGET = {
    "food": 6,
    "new_point": 3,
}

# Quantos templates da categoria a passada olha antes de parar, e SÓ se já
# achou alguma coisa. Sem isto, o teto acima não protege o caso em que os
# primeiros templates da fila não batem: a passada percorreria os 86 de food
# mesmo tendo 6 pratos na tela. Nada encontrado = varredura completa, sempre
# (é o caso em que cortar cegaria o bot).
CATEGORY_SCAN_QUOTA = {
    "food": 20,
    "new_point": 12,
}

# Por quanto tempo um template que bateu continua no início da fila da sua
# categoria. Um restaurante tem ~6-10 pratos possíveis entre os 86 templates
# de food; procurar os que apareceram há pouco antes dos outros faz o teto
# acima ser atingido nos primeiros templates, não no quinquagésimo.
QUENTES_TTL = 30.0

CATEGORY_THRESHOLDS = {
    "build": 0.95,
    "new_point": 0.97,
    "food": 0.90,
    "upgrade": 0.95,
    "up_food": 0.90,
    "up_upgrade": 0.90,
    "close": 0.85,
    "plane": 0.98,
    "box": 0.95,
    "delivery": 0.95,
    "helper": 0.95,
}

CATEGORY_COLOR_THRESHOLDS = {
    "delivery": 0.99
}


CATEGORY_ROIS = {}

# Ao detectar via template default, grava o recorte na resolução exata do
# device (mesmo na referência) — as próximas passadas usam esse override
# em vez do default.
LEARN_RESOLUTION_TEMPLATES = False

# De quanto em quanto tempo checar se a pasta de templates mudou (template
# novo, editado ou removido) e recarregar sem reiniciar o bot. 0 desliga.
TEMPLATES_WATCH_INTERVAL = 0


# ==== DETECTOR ====

DETECTOR_DEBUG_MISSES = False
DETECTOR_DEBUG_INTERVAL = 3.0

# Teto da escala do estágio grosso. _coarse_scale_for devolve
# min(COARSE_SCALE, grade), então isto só vincula para template PEQUENO: com
# teto 0.40, todo template de lado menor < 30px caía fora do estágio grosso
# (12/0.40) e ia para matchTemplate em resolução cheia na TELA INTEIRA, o
# caminho mais caro do detector. Template grande não muda (lado 100px -> 0.15).
COARSE_SCALE = 0.70
COARSE_SCALE_MIN = 0.10
COARSE_SCALE_STEP = 0.05
COARSE_MARGIN = 0.18

REFINE_SLACK = 8
MAX_MATCHES_PER_TEMPLATE = 12
NMS_IOU = 0.35


# ==== ACTIONS ====

UPGRADE_ITEM_CLICKS = 5
UPGRADE_FOOD_PRESS = 4.0

DISMISS_POINT = (10, 2200)
GRAY_COIN_POINT = (1070, 250)
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
FOOD_IGNORE_IOU = 0.5
FOOD_MAX_ATTEMPTS = 3


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
