"""Template-matching detector: 2-stage search, ROI/category filters, NMS."""

from pathlib import Path

import json
import math
import queue
import re
import threading
import time

import cv2
import numpy as np

from core import log
from core.metrics import formata_duracao
from core.config import (
    CATEGORY_COLOR_THRESHOLDS,
    CATEGORY_MATCH_BUDGET,
    CATEGORY_ROIS,
    CATEGORY_SCAN_QUOTA,
    CATEGORY_THRESHOLDS,
    COARSE_MARGIN,
    COARSE_SCALE,
    COLOR_THRESHOLD,
    DETECTOR_DEBUG_INTERVAL,
    DETECTOR_DEBUG_MISSES,
    ESCALA_LIBERA_SECAS,
    ESCALA_UNICA_POR_TEMPLATE,
    LEARN_RESOLUTION_TEMPLATES,
    MAX_DETECTION_AGE,
    TEMPLATES_WATCH_INTERVAL,
    MAX_MATCHES_PER_TEMPLATE,
    NMS_IOU,
    QUENTES_TTL,
    REFINE_SLACK,
    SHAPE_THRESHOLD,
    TEMPLATE_SCALE_BASIS,
    TEMPLATE_SCALE_STEPS,
    VISION_PRIORITY_STOP,
    COARSE_SCALE_MIN,
    COARSE_SCALE_STEP,
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    BATTERY_POLL_INTERVAL,
    BATTERY_WARNING_LEVEL,
    CYCLE_STALL_FACTOR,
    SHOW_BATTERY,
    SHOW_CYCLE_TIME,
    SHOW_DETECTION_LABELS,
    SHOW_DETECTION_LAG,
    SHOW_FPS,
)

logger = log.get("detector")


# Relative to this file, not the cwd — independent of running from the repo root.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Templates shared by every device. Sibling folders of this one, named "WIDTHxHEIGHT"
# (in the frame's real resolution, without normalizing orientation), hold extras/
# overrides for that device only — merged with these at runtime.
DEFAULT_TEMPLATES_DIR = TEMPLATES_DIR / "default"

# Manual captures outside the reference resolution (tools/template_selector.py):
# stay here, raw, without entering default or any resolution — they are not
# loaded at runtime.
FALLBACK_SELECTOR_DIR = TEMPLATES_DIR / "default_selector"

# Below this size downscaling destroys the template.
MIN_COARSE_SIDE = 12


class Detector:

    def __init__(
        self,
        threshold=SHAPE_THRESHOLD,
        color_threshold=COLOR_THRESHOLD,
        category_thresholds=None,
        category_rois=None,
        coarse_scale=COARSE_SCALE,
        coarse_margin=COARSE_MARGIN,
        use_defaults=True,
        learn_only_actions=False,
    ):

        # False: search only the resolution overrides (fast, but blind where
        # there is no override yet) — lets the bot run while another process
        # (main_layouts.py) scans the defaults and updates the overrides.
        self.use_defaults = use_defaults
        self.learn_only_actions = learn_only_actions

        # Shape/pattern similarity
        self.threshold = threshold

        # Color similarity
        self.color_threshold = color_threshold

        self.category_thresholds = (
            CATEGORY_THRESHOLDS
            if category_thresholds is None
            else category_thresholds
        )

        self.category_rois = (
            CATEGORY_ROIS
            if category_rois is None
            else category_rois
        )

        self.coarse_scale = coarse_scale
        self.coarse_margin = coarse_margin

        self.templates = []

        # Per-category index: with the state filter a pass looks at 2 categories, not 43.
        self.by_category = {}

        # Templates are rescaled to the frame (1x per resolution), not the
        # other way around: avoids letterboxing the whole frame and per-pass
        # coordinate conversion.
        self._scaled_cache = {}

        # Rescaled defaults, kept apart from the merge above: a learned
        # template invalidates that resolution's merge, and without this cache
        # that would force rescaling every default on the next frame.
        self._defaults_cache = {}

        # ROI in pixels per (category, width, height).
        self._roi_cache = {}

        # (resolution, category) -> {nome_base: (scale that hit, when)}.
        # While a scale is known, the pass only looks at it — the other
        # comb variants exist in the cache but are not searched.
        self._escala_ativa = {}

        # (resolution, category) -> passes run, to rotate the scale
        # probe of templates that never matched.
        self._sonda = {}

        # (resolution, category) -> consecutive full passes with no detection.
        # Once ESCALA_LIBERA_SECAS is reached the category's locked scales
        # are released, and the probe resumes searching for the winner.
        self._secas = {}

        # (resolution, category) -> {nome_base: time of last match}.
        # Orders the category queue: recent hitters go first.
        self._quentes = {}

        # Order already built per (resolution, category) plus the version of the
        # hot set that produced it — reordering 86 templates every frame would cost
        # more than the matches the order saves.
        self._ordem_cache = {}
        self._quentes_versao = 0

        # Categories whose last pass was cut by the detection cap:
        # _report_misses must not call "not detected" what was never
        # looked at, and the HUD must not claim it searched everything.
        self.last_total = 0
        self._parciais = set()

        # Bumped on every committed override or reload: whoever computed a merge
        # outside the lock checks before publishing it.
        self._overrides_versao = 0

        # Extra/override templates per device: (frame_width, frame_height) ->
        # {category: [template]}. Loaded 1x from disk, no rescale (already
        # born at the device resolution).
        self._resolution_templates = {}

        # (key, category, name) -> confidence of the match that produced the
        # saved override version. A new candidate replaces it only with
        # higher confidence — the override improves over time, never regresses.
        self._fidelity = {}

        # Guards _fidelity/_resolution_templates/_scaled_cache against races
        # between the vision thread (detect) and the learning thread.
        self._learn_lock = threading.Lock()

        # Droppable queue: the vision thread only copies the crop (cheap) and
        # returns — the learning thread writes to disk, so as not
        # to block detect() with I/O.
        self._learn_queue = queue.Queue()

        self._learn_thread = threading.Thread(
            target=self._learn_worker,
            name="template-learner",
            daemon=True,
        )

        self._learn_thread.start()

        # (newest mtime, total .png count) at the last check — lets us
        # notice new/edited templates in the folder without restarting the bot.
        self._templates_signature_cache = None
        self._templates_checked_at = 0.0

        # Templates examined on the last pass.
        self.last_searched = 0

        # Resolution of the last analyzed frame: draw() receives the frame already
        # downscaled for the window and would have no way to know the original.
        self._last_frame_size = None

        # Only populated with DETECTOR_DEBUG_MISSES on (cost: 1 bool per
        # match). Distinguishes "did not detect" from "detected but the threshold
        # cut it" — opposite fixes for the same symptom (stalled bot).
        self._best = {}
        self._best_color = {}

        self._debug_category = None
        self._debug_at = 0.0

        # Text report for the panel: with the panel on the console
        # stays at WARNING and the INFO log never shows.
        self._miss_text = ""
        self._miss_at = 0.0

        self.load_templates()

    def _coarse_scale_for(self, smallest_side):
        """Smallest scale with MIN_COARSE_SIDE on a side; rounds up to the grid (safest option)."""

        if smallest_side <= 0:
            return self.coarse_scale

        exata = MIN_COARSE_SIDE / smallest_side

        na_grade = (
            math.ceil(exata / COARSE_SCALE_STEP)
            * COARSE_SCALE_STEP
        )

        # round() keeps 0.15000000000000002 from becoming a duplicate key in the scale cache.
        return round(
            min(
                self.coarse_scale,
                max(COARSE_SCALE_MIN, na_grade),
            ),
            4,
        )

    def load_templates(self):

        with self._learn_lock:
            self._load_templates_locked()

    def _load_templates_locked(self, manter_defaults=False):
        """Body of load_templates() guarded by the lock — the learning
        thread also mutates _resolution_templates/_fidelity.

        manter_defaults: the default folder .png files did not change, only some
        resolution's did. Re-reading the defaults and rescaling every variant
        in that case would cost ~1400 cv2.resize calls inside detect(), on the
        vision thread, for each learned template."""

        self._scaled_cache.clear()
        self._ordem_cache.clear()
        self._resolution_templates.clear()
        self._fidelity.clear()

        self._overrides_versao += 1

        if not manter_defaults:

            self.templates.clear()
            self.by_category.clear()
            self._defaults_cache.clear()
            self._escala_ativa.clear()
            self._sonda.clear()
            self._secas.clear()
            self._quentes.clear()

        if not TEMPLATES_DIR.exists():

            logger.error(
                "Template folder not found: %s",
                TEMPLATES_DIR,
            )

            self._templates_signature_cache = (
                (0.0, 0),
                (0.0, 0),
            )
            self._templates_checked_at = time.monotonic()

            return

        if not manter_defaults:

            self.templates = self._load_category_dirs(
                DEFAULT_TEMPLATES_DIR,
                origin="default",
            )

            for template in self.templates:

                self.by_category.setdefault(
                    template["category"],
                    [],
                ).append(template)

        for resolution_dir in sorted(TEMPLATES_DIR.iterdir()):

            if (
                not resolution_dir.is_dir()
                or resolution_dir in (
                    DEFAULT_TEMPLATES_DIR,
                    FALLBACK_SELECTOR_DIR,
                )
            ):
                continue

            resolucao = self._parse_resolution_dir_name(
                resolution_dir.name
            )

            if resolucao is None:

                logger.warning(
                    "Template folder ignored (name is not "
                    "'WIDTHxHEIGHT'): %s",
                    resolution_dir.name,
                )

                continue

            extras = self._load_category_dirs(
                resolution_dir,
                origin="override",
            )

            por_categoria = {}

            for template in extras:

                por_categoria.setdefault(
                    template["category"],
                    [],
                ).append(template)

            if por_categoria:

                self._resolution_templates[resolucao] = por_categoria

                self._load_fidelity(resolucao, resolution_dir)

        logger.info(
            "Templates: %d default across %d categories %s | "
            "overrides for %d resolution(s) %s",
            len(self.templates),
            len(self.by_category),
            {
                categoria: len(lista)
                for categoria, lista in sorted(
                    self.by_category.items()
                )
            },
            len(self._resolution_templates),
            sorted(
                f"{w}x{h}"
                for w, h in self._resolution_templates
            ),
        )

        # With no templates there is no detection — same symptom as a broken detector (stalled bot).
        if not self.templates:

            logger.error(
                "NO templates loaded from %s — the detector "
                "has nothing to compare against, and the bot "
                "will not act.",
                DEFAULT_TEMPLATES_DIR,
            )

        self._templates_signature_cache = self._templates_signature()
        self._templates_checked_at = time.monotonic()

    @staticmethod
    def _assina_pastas(pastas):
        """(newest mtime, total .png count) — changes with any new,
        edited, or removed template, without comparing file contents one by one."""

        ultima = 0.0
        total = 0

        for pasta in pastas:

            if not pasta.exists():
                continue

            for arquivo in pasta.rglob("*.png"):

                try:
                    mtime = arquivo.stat().st_mtime

                except OSError:
                    continue

                ultima = max(ultima, mtime)
                total += 1

        return (round(ultima, 3), total)

    def _pastas_resolucao(self):
        """"WIDTHxHEIGHT" folders next to default — the per-device overrides."""

        pastas = []

        if not TEMPLATES_DIR.exists():
            return pastas

        for pasta in sorted(TEMPLATES_DIR.iterdir()):

            if not pasta.is_dir():
                continue

            if pasta in (
                DEFAULT_TEMPLATES_DIR,
                FALLBACK_SELECTOR_DIR,
            ):
                continue

            if self._parse_resolution_dir_name(pasta.name) is None:
                continue

            pastas.append(pasta)

        return pastas

    def _templates_signature(self):
        """Two-half signature: (defaults, overrides). Split because
        only the first forces a full rescale — and the second is what the
        learning thread itself changes every time it writes a template."""

        return (
            self._assina_pastas([DEFAULT_TEMPLATES_DIR]),
            self._assina_pastas(self._pastas_resolucao()),
        )

    def _reassina_overrides(self):
        """Brings the overrides half of the signature up to date with disk. Called
        by the learning thread after writing: without this the watcher reads its
        own write as an external change and reloads the templates on every
        learned item.

        Recomputes from disk instead of adding 1 to the total — cv2.imwrite runs
        outside the lock, so the vision thread may already have absorbed that
        file, and a blind increment would leave the count permanently
        ahead (no reload would ever happen again). The rglob runs on the
        learning thread, not the vision one."""

        if self._templates_signature_cache is None:
            return

        defaults, _ = self._templates_signature_cache

        self._templates_signature_cache = (
            defaults,
            self._assina_pastas(self._pastas_resolucao()),
        )

    def _maybe_reload_templates(self):
        """Reloads from disk when the template folder changes — allows
        adding/editing templates (e.g. tools/template_selector.py) without
        restarting the bot."""

        if not TEMPLATES_WATCH_INTERVAL:
            return

        agora = time.monotonic()

        if (
            agora - self._templates_checked_at
            < TEMPLATES_WATCH_INTERVAL
        ):
            return

        self._templates_checked_at = agora

        # Compare and reload under the SAME lock: between computing the signature
        # and acting on it, the learning thread may write and absorb its
        # own write, and the next reload would undo that record.
        # _load_templates_locked (not load_templates) because the lock is not
        # reentrant.
        with self._learn_lock:

            assinatura = self._templates_signature()

            if assinatura == self._templates_signature_cache:
                return

            manter_defaults = (
                self._templates_signature_cache is not None
                and assinatura[0]
                == self._templates_signature_cache[0]
            )

            logger.info(
                "Template folder changed (%s) — reloading.",
                "overrides"
                if manter_defaults
                else "defaults",
            )

            self._load_templates_locked(
                manter_defaults=manter_defaults,
            )

    def _load_category_dirs(self, base_dir, origin):
        """[template] from every category subfolder inside base_dir."""

        templates = []

        if not base_dir.exists():
            return templates

        for category_dir in sorted(base_dir.iterdir()):

            if not category_dir.is_dir():
                continue

            category = category_dir.name

            for image_path in sorted(
                category_dir.glob("*.png")
            ):

                template = self._load_template(
                    category,
                    image_path,
                    origin,
                )

                if template is not None:
                    templates.append(template)

        return templates

    @staticmethod
    def _parse_resolution_dir_name(name):
        """"WIDTHxHEIGHT" -> (width, height), or None if the name does not follow the pattern."""

        largura, _, altura = name.partition("x")

        if not largura.isdigit() or not altura.isdigit():
            return None

        return int(largura), int(altura)

    # "1080x2400_item_001.png" -> ("1080x2400", "item_001.png"). Default
    # accumulates crops from different devices, so the name carries the
    # source resolution; the middle is the identity linking default and override.
    _SOURCE_PREFIX = re.compile(r"^(\d+x\d+)_(.+)$")

    @classmethod
    def _split_source(cls, name):

        casou = cls._SOURCE_PREFIX.match(name)

        if casou is None:
            return None, name

        return casou.group(1), casou.group(2)

    @classmethod
    def _base_name(cls, name):
        """Name without the resolution prefix — same identity in any folder."""

        return cls._split_source(name)[1]

    def template_coverage(self, frame_width=None, frame_height=None):
        """(how many defaults already have an override at this resolution, total defaults).
        The number that tells whether the resolution folder is complete."""

        if frame_width is None or frame_height is None:

            if self._last_frame_size is None:
                return None

            frame_width, frame_height = self._last_frame_size

        with self._learn_lock:

            # (category, base-name) pair: different categories reuse the
            # same item_NNN.png, so the name alone would collapse box/item_001
            # with upgrade/item_001 and the total would come out smaller than reality.
            defaults = {
                (
                    template["category"],
                    self._base_name(template["name"]),
                )
                for template in self.templates
            }

            extras = self._resolution_templates.get(
                (frame_width, frame_height),
                {},
            )

            overrides = {
                (
                    template["category"],
                    self._base_name(template["name"]),
                )
                for lista in extras.values()
                for template in lista
            }

        return len(defaults & overrides), len(defaults)

    @staticmethod
    def _fidelity_path(resolution_dir):
        return resolution_dir / "_fidelity.json"

    def _load_fidelity(self, chave, resolution_dir):
        """Loads the saved confidence of each override at this resolution — without
        an entry here, the next candidate (even a weak one) replaces it outright."""

        caminho = self._fidelity_path(resolution_dir)

        if not caminho.exists():
            return

        try:

            dados = json.loads(caminho.read_text(encoding="utf-8"))

        except (OSError, ValueError) as error:

            logger.warning(
                "Could not read %s: %s",
                caminho,
                error,
            )

            return

        for chave_item, confianca in dados.items():

            categoria, _, nome = chave_item.partition("/")

            if not categoria or not nome:
                continue

            # Base name in the key: it is the identity linking default and
            # override, and the format the _fidelity.json files already use on disk.
            self._fidelity[
                (chave, categoria, self._base_name(nome))
            ] = float(confianca)

    def _save_fidelity(self, chave):
        """Persists the override fidelity for this resolution — without this, a
        restart forgets the confidence and the next candidate replaces it needlessly."""

        resolution_dir = TEMPLATES_DIR / f"{chave[0]}x{chave[1]}"

        dados = {
            f"{categoria}/{nome}": confianca
            for (res, categoria, nome), confianca in self._fidelity.items()
            if res == chave
        }

        try:

            self._fidelity_path(resolution_dir).write_text(
                json.dumps(dados, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        except OSError as error:

            logger.warning(
                "Could not save %s: %s",
                self._fidelity_path(resolution_dir),
                error,
            )

    def _load_template(self, category, image_path, origin):

        raw = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if raw is None:

            logger.error(
                "Error loading: %s",
                image_path,
            )

            return None

        mask = None

        if raw.ndim == 3 and raw.shape[2] == 4:

            alpha = raw[:, :, 3]

            image = raw[:, :, :3]

            # Only worth it if there is real transparency.
            if (alpha < 255).any():

                mask = (alpha > 0).astype(np.uint8) * 255

        elif raw.ndim == 2:

            image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        else:

            image = raw

        return self._build_template(
            category,
            image_path.name,
            image,
            mask,
            origin,
        )

    def _build_template(self, category, name, image, mask, origin):
        """Split from loading: same path for the original PNG and the rescaled version, so the two never diverge."""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # HSV computed once here — not per candidate in
        # _color_similarity (the image never changes).
        template_hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        ).astype(np.int16)

        height, width = gray.shape[:2]

        # This template's own scale (not the global one, pinned by the smallest of all).
        scale = self._coarse_scale_for(min(height, width))

        coarse_gray = None
        coarse_mask = None

        if (
            scale < 1.0
            and min(height, width) * scale >= MIN_COARSE_SIDE
        ):

            coarse_gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

            if mask is not None:

                coarse_mask = cv2.resize(
                    mask,
                    (
                        coarse_gray.shape[1],
                        coarse_gray.shape[0],
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

        return {
            "category": category,
            "name": name,
            "origin": origin,

            # Resolution where this crop was captured, when the name
            # says so — what the overlay shows as "layout".
            "source": self._split_source(name)[0],

            "image": image,
            "gray": gray,
            "hsv": template_hsv,
            "mask": mask,

            # Precomputed bool: `mask > 0` per candidate is
            # one extra pass over a constant mask.
            "mask_valid": None if mask is None else mask > 0,

            "coarse_gray": coarse_gray,
            "coarse_mask": coarse_mask,
            "coarse_scale": scale,

            "width": width,
            "height": height,
        }

    @staticmethod
    def _reference_size(frame_width, frame_height):
        """Reference in the frame's orientation: the game runs landscape (2400x1080) but the reference is written 1080x2400 — without the swap the ratio comes out absurd and nothing detects."""

        ref_width = REFERENCE_WIDTH
        ref_height = REFERENCE_HEIGHT

        if (frame_width > frame_height) != (
            ref_width > ref_height
        ):
            ref_width, ref_height = ref_height, ref_width

        return ref_width, ref_height

    @classmethod
    def _frame_scale(cls, frame_width, frame_height):
        """Template scale factor for this frame, via TEMPLATE_SCALE_BASIS — no universal rule exists because what varies across devices is aspect ratio, not just resolution."""

        if not REFERENCE_WIDTH or not REFERENCE_HEIGHT:
            return 1.0

        if not frame_width or not frame_height:
            return 1.0

        ref_width, ref_height = cls._reference_size(
            frame_width,
            frame_height,
        )

        por_largura = frame_width / ref_width
        por_altura = frame_height / ref_height

        if TEMPLATE_SCALE_BASIS == "width":
            return por_largura

        if TEMPLATE_SCALE_BASIS == "height":
            return por_altura

        if TEMPLATE_SCALE_BASIS == "min":
            return min(por_largura, por_altura)

        if TEMPLATE_SCALE_BASIS == "long_side":

            return (
                max(frame_width, frame_height)
                / max(ref_width, ref_height)
            )

        if TEMPLATE_SCALE_BASIS != "short_side":

            logger.warning(
                "Unknown TEMPLATE_SCALE_BASIS (%r) — "
                "using 'short_side'.",
                TEMPLATE_SCALE_BASIS,
            )

        # Mobile game UI anchors to the NARROW dimension: the
        # other dimension's surplus becomes more scenery, not a
        # bigger interface.
        return (
            min(frame_width, frame_height)
            / min(ref_width, ref_height)
        )

    @classmethod
    def _is_reference(cls, frame_width, frame_height):
        """Frame already at the reference resolution (either orientation) — zero-cost case, nothing to rescale."""

        return (frame_width, frame_height) in (
            (REFERENCE_WIDTH, REFERENCE_HEIGHT),
            (REFERENCE_HEIGHT, REFERENCE_WIDTH),
        )

    def _rescale_template(self, template, scale):
        """Same template at the frame scale, or None if it shrinks too far."""

        # Scale 1.0: return the original — resize would give back the same dimensions,
        # losing sharpness for nothing.
        if abs(scale - 1.0) <= 0.005:
            return template

        width = int(round(template["width"] * scale))
        height = int(round(template["height"] * scale))

        if width < 4 or height < 4:
            return None

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )

        image = cv2.resize(
            template["image"],
            (width, height),
            interpolation=interpolation,
        )

        mask = None

        if template["mask"] is not None:

            mask = cv2.resize(
                template["mask"],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        return self._build_template(
            template["category"],
            template["name"],
            image,
            mask,
            template["origin"],
        )

    def _templates_for(self, frame_width, frame_height):
        """Templates for this frame per category: default (rescaled to the frame, at several scales) + device-folder overrides, cached per resolution."""

        chave = (frame_width, frame_height)

        cacheados = self._scaled_cache.get(chave)

        if cacheados is not None:
            return cacheados

        # Quick snapshot under lock: the learning thread also writes
        # to _resolution_templates/_scaled_cache, and the rest of this function
        # (rescale, merge) is too expensive to run while holding the lock.
        with self._learn_lock:

            versao = self._overrides_versao

            extras_brutos = self._resolution_templates.get(chave)

            extras = (
                None
                if extras_brutos is None
                else {
                    categoria: list(lista)
                    for categoria, lista in extras_brutos.items()
                }
            )

        if (
            self._is_reference(frame_width, frame_height)
            and not extras
        ):
            return self.by_category

        default_categorias = (
            self._defaults_for(frame_width, frame_height)
            if self.use_defaults
            else {}
        )

        # Always a copy, never the _defaults_cache dict itself: the two
        # caches are invalidated at different times (a learned override
        # discards only this one), and sharing the lists let a pop() here
        # mutilate the defaults cache.
        cacheados = {
            categoria: list(lista)
            for categoria, lista in default_categorias.items()
        }

        if extras:

            # Overrides do not replace defaults in the search — they only go
            # first. If one misses on this frame, the same-named default
            # is still in the list and can save the detection. Removing the
            # scale variants here would remove exactly the one
            # winning on this device, and the scale probe already makes the whole
            # comb cost 1 match per pass — nothing left to save.
            for categoria, lista in extras.items():

                cacheados[categoria] = (
                    lista + cacheados.get(categoria, [])
                )

        with self._learn_lock:

            # Override committed while this merge was being computed: its pop
            # found nothing to invalidate, and publishing now would leave
            # the cache without the override, with nothing pending to fix it.
            if versao == self._overrides_versao:
                self._scaled_cache[chave] = cacheados

        return cacheados

    def _defaults_for(self, frame_width, frame_height):
        """Rescaled defaults for this resolution, cached apart from the merge.

        Also at the reference: the game camera zoom changes the on-screen
        icon size (measured: 10% smaller drops 0.98 -> 0.80), so a fixed
        1.0 scale left exactly the reference resolution without the tolerance
        every other one already had."""

        chave = (frame_width, frame_height)

        cacheados = self._defaults_cache.get(chave)

        if cacheados is None:

            cacheados = self._rescale_defaults(
                frame_width,
                frame_height,
            )

            self._defaults_cache[chave] = cacheados

        return cacheados

    def _rescale_defaults(self, frame_width, frame_height):
        """Default templates rescaled to this frame, at several scales around the base (no single bet: 8% scale error already drops confidence below 0.95, and overlap suppression + confidence ordering picks the best of them)."""

        base = self._frame_scale(frame_width, frame_height)

        escalas_base = sorted(
            {
                round(base * passo, 4)
                for passo in TEMPLATE_SCALE_STEPS
                if passo > 0
            }
        ) or [round(base, 4)]

        base_arredondada = round(base, 4)

        total_escalas = len(escalas_base)

        categorias = {}

        for template in self.templates:

            nome_base = self._base_name(template["name"])

            for indice, escala in enumerate(escalas_base):

                variante = self._rescale_template(
                    template,
                    escala,
                )

                if variante is None:
                    continue

                # _rescale_template returns the template ITSELF when the
                # scale is ~1.0: annotating it would contaminate self.templates.
                if variante is template:
                    variante = dict(template)

                variante["nome_base"] = nome_base
                variante["escala"] = escala
                variante["escala_base"] = escala == base_arredondada

                # Position in the comb: while this template's scale is
                # unknown, each pass probes ONE (rotating) position,
                # so the cost is 1 match per template per pass instead
                # of 5, and the right scale shows up within 5 passes.
                variante["indice_escala"] = indice
                variante["total_escalas"] = total_escalas

                categorias.setdefault(
                    variante["category"],
                    [],
                ).append(variante)

        logger.info(
            "Frame %dx%d (reference %dx%d) — %d "
            "default templates across %d scale(s) %s (base %.3f by '%s').",
            frame_width,
            frame_height,
            REFERENCE_WIDTH,
            REFERENCE_HEIGHT,
            sum(len(lista) for lista in categorias.values()),
            len(escalas_base),
            escalas_base,
            base,
            TEMPLATE_SCALE_BASIS,
        )

        return categorias

    def _queue_learn_candidate(
        self,
        frame_color,
        template,
        x,
        y,
        frame_width,
        frame_height,
        confidence,
    ):
        """Called from inside detect() — must stay cheap. Only copies the
        (small) crop and hands it to the learning thread to decide; no
        disk I/O here, so vision never stalls."""

        chave = (frame_width, frame_height)
        categoria = template["category"]

        # Base name: default carries the resolution it was cropped at and
        # override does not, so only the middle identifies both as the same
        # template — without this the same item would become two fidelity
        # entries and learning would rewrite on every pass.
        nome = self._base_name(template["name"])

        # Lock-free check: dict reads are atomic enough
        # for this purpose — worst case is one extra candidate in the
        # queue, which the learning thread drops at final review.
        if confidence <= self._fidelity.get((chave, categoria, nome), 0.0):
            return

        altura = template["height"]
        largura = template["width"]

        crop = frame_color[y:y + altura, x:x + largura]

        if crop.shape[:2] != (altura, largura):
            return

        try:

            self._learn_queue.put_nowait(
                (categoria, nome, chave, crop.copy(), confidence)
            )

        except queue.Full:
            pass

    def learn_action(self, frame, detection):
        """Learns only the detection that originated an executed action."""

        if not LEARN_RESOLUTION_TEMPLATES or frame is None:
            return

        template = detection.get("_template") if detection else None

        if template is None:
            return

        self._queue_learn_candidate(
            frame,
            template,
            detection["x"],
            detection["y"],
            frame.shape[1],
            frame.shape[0],
            detection.get("confidence", 0.0),
        )

    def _learn_worker(self):
        """Runs on its own thread: writes to disk and updates the in-use overrides
        without blocking the vision thread's detect()."""

        while True:

            item = self._learn_queue.get()

            try:

                self._process_learn_candidate(*item)

            except Exception:

                logger.exception(
                    "Error processing learning candidate"
                )

    def _process_learn_candidate(
        self,
        categoria,
        nome,
        chave,
        crop,
        confidence,
    ):

        fidelity_id = (chave, categoria, nome)

        with self._learn_lock:

            if confidence <= self._fidelity.get(fidelity_id, 0.0):
                return

        frame_width, frame_height = chave

        destino_dir = (
            TEMPLATES_DIR
            / f"{frame_width}x{frame_height}"
            / categoria
        )

        destino_dir.mkdir(parents=True, exist_ok=True)

        destino = destino_dir / nome

        if not cv2.imwrite(str(destino), crop):

            logger.warning(
                "Could not write learned template: %s",
                destino,
            )

            return

        novo = self._build_template(
            categoria,
            nome,
            crop,
            None,
            "override",
        )

        with self._learn_lock:

            # Another candidate may have won while we were writing.
            if confidence <= self._fidelity.get(fidelity_id, 0.0):
                return

            anterior = self._fidelity.get(fidelity_id)

            self._fidelity[fidelity_id] = confidence

            lista = self._resolution_templates.setdefault(
                chave, {}
            ).setdefault(categoria, [])

            lista[:] = [
                t
                for t in lista
                if self._base_name(t["name"]) != nome
            ]

            lista.append(novo)

            # This resolution's merged cache went stale: without this, the
            # next pass would keep using the old version.
            self._scaled_cache.pop(chave, None)

            # The ordered queue points at the old cache's templates.
            self._ordem_cache.clear()

            self._overrides_versao += 1

            self._save_fidelity(chave)

            self._reassina_overrides()

        logger.info(
            "Override %dx%d %s/%s: fidelity %s -> %.4f",
            frame_width,
            frame_height,
            categoria,
            nome,
            "-" if anterior is None else f"{anterior:.4f}",
            confidence,
        )

    def _color_similarity(
        self,
        frame,
        template,
        x,
        y,
    ):
        """Compares in HSV: hue is circular (distance 179<->0 is 1), weighted by saturation (desaturated = hue is noise); brightness weighs little (varies with animation/lighting)."""

        template_height = template["height"]
        template_width = template["width"]

        roi = frame[
            y:y + template_height,
            x:x + template_width
        ]

        if roi.shape[:2] != (template_height, template_width):
            return 0.0

        # Only the ROI is converted: the template HSV already came
        # ready-made from _build_template.
        roi_hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV,
        ).astype(np.int16)

        template_hsv = template["hsv"]

        # OpenCV uses H in 0..179 — a full turn is 180, so the real max distance is 90.
        hue_difference = np.abs(
            roi_hsv[:, :, 0] - template_hsv[:, :, 0]
        )

        hue_difference = np.minimum(
            hue_difference,
            180 - hue_difference,
        )

        hue_difference = hue_difference / 90.0

        saturation_difference = np.abs(
            roi_hsv[:, :, 1] - template_hsv[:, :, 1]
        ) / 255.0

        value_difference = np.abs(
            roi_hsv[:, :, 2] - template_hsv[:, :, 2]
        ) / 255.0

        # On a gray pixel hue means nothing — hence the saturation weighting.
        saturation_weight = np.minimum(
            roi_hsv[:, :, 1],
            template_hsv[:, :, 1],
        ) / 255.0

        difference = (
            0.60 * hue_difference * saturation_weight
            + 0.30 * saturation_difference
            + 0.10 * value_difference
        )

        valid = template["mask_valid"]

        if valid is not None:

            if not valid.any():
                return 0.0

            mean_difference = float(
                difference[valid].mean()
            )

        else:

            mean_difference = float(difference.mean())

        return max(0.0, 1.0 - mean_difference)

    def _resolve_roi(
        self,
        category,
        frame_width,
        frame_height,
    ):
        """Converts the category's fractional ROI to pixels: (x1, y1, x2, y2)."""

        chave = (category, frame_width, frame_height)

        cacheada = self._roi_cache.get(chave)

        if cacheada is not None:
            return cacheada

        resolvida = self._compute_roi(
            category,
            frame_width,
            frame_height,
        )

        self._roi_cache[chave] = resolvida

        return resolvida

    def _compute_roi(
        self,
        category,
        frame_width,
        frame_height,
    ):

        roi = self.category_rois.get(category)

        if not roi:

            return (0, 0, frame_width, frame_height)

        x1 = max(0, int(roi[0] * frame_width))
        y1 = max(0, int(roi[1] * frame_height))

        x2 = min(frame_width, int(roi[2] * frame_width))
        y2 = min(frame_height, int(roi[3] * frame_height))

        if x2 <= x1 or y2 <= y1:

            logger.warning(
                "Invalid ROI for %s: %s",
                category,
                roi,
            )

            return (0, 0, frame_width, frame_height)

        return (x1, y1, x2, y2)

    def _na_escala_da_vez(self, template, escalas_ativas, agora):
        """Does this variant join the pass?

        Overrides always join (born at the device resolution, no comb).
        Already-promoted scale: only its variant. Otherwise, only
        not-yet-seen variants — 1 match per pass instead of the comb's 5, with the
        whole comb seen within 5 passes."""

        nome_base = template.get("nome_base")

        if nome_base is None or not ESCALA_UNICA_POR_TEMPLATE:
            return True

        entrada = escalas_ativas.get(nome_base)

        if entrada is None:
            return True

        escala_vencedora, _, quando, vistos = entrada

        if agora - quando > QUENTES_TTL:

            # Stopped hitting too long ago to still be the right
            # scale (zoom changed): resume probing instead of going blind.
            del escalas_ativas[nome_base]
            return True

        total = template.get("total_escalas") or 1
        completo = (1 << total) - 1

        indice = template.get("indice_escala", 0)

        if vistos == completo:
            # Promoted: only the winning scale
            return template.get("escala") == escala_vencedora

        # Provisional: keep the winning scale active, or test unseen scales
        if template.get("escala") == escala_vencedora:
            return True
        return not (vistos >> indice & 1)

    @staticmethod
    def _anota_escala(
        escalas_ativas,
        template,
        confidence,
        agora,
    ):
        """Records the scale that hit. Marks the seen bit and compares
        confidence. Promotes once every scale has been seen."""

        nome_base = template["nome_base"]
        indice = template.get("indice_escala", 0)
        total = template.get("total_escalas") or 1
        completo = (1 << total) - 1

        anterior = escalas_ativas.get(nome_base)

        if anterior is None:

            escalas_ativas[nome_base] = (
                template["escala"],
                confidence,
                agora,
                1 << indice,
            )

            return

        escala, melhor, _, vistos = anterior

        vistos |= 1 << indice

        if vistos < completo and confidence > melhor:

            escala = template["escala"]
            melhor = confidence

        escalas_ativas[nome_base] = (
            escala,
            melhor,
            agora,
            vistos,
        )

    def _em_sonda(self, categoria):
        """Category with no promoted scale — the peak seen in the
        diagnostics is from one of the five scales, not the best possible."""

        if self._last_frame_size is None:
            return False

        ativas = self._escala_ativa.get(
            (self._last_frame_size, categoria),
            {},
        )

        if not ativas:
            return True

        total = 5

        for _, _, _, vistos in ativas.values():
            if vistos == (1 << total) - 1:
                return False

        return True

    def _marca_quente(self, chave_categoria, nome, agora):
        """Recent hitters move to the front of the category queue."""

        quentes = self._quentes.setdefault(chave_categoria, {})

        if nome not in quentes:
            self._quentes_versao += 1

        quentes[nome] = agora

    def _particao_quente(self, chave_categoria, lista, teto, agora):
        """(hot, cold) split of the category. The hot prefix is capped at
        teto-2 names: without that slack, a few stable false positives in the
        first positions would eat the whole cap every frame and the tail would
        never be searched."""

        cacheada = self._ordem_cache.get(chave_categoria)

        # cacheada[1] is lista: a reload/override swaps the category's
        # list, and the vision thread would rewrite the stale partition over
        # the learning thread's clear.
        if (
            cacheada is not None
            and cacheada[0] == self._quentes_versao
            and cacheada[1] is lista
            and agora - cacheada[2] <= QUENTES_TTL
        ):
            return cacheada[3], cacheada[4]

        quentes = self._quentes.get(chave_categoria) or {}

        recentes = sorted(
            (
                nome
                for nome, quando in quentes.items()
                if agora - quando <= QUENTES_TTL
            ),
            key=lambda nome: quentes[nome],
            reverse=True,
        )[: max(1, teto - 2)]

        frescos = set(recentes)

        na_frente = []
        frios = []

        for template in lista:

            nome = (
                template.get("nome_base")
                or template["name"]
            )

            if nome in frescos:
                na_frente.append(template)

            else:
                frios.append(template)

        self._ordem_cache[chave_categoria] = (
            self._quentes_versao,
            lista,
            agora,
            na_frente,
            frios,
        )

        return na_frente, frios

    def _fila_categoria(self, chave_categoria, lista, sonda, agora):
        """Category queue: recent hitters first, cold tail rotated
        each pass. Without the rotation, the quota cut would leave the tail
        end never searched."""

        teto = CATEGORY_MATCH_BUDGET.get(chave_categoria[1], 0)

        if not teto:
            return lista

        na_frente, frios = self._particao_quente(
            chave_categoria,
            lista,
            teto,
            agora,
        )

        if not frios:
            return na_frente

        giro = (sonda * teto) % len(frios)

        return (
            na_frente
            + frios[giro:]
            + frios[:giro]
        )

    def detect(self, frame, categories=None):
        """Approved detections, ordered by confidence (highest first). `categories` filters the search (lets the StateMachine look for 2 templates instead of 43)."""

        self._maybe_reload_templates()

        if frame is None:
            return []

        if DETECTOR_DEBUG_MISSES:

            self._best = {}
            self._best_color = {}

        if frame.ndim == 3:

            frame_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            frame_color = frame

        else:

            frame_gray = frame

            frame_color = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

        frame_height, frame_width = frame_gray.shape[:2]

        chave = (frame_width, frame_height)

        self._last_frame_size = chave

        # Downscaled frames on demand (each template has its scale): in the
        # UPGRADE state, with 4 templates, that is 1 resize instead of 7.
        coarse_frames = {}

        def coarse_for(scale):

            if scale >= 1.0:
                return None

            if scale not in coarse_frames:

                coarse_frames[scale] = cv2.resize(
                    frame_gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            return coarse_frames[scale]

        # This frame's templates per category (original index when already at reference).
        por_categoria = self._templates_for(
            frame_width,
            frame_height,
        )

        # `categories` comes in the state's rule order; the StateMachine acts on
        # the first that matches, so stopping here saves work that would never
        # become an action (measured: 264ms -> ~20ms when an upper rule matches).
        # Stopping is per category, never inside one (avoids losing repeated
        # detections of the same category). Without a filter, nothing is cut short —
        # the diagnostics-overlay and dataset path, which want the whole screen.
        if categories is not None:

            ordem = []
            vistas = set()

            for category in categories:

                if category in vistas:
                    continue

                vistas.add(category)

                if category in por_categoria:
                    ordem.append(category)

            parar_na_prioridade = VISION_PRIORITY_STOP

        else:

            ordem = list(por_categoria)

            parar_na_prioridade = False

        # Without a category filter, the caller wants the whole screen
        # (main_layouts coverage, diagnostics overlay): neither detection cap
        # nor scale probe — both exist to save work that would never become
        # an action, and here there is no action at all.
        exaustiva = categories is None

        # How many templates the pass actually examined: the
        # number that explains an empty HUD ("searched 0").
        self.last_searched = 0
        self.last_total = 0

        self._parciais = set()

        agora = time.monotonic()

        detections = []

        procuradas = []

        for category in ordem:

            procuradas.append(category)

            min_threshold = self.category_thresholds.get(
                category,
                self.threshold,
            )
            min_color_threshold = CATEGORY_COLOR_THRESHOLDS.get(
                category,
                self.color_threshold,
            )

            roi = self._resolve_roi(
                category,
                frame_width,
                frame_height,
            )

            # Where _match records this category's best
            # match, for diagnostics.
            self._debug_category = category

            chave_categoria = (chave, category)

            escalas_ativas = self._escala_ativa.setdefault(
                chave_categoria,
                {},
            )

            sonda = self._sonda.get(chave_categoria, 0)

            # The StateMachine acts on ONE detection per frame: sweeping all 86
            # food templates after already holding the dish cap changes
            # no decision. Cuts only AFTER finding, so a partial
            # pass is never mistaken for an empty screen (which would trigger the
            # exploration swipe). 0 = fully swept category.
            teto = (
                0
                if exaustiva
                else CATEGORY_MATCH_BUDGET.get(category, 0)
            )

            cota = (
                0
                if exaustiva
                else CATEGORY_SCAN_QUOTA.get(category, 0)
            )

            fila = (
                por_categoria[category]
                if exaustiva
                else self._fila_categoria(
                    chave_categoria,
                    por_categoria[category],
                    sonda,
                    agora,
                )
            )

            self.last_total += len(fila)

            # Distinct items, not candidates: MAX_MATCHES_PER_TEMPLATE
            # allows 12 candidates from a single template, and the cap exists
            # so the StateMachine has real alternatives when it
            # discards a detection (_is_ignored_food).
            achados = set()

            olhados = 0

            for template in fila:

                if (
                    template["width"] > frame_width
                    or template["height"] > frame_height
                ):
                    continue

                if not exaustiva and not self._na_escala_da_vez(
                    template,
                    escalas_ativas,
                    agora,
                ):
                    continue

                self.last_searched += 1

                olhados += 1

                candidates = self._search(
                    frame_gray,
                    coarse_for(template["coarse_scale"]),
                    template,
                    min_threshold,
                    roi,
                )

                for x, y, confidence in candidates:

                    color_similarity = self._color_similarity(
                        frame_color,
                        template,
                        x,
                        y,
                    )

                    if DETECTOR_DEBUG_MISSES:

                        self._best_color[category] = max(
                            self._best_color.get(
                                category,
                                0.0,
                            ),
                            color_similarity,
                        )

                    if color_similarity < min_color_threshold:
                        continue

                    nome_base = template.get("nome_base")

                    achados.add(
                        nome_base or template["name"]
                    )

                    if nome_base is not None:

                        self._anota_escala(
                            escalas_ativas,
                            template,
                            confidence,
                            agora,
                        )

                    self._marca_quente(
                        chave_categoria,
                        nome_base or template["name"],
                        agora,
                    )

                    # Template path: /origin/category/name
                    origin = template.get("origin", "default")
                    template_path = f"/{origin}/{category}/{template['name']}"

                    detections.append(
                        {
                            "category": category,
                            "name": template["name"],
                            "template_path": template_path,
                            "origin": template["origin"],
                            "source": template["source"],
                            "_template": template,

                            "confidence": float(confidence),
                            "color_similarity": float(
                                color_similarity
                            ),
                            "min_threshold": float(
                                min_threshold
                            ),

                            "x": int(x),
                            "y": int(y),

                            "width": template["width"],
                            "height": template["height"],
                        }
                    )

                    if (
                        LEARN_RESOLUTION_TEMPLATES
                        and not self.learn_only_actions
                    ):

                        self._queue_learn_candidate(
                            frame_color,
                            template,
                            x,
                            y,
                            frame_width,
                            frame_height,
                            confidence,
                        )

                # Both cuts apply only AFTER finding: a partial pass
                # with detections must not be mistaken for an empty screen, which in
                # NORMAL triggers the exploration swipe. Nothing found =
                # fully swept category.
                if achados and (
                    (teto and len(achados) >= teto)
                    or (cota and olhados >= cota)
                ):

                    self._parciais.add(category)

                    break

            achou = len(achados)

            if achou:

                self._secas.pop(chave_categoria, None)

            elif category not in self._parciais:

                secas = self._secas.get(chave_categoria, 0) + 1

                # Category gone for several passes in a row: may be the
                # game camera zoom, which rescales everything. Removes
                # only the promoted scales (vistos == completo), keeping
                # the provisional ones and the accumulated confidence.
                if secas >= ESCALA_LIBERA_SECAS:

                    for nome in list(escalas_ativas.keys()):

                        _, _, _, vistos = escalas_ativas[nome]

                        total = self.templates[0].get("total_escalas", 5) if self.templates else 5

                        if vistos == (1 << total) - 1:
                            del escalas_ativas[nome]

                    secas = 0

                self._secas[chave_categoria] = secas

            self._sonda[chave_categoria] = sonda + 1

            if achou and parar_na_prioridade:
                break

        # Several templates of the same category hit the same icon — without
        # suppression, one food item becomes 5 detections.
        if DETECTOR_DEBUG_MISSES:

            # Only categories the pass actually examined — with priority
            # stopping, listing the lower ones as "not detected" would be wrong.
            self._report_misses(
                set(procuradas) - self._parciais,
                detections,
            )

        detections = self._suppress(detections)

        # Highest confidence first: the StateMachine now acts
        # on the BEST match, not the first one
        # in filename-alphabetical order.
        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return detections

    def _search(
        self,
        frame_gray,
        coarse_frame,
        template,
        min_threshold,
        roi,
    ):
        """Returns [(x, y, confidence)] at full resolution."""

        coarse_template = template["coarse_gray"]

        # No coarse stage: direct search.
        if coarse_frame is None or coarse_template is None:

            return self._match(
                frame_gray,
                template["gray"],
                template["mask"],
                min_threshold,
                roi,
                MAX_MATCHES_PER_TEMPLATE,
            )

        # Stage 1 (coarse): more lenient threshold — downscaling degrades
        # confidence, and discarding here is irreversible.
        scale = template["coarse_scale"]

        coarse_roi = (
            int(roi[0] * scale),
            int(roi[1] * scale),
            min(
                coarse_frame.shape[1],
                int(round(roi[2] * scale)),
            ),
            min(
                coarse_frame.shape[0],
                int(round(roi[3] * scale)),
            ),
        )

        coarse_hits = self._match(
            coarse_frame,
            coarse_template,
            template["coarse_mask"],
            max(0.0, min_threshold - self.coarse_margin),
            coarse_roi,
            MAX_MATCHES_PER_TEMPLATE,
        )

        if not coarse_hits:
            return []

        # Stage 2 (fine): re-confirms each candidate at full resolution
        # in a window — the returned confidence is already comparable to the stageless version.
        results = []

        template_height = template["height"]
        template_width = template["width"]

        frame_height, frame_width = frame_gray.shape[:2]

        for coarse_x, coarse_y, _ in coarse_hits:

            estimated_x = int(coarse_x / scale)
            estimated_y = int(coarse_y / scale)

            window = (
                max(0, estimated_x - REFINE_SLACK),
                max(0, estimated_y - REFINE_SLACK),
                min(
                    frame_width,
                    estimated_x + template_width
                    + REFINE_SLACK,
                ),
                min(
                    frame_height,
                    estimated_y + template_height
                    + REFINE_SLACK,
                ),
            )

            refined = self._match(
                frame_gray,
                template["gray"],
                template["mask"],
                min_threshold,
                window,

                # In the fine window only the best point matters.
                1,
            )

            results.extend(refined)

        return results

    def _match(
        self,
        image,
        template_image,
        mask,
        min_threshold,
        roi,
        max_matches,
    ):
        """matchTemplate in a region; returns local maxima above the threshold in absolute image coordinates."""

        x1, y1, x2, y2 = roi

        template_height, template_width = (
            template_image.shape[:2]
        )

        if (
            x2 - x1 < template_width
            or y2 - y1 < template_height
        ):
            return []

        region = image[y1:y2, x1:x2]

        # Masks are only supported in TM_CCORR_NORMED/TM_SQDIFF; without a mask
        # we keep TM_CCOEFF_NORMED so threshold scales stay unchanged.
        if mask is not None:

            result = cv2.matchTemplate(
                region,
                template_image,
                cv2.TM_CCORR_NORMED,
                mask=mask,
            )

        else:

            result = cv2.matchTemplate(
                region,
                template_image,
                cv2.TM_CCOEFF_NORMED,
            )

        # copy=False: without this each match allocates another copy of the
        # response map, which on the no-coarse-stage path is screen-sized.
        result = np.nan_to_num(
            result,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Raw maximum before the cutoff — the information the threshold
        # hides and diagnostics needs.
        if DETECTOR_DEBUG_MISSES and result.size:

            if self._debug_category is not None:

                self._best[self._debug_category] = max(
                    self._best.get(self._debug_category, 0.0),
                    float(result.max()),
                )

        if max_matches == 1:

            _, confidence, _, location = cv2.minMaxLoc(result)

            if confidence < min_threshold:
                return []

            return [
                (
                    x1 + location[0],
                    y1 + location[1],
                    confidence,
                )
            ]

        ys, xs = np.where(result >= min_threshold)

        if len(xs) == 0:
            return []

        confidences = result[ys, xs]

        # Best first, so suppression keeps the good one.
        order = np.argsort(-confidences)

        hits = []

        for index in order:

            x = int(xs[index]) + x1
            y = int(ys[index]) + y1

            confidence = float(confidences[index])

            # Drops neighbors of the same peak.
            if any(
                abs(x - other_x) < template_width * 0.5
                and abs(y - other_y) < template_height * 0.5
                for other_x, other_y, _ in hits
            ):
                continue

            hits.append((x, y, confidence))

            if len(hits) >= max_matches:
                break

        return hits

    def _report_misses(self, wanted, detections):
        """Per missing category, the best match seen: F:0.931/0.98 = threshold cut it (lower the threshold); F:0.412/0.98 = template does not look like the screen (threshold will not help)."""

        agora = time.monotonic()

        if agora - self._debug_at < DETECTOR_DEBUG_INTERVAL:
            return

        self._debug_at = agora

        encontradas = {d["category"] for d in detections}

        procuradas = (
            wanted
            if wanted is not None
            else {t["category"] for t in self.templates}
        )

        faltando = sorted(procuradas - encontradas)

        if not faltando:

            # Clears the previous report: stale problem
            # text is worse than no text.
            self._miss_text = ""
            self._miss_at = agora

            return

        partes = []

        for categoria in faltando:

            corte = self.category_thresholds.get(
                categoria,
                self.threshold,
            )

            forma = self._best.get(categoria)
            cor = self._best_color.get(categoria)

            texto = f"{categoria} F:"
            texto += "-" if forma is None else f"{forma:.3f}"
            texto += f"/{corte:.2f}"

            # Color only when shape passed: otherwise the
            # color comparison never ran, and a "C:-"
            # would look like a failure.
            if cor is not None:
                limite_cor = CATEGORY_COLOR_THRESHOLDS.get(
                    categoria,
                    self.color_threshold,
                )
                texto += (
                    f" C:{cor:.3f}"
                    f"/{limite_cor:.2f}"
                )

            # Without a promoted scale, the peak above is from ONE of the five scales:
            # reading "0.412" as "the template does not look like the screen" would be
            # wrong — it may just be a scale not probed yet.
            if self._em_sonda(categoria):
                texto += " ~probing"

            partes.append(texto)

        self._miss_text = "  ".join(partes)
        self._miss_at = agora

        logger.info(
            "not detected (best match / cutoff): %s",
            self._miss_text,
        )

    def miss_report(self):
        """Latest report, or "" when missing or too old (minutes-old text would look current on the panel)."""

        if not DETECTOR_DEBUG_MISSES or not self._miss_text:
            return ""

        if (
            time.monotonic() - self._miss_at
            > DETECTOR_DEBUG_INTERVAL * 3
        ):
            return ""

        return self._miss_text

    def _suppress(self, detections):
        """Keeps, per category, the best detection of each overlapping-box group."""

        if len(detections) < 2:
            return detections

        by_category = {}

        for detection in detections:

            by_category.setdefault(
                detection["category"],
                [],
            ).append(detection)

        kept = []

        for group in by_category.values():

            group.sort(
                key=lambda item: item["confidence"],
                reverse=True,
            )

            for candidate in group:

                overlapped = any(
                    self._iou(candidate, chosen) > NMS_IOU
                    for chosen in kept
                    if chosen["category"]
                    == candidate["category"]
                )

                if not overlapped:

                    kept.append(candidate)

        return kept

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

        if union <= 0:
            return 0.0

        return intersection / union

    # Per-category color: shows what is what without reading the text.
    COLORS = {
        "food": (0, 220, 255),
        "upgrade": (0, 255, 0),
        "up_upgrade": (0, 200, 120),
        "up_food": (120, 255, 0),
        "new_point": (255, 120, 0),
        "box": (255, 0, 220),
        "close": (0, 0, 255),
        "plane": (255, 255, 0),
        "build": (100, 0, 220),
        "fly": (0, 150, 255),
        "delivery": (255, 0, 0),
        "accept_delivery": (0, 100, 255),
        "helper": (180, 0, 255),
        "invite_helper": (0, 180, 80),
        "gray_max": (128, 128, 128),
        "gray_coin": (90, 160, 200),
        "open_store": (255, 180, 0),
        "renovate_coin": (0, 255, 255),
    }

    DEFAULT_COLOR = (160, 0, 200)

    # HUD colors
    HUD_OK = (0, 255, 0)
    HUD_BAD = (0, 0, 255)
    HUD_NEUTRAL = (255, 255, 255)

    def _hud_text(self, output, text, y, color, scale=0.8):
        """Text with a dark outline, legible over any game background."""

        for thickness, text_color in (
            (5, (0, 0, 0)),
            (2, color),
        ):

            cv2.putText(
                output,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    def _layout_label(self, detection):
        """First label line: which layout the matching template came from."""

        origem = detection.get("origin")
        source = detection.get("source")

        # Detections not from a template (IA/bot_ai.py uses the model and does not
        # fill origin) have no layout to show.
        if origem not in ("default", "override"):
            return ""

        if origem == "override":

            # Resolution-specific: shows the resolution instead of the type.
            # The template is always from the frame's resolution; the name only confirms it.
            if source is None and self._last_frame_size is not None:
                source = "x".join(str(n) for n in self._last_frame_size)

            return source or "resolution"

        # In default the resolution is provenance: it says on which screen the crop
        # was taken, not which one it matched.
        return f"default {source}" if source else "default"

    def draw(self, frame, detections, stats=None, scale=1.0):
        """stats: capture_fps, detect_fps, detect_ms, lag, battery, cycle — missing keys are hidden. `scale` maps detections (in full-frame coords) back to the downscaled frame arriving here, otherwise boxes show up misplaced."""

        output = frame

        for detection in detections:

            x = int(detection["x"] * scale)
            y = int(detection["y"] * scale)

            width = int(detection["width"] * scale)
            height = int(detection["height"] * scale)

            category = detection["category"]

            color = self.COLORS.get(
                category,
                self.DEFAULT_COLOR,
            )

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                color,
                3,
            )

            if not SHOW_DETECTION_LABELS:
                continue

            # In food the item name matters more than the category (70+
            # templates in a single category); elsewhere the category alone says it all.
            if category == "food":
                rotulo = self._base_name(detection["name"]).removesuffix(".png")
            else:
                rotulo = category

            linhas = [
                linha
                for linha in (
                    self._layout_label(detection),
                    (
                        f"{rotulo} "
                        f"F:{detection['confidence']:.2f} "
                        f"C:{detection.get('color_similarity', 0.0):.2f}"
                    ),
                )
                if linha
            ]

            # Stacked above the box; the floor keeps both on screen without
            # one writing over the other.
            primeira_y = max(y - 32, 16)

            for indice, linha in enumerate(linhas):

                # Dark outline + colored text.
                for thickness, text_color in (
                    (5, (0, 0, 0)),
                    (1, color),
                ):

                    cv2.putText(
                        output,
                        linha,
                        (x, primeira_y + indice * 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        thickness,
                        cv2.LINE_AA,
                    )

        if not stats:
            return output

        y = 50

        # Two distinct numbers on purpose: capture can run at 60 fps
        # while the detector does 3 passes per second — the detector
        # drives the bot's reaction time.
        if SHOW_FPS:

            capture_fps = stats.get("capture_fps")

            if capture_fps is not None:

                self._hud_text(
                    output,
                    f"capture  {capture_fps:5.1f} fps",
                    y,
                    self.HUD_NEUTRAL,
                )

                y += 40

            detect_fps = stats.get("detect_fps")

            if detect_fps is not None:

                label = f"detector {detect_fps:5.1f} fps"

                detect_ms = stats.get("detect_ms")

                if detect_ms:

                    label += f"  {detect_ms:.0f} ms"

                # Below 1/MAX_DETECTION_AGE detections are born stale
                # and the StateMachine stops clicking — same limit as the age check.
                floor = (
                    1.0 / MAX_DETECTION_AGE
                    if MAX_DETECTION_AGE
                    else 0.0
                )

                self._hud_text(
                    output,
                    label,
                    y,
                    self.HUD_OK
                    if detect_fps >= floor
                    else self.HUD_BAD,
                )

                y += 40

        # Without this, a late detection looks like a wrong detection.
        lag = stats.get("lag")

        if SHOW_DETECTION_LAG and lag is not None:

            limit = (
                MAX_DETECTION_AGE
                if MAX_DETECTION_AGE
                else 1.0
            )

            self._hud_text(
                output,
                f"lag      {lag * 1000:5.0f} ms",
                y,
                self.HUD_OK
                if lag <= limit * 0.5
                else self.HUD_BAD,
            )

            y += 40

        # A session killed by a drained battery leaves no trace in the log — the bot just stops acting.
        battery = stats.get("battery")

        if SHOW_BATTERY and battery:

            level, charging, age = battery

            if level is not None:

                label = f"battery  {level:5d} %"

                if charging:
                    label += "  charging"

                # A stale reading is worse than a missing one (it looks current); past
                # twice the interval means adb is stuck.
                elif age > BATTERY_POLL_INTERVAL * 2:
                    label += f"  ({age:.0f}s stale)"

                self._hud_text(
                    output,
                    label,
                    y,
                    self.HUD_OK
                    if charging or level > BATTERY_WARNING_LEVEL
                    else self.HUD_BAD,
                )

                y += 40

        # Elapsed time between renovations is the only HUD number measuring
        # progress (FPS only says vision is healthy).

        # Without "state", "did not detect" and "did not search" look identical in the
        # overlay; with state filtering, knowing which state and how many
        # templates explains an empty overlay.
        estado = stats.get("state")

        if estado:

            label = f"state    {estado}"

            procurados = stats.get("searched")

            if procurados is not None:

                # "12/430" instead of "12": with scale probing and per-category
                # caps the pass deliberately examines a fraction of the list,
                # and bare "12" would look like a broken detector.
                na_lista = stats.get("searchable")

                label += (
                    f"  ({procurados} templates)"
                    if not na_lista
                    else f"  ({procurados}/{na_lista} templates)"
                )

            self._hud_text(
                output,
                label,
                y,
                self.HUD_NEUTRAL,
            )

            y += 40

        # How many defaults already have an equivalent in this resolution's folder:
        # what tells whether device coverage is complete (and, when it
        # is not, that search still relies on the rescaled default).
        coverage = stats.get("coverage")

        if coverage:

            cobertos, total = coverage

            self._hud_text(
                output,
                f"template {cobertos:4d}/{total} of resolution",
                y,
                self.HUD_OK
                if total and cobertos >= total
                else self.HUD_NEUTRAL,
            )

            y += 40

        # Zero detections in red — except during settle, where the worker
        # skips frames on purpose and that is not a defect.
        quantas = stats.get("detections")

        aguardando = stats.get("waiting_settle")

        if quantas is not None:

            label = f"detections {quantas:4d}"

            if aguardando:
                label += "  (waiting for screen to settle)"

            self._hud_text(
                output,
                label,
                y,
                self.HUD_NEUTRAL
                if aguardando
                else (
                    self.HUD_OK if quantas else self.HUD_BAD
                ),
            )

            y += 40

        # A dead detector thread has the same symptom as everything else (stalled
        # bot) — here it speaks up.
        erro = stats.get("vision_error")

        if erro:

            self._hud_text(
                output,
                f"VISION: {erro[:48]}",
                y,
                self.HUD_BAD,
                scale=0.6,
            )

            y += 40

        cycle = stats.get("cycle")

        if SHOW_CYCLE_TIME and cycle:

            corrido, ultimo, quantos = cycle

            label = f"renovation {quantos:5d}  {formata_duracao(corrido)}"

            if ultimo is not None:
                label += f"  (last {formata_duracao(ultimo)})"

            # Red only when there is something to compare against: without
            # a previous cycle, "slow" means nothing.
            travado = (
                ultimo is not None
                and corrido > ultimo * CYCLE_STALL_FACTOR
            )

            self._hud_text(
                output,
                label,
                y,
                self.HUD_BAD if travado else self.HUD_NEUTRAL,
            )

        return output
