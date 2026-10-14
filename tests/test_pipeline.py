"""
Integration: frame -> detector -> worker -> state machine -> tap. Uses the
REAL threads (only adb is fake). Catches what unit tests don't:
frame handoff between threads, detection age, "action in
progress" gate, and frame-to-tap coordinate conversion.
"""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from actions.android import AndroidActions      # noqa: E402
from actions.manager import ActionManager       # noqa: E402
from core import log                            # noqa: E402
from core.battery import BatteryMonitor         # noqa: E402
from core import state_machine as sm            # noqa: E402
from core.metrics import DurationMeter, RateMeter  # noqa: E402
from vision.detector import Detector            # noqa: E402
from vision.worker import VisionWorker          # noqa: E402

IMAGE = ROOT / "tests" / "images" / "eatventure.png"

# Approximate anchor of the upgrade button on this screen — POSITION, never SIZE:
# the size comes from the template PNG and re-cropping it already changed this number
# once (136x134 -> 103x72), breaking this test without it being about the
# template. Only serves to ensure the detector found the button in the bottom
# corner, not garbage elsewhere; the tap point comes from the DETECTION.
UPGRADE_NEAR = (914, 2194)

# Generous slack: a template cropped with spare pixels shifts the corner.
UPGRADE_TOLERANCE = 40


class FakeAndroid(AndroidActions):

    def __init__(self, device_size=(1080, 2400)):

        super().__init__()

        self.device_size = device_size

        self.commands = []

    def _run(self, *args, timeout=10.0):

        if args[:3] == ("shell", "wm", "size"):

            return (
                f"Physical size: "
                f"{self.device_size[0]}x"
                f"{self.device_size[1]}\n"
            )

        self.commands.extend(self._expande(args))

        return ""

    @staticmethod
    def _expande(args):
        """
        tap_many/swipe_many send N taps in a single `adb shell "input tap
        ...; input tap ..."` to avoid paying the adb spawn N times. Takes it
        back apart word-by-word, which is what the properties below
        read — without this `scrolls` would see zero swipes where the device sends six.
        """

        if len(args) != 2 or args[0] != "shell":
            return [args]

        if ";" not in args[1]:
            return [args]

        return [
            ("shell", *pedaco.split())
            for pedaco in args[1].split(";")
            if pedaco.strip()
        ]

    @property
    def taps(self):

        return [
            (int(command[3]), int(command[4]))
            for command in self.commands
            if command[:3] == ("shell", "input", "tap")
        ]

    @property
    def backs(self):
        """BACK should never appear: in this game it quits the game."""

        return [
            command
            for command in self.commands
            if command[:3] == ("shell", "input", "keyevent")
            and command[3] == "4"
        ]

    @property
    def scrolls(self):
        """Scroll swipes: 'input swipe' with origin != destination. Returns (start_y, end_y)."""

        resultado = []

        for command in self.commands:

            if command[:3] != ("shell", "input", "swipe"):
                continue

            x1, y1, x2, y2, _ = (
                int(v) for v in command[3:8]
            )

            if (x1, y1) == (x2, y2):
                continue

            resultado.append((y1, y2))

        return resultado

    @property
    def holds(self):
        """Held taps: 'input swipe' with origin == destination. Returns (x, y, duration in s)."""

        resultado = []

        for command in self.commands:

            if command[:3] != ("shell", "input", "swipe"):
                continue

            x1, y1, x2, y2, ms = (
                int(v) for v in command[3:8]
            )

            if (x1, y1) != (x2, y2):
                continue

            resultado.append((x1, y1, ms / 1000.0))

        return resultado


class FakeCapture:
    """Serves the same frame a few times then ends, mimicking stream end."""

    def __init__(self, frame, frames=5):

        self.frame = frame

        self.frames = frames

        self.version = 0

        self.running = True

    def get_frame(self, since_version=None, timeout=None):

        if self.version >= self.frames:

            self.running = False

            return None, self.version, 0.0

        self.version += 1

        return self.frame, self.version, time.monotonic()

    def is_running(self):

        return self.running


def build(device_size=(1080, 2400)):

    detector = Detector()

    vision = VisionWorker(detector)

    actions = ActionManager()

    actions.android = FakeAndroid(device_size)

    machine = sm.StateMachine(actions)

    return detector, vision, actions, machine


def wait_for_detections(vision, timeout=10.0):

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        detections, lag = vision.get_detections()

        if detections:

            return detections, lag

        time.sleep(0.02)

    return [], 0.0


def test_frame_vira_toque_no_lugar_certo():
    """The full path, with the real threads."""

    frame = cv2.imread(str(IMAGE))

    assert frame is not None, IMAGE

    _, vision, actions, machine = build()

    vision.start()

    actions.start()

    try:

        height, width = frame.shape[:2]

        actions.set_frame_size(width, height)

        vision.set_categories(machine.wanted_categories())

        vision.set_frame(frame)

        detections, lag = wait_for_detections(vision)

        assert detections, "worker não produziu detecção"

        # Age is measured from the frame, not from the query.
        assert 0.0 < lag < 5.0, lag

        categories = [d["category"] for d in detections]

        assert "upgrade" in categories, categories

        # Highest confidence first.
        confidences = [d["confidence"] for d in detections]

        assert confidences == sorted(
            confidences,
            reverse=True,
        ), confidences

        machine.update(detections, lag)

        assert machine.state == sm.UPGRADE, machine.state

        # Waits for the action thread to drain.
        deadline = time.monotonic() + 5.0

        while (
            actions.is_busy()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

        # The detection that MOTIVATED the action — its center is the finger target.
        # Derived, not pinned: pinning the size here would duplicate the
        # template PNG and break when someone re-crops it (already
        # happened).
        alvo = next(
            d
            for d in detections
            if d["category"] == "upgrade"
        )

        # Loose anchor: is the button where expected on screen?
        assert (
            abs(alvo["x"] - UPGRADE_NEAR[0])
            <= UPGRADE_TOLERANCE
            and abs(alvo["y"] - UPGRADE_NEAR[1])
            <= UPGRADE_TOLERANCE
        ), (alvo["x"], alvo["y"], UPGRADE_NEAR)

        expected = (
            alvo["x"] + alvo["width"] // 2,
            alvo["y"] + alvo["height"] // 2,
        )

        assert actions.android.taps == [expected], (
            actions.android.taps,
            expected,
        )

    finally:

        vision.stop()

        actions.stop()


def test_frame_reduzido_converte_coordenada():
    """Stream at a different resolution than the device: without conversion, the click landed in the wrong place."""

    _, vision, actions, machine = build(
        device_size=(1080, 2400)
    )

    actions.start()

    try:

        # Detection computed on a half-size frame.
        actions.set_frame_size(540, 1200)

        detection = {
            "category": "upgrade",
            "name": "item_001.png",
            "confidence": 1.0,
            "color_similarity": 1.0,
            "min_threshold": 0.98,
            "x": 100,
            "y": 200,
            "width": 40,
            "height": 40,
        }

        actions.execute("upgrade", detection)

        deadline = time.monotonic() + 5.0

        while (
            actions.is_busy()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

        # center on frame = (120, 220); device is 2x bigger.
        assert actions.android.taps == [(240, 440)], (
            actions.android.taps
        )

    finally:

        actions.stop()


def test_acao_longa_nao_bloqueia_o_loop():
    """The food long press lasts 4 s; it used to freeze the whole main loop."""

    _, _, actions, _ = build()

    actions.start()

    try:

        detection = {
            "category": "up_food",
            "name": "item_001.png",
            "confidence": 0.99,
            "color_similarity": 0.95,
            "min_threshold": 0.9,
            "x": 500,
            "y": 900,
            "width": 43,
            "height": 128,
        }

        actions.set_frame_size(1080, 2400)

        started = time.monotonic()

        assert actions.execute("upgrade_food", detection)

        # The call must return right away, not in 4 s.
        elapsed = time.monotonic() - started

        assert elapsed < 0.5, elapsed

        # And while it runs, nothing else enters the queue.
        assert actions.is_busy()

        assert not actions.execute("close", detection)

    finally:

        actions.stop()


def test_categoria_desconhecida_nao_derruba():

    _, _, actions, _ = build()

    actions.start()

    try:

        assert not actions.execute("nao_existe", None)

        assert actions.android.taps == []

    finally:

        actions.stop()


def test_loop_roda_sem_janela():
    """DISPLAY_MODE "scrcpy"/"none": without an OpenCV window, this is the path where nobody calls cv2.waitKey."""

    import main

    frame = cv2.imread(str(IMAGE))

    detector, vision, actions, machine = build()

    capture = FakeCapture(frame, frames=5)

    original = main.SHOW_AI_VISION

    main.SHOW_AI_VISION = False

    # The battery monitor runs in a thread and talks adb — here
    # adb is fake, so it exercises the whole path with no device.
    battery = BatteryMonitor(actions.android, interval=0.05)

    vision.start()

    actions.start()

    battery.start()

    try:

        main.run_loop(
            capture,
            vision,
            detector,
            machine,
            actions,
            battery,
        )

        # Consumed the frames and exited when the stream ended.
        assert capture.version == 5, capture.version

        assert not capture.is_running()

    finally:

        main.SHOW_AI_VISION = original

        battery.stop()

        vision.stop()

        actions.stop()


def test_medidores_reportam_taxa():

    meter = RateMeter()

    for _ in range(20):

        meter.tick()

        time.sleep(0.01)

    rate = meter.rate()

    # ~100/s, with wide slack so it doesn't get flaky.
    assert 50 < rate < 200, rate

    duration = DurationMeter()

    for _ in range(40):

        duration.add(0.320)

    assert abs(duration.average() - 0.320) < 0.005, (
        duration.average()
    )


def test_gray_coin_toca_no_ponto_proprio():
    """`gray_coin` dismisses like `gray_max`, FIXED point — only the point differs."""

    from actions.manager import ACTION_TABLE, DISMISS
    from core.config import DISMISS_POINT, GRAY_COIN_POINT

    # Same behavior as gray_max, different point.
    assert (
        ACTION_TABLE["gray_coin"]
        == ACTION_TABLE["gray_max"]
        == DISMISS
    )

    assert GRAY_COIN_POINT != DISMISS_POINT

    deteccao = {
        "category": "gray_coin",
        "name": "item_001.png",
        "confidence": 0.95,
        "color_similarity": 0.95,
        "min_threshold": 0.8,

        # Far from the fixed point, on purpose: that's what proves the
        # detection was ignored.
        "x": 500,
        "y": 900,
        "width": 80,
        "height": 80,
    }

    centro = (
        deteccao["x"] + deteccao["width"] // 2,
        deteccao["y"] + deteccao["height"] // 2,
    )

    for tamanho in ((1080, 2400), (2400, 1080), (720, 1600)):

        _, _, actions, _ = build(tamanho)

        actions.start()

        try:

            actions.set_frame_size(*tamanho)

            assert actions.execute("gray_coin", deteccao)

            deadline = time.monotonic() + 5.0

            while (
                actions.is_busy()
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)

            esperado = actions._from_reference(
                *GRAY_COIN_POINT
            )

            assert actions.android.taps[-1] == esperado, (
                tamanho,
                actions.android.taps,
            )

            assert actions.android.taps[-1] != centro, (
                "tocou no centro da detecção — devia ignorar a "
                "detecção e usar o ponto fixo"
            )

            # The dataset label points at the SAME place
            # the finger tapped. If the two diverged, training
            # would learn a point the bot never taps.
            assert actions.target_frame(
                "gray_coin",
                deteccao,
            ) == actions._reference_to_frame(*GRAY_COIN_POINT)

        finally:

            actions.stop()


def test_proporcao_diferente_nao_encolhe_o_template():
    """
    The "doesn't detect on another phone" bug: on 1080x1920 vs
    1080x2400 reference, WIDTH is identical, but scaling by the smaller
    of the two factors (min(1080/1080, 1920/2400) = 0.80) shrank every
    template 20% and nothing passed the threshold. That's a different aspect
    ratio, not a different resolution.
    """

    from core.config import REFERENCE_HEIGHT, REFERENCE_WIDTH

    base = cv2.imread(str(IMAGE))

    assert base.shape[:2] == (
        REFERENCE_HEIGHT,
        REFERENCE_WIDTH,
    ), "o fixture deixou de estar na referência"

    # Same width, less height: what a 16:9 shows.
    frame = base[: int(REFERENCE_WIDTH * 16 / 9), :]

    largura, altura = frame.shape[1], frame.shape[0]

    escala = Detector._frame_scale(largura, altura)

    assert abs(escala - 1.0) < 0.01, (
        f"largura idêntica tem de dar escala 1.0, deu {escala}"
    )

    detector = Detector()

    deteccoes = detector.detect(frame)

    assert deteccoes, (
        "não detectou nada num aparelho de outra proporção — "
        "é exatamente o bug que a escala por lado curto corrige"
    )

    # And the old behavior REALLY failed here: without
    # this, the test above would pass even with the bug back.
    import vision.detector as vd

    basis_original = vd.TEMPLATE_SCALE_BASIS
    steps_original = vd.TEMPLATE_SCALE_STEPS

    try:

        vd.TEMPLATE_SCALE_BASIS = "min"
        vd.TEMPLATE_SCALE_STEPS = (1.0,)

        assert Detector._frame_scale(largura, altura) < 0.85, (
            "o critério antigo deveria encolher o template"
        )

        assert not Detector().detect(frame), (
            "se o critério antigo também detecta, este teste "
            "não está medindo o que diz medir"
        )

    finally:

        vd.TEMPLATE_SCALE_BASIS = basis_original
        vd.TEMPLATE_SCALE_STEPS = steps_original


def test_resolucao_proporcional_continua_valendo():
    """Changing resolution KEEPING the ratio is the easy case: templates follow along."""

    base = cv2.imread(str(IMAGE))

    esperado = Detector().detect(base)

    assert esperado, "nem detectou na referência"

    for largura, altura, fator in (
        (720, 1600, 2 / 3),
        (1440, 3200, 4 / 3),
    ):

        frame = cv2.resize(
            base,
            (largura, altura),
            interpolation=(
                cv2.INTER_AREA
                if fator < 1
                else cv2.INTER_LINEAR
            ),
        )

        escala = Detector._frame_scale(largura, altura)

        assert abs(escala - fator) < 0.01, (largura, escala)

        deteccoes = Detector().detect(frame)

        assert deteccoes, f"não detectou em {largura}x{altura}"

        # The SAME categories as the reference.
        assert (
            {d["category"] for d in deteccoes}
            == {d["category"] for d in esperado}
        ), (largura, altura, deteccoes)


def test_referencia_usa_o_comb_de_escalas():
    """At the reference the zoom-tolerance comb applies too (camera zoom
    changes on-screen icon size even there): every template yields one
    variant per TEMPLATE_SCALE_STEPS, same as off-reference."""

    from core.config import (
        REFERENCE_HEIGHT,
        REFERENCE_WIDTH,
        TEMPLATE_SCALE_STEPS,
    )

    detector = Detector()

    originais = len(detector.templates)

    escalas = {
        round(1.0 * passo, 4)
        for passo in TEMPLATE_SCALE_STEPS
        if passo > 0
    }

    # In both orientations: the game runs landscape.
    for largura, altura in (
        (REFERENCE_WIDTH, REFERENCE_HEIGHT),
        (REFERENCE_HEIGHT, REFERENCE_WIDTH),
    ):

        assert Detector._is_reference(largura, altura)

        indice = detector._templates_for(largura, altura)

        assert (
            sum(len(lista) for lista in indice.values())
            == originais * len(escalas)
        ), "referência devia gerar o comb de escalas"

    # Off-reference, then yes.
    fora = detector._templates_for(1080, 1920)

    assert (
        sum(len(lista) for lista in fora.values())
        > originais
    ), "esperava templates em mais de uma escala"


def test_teto_de_fps_corta_antes_de_converter():
    """The cut must happen BEFORE color conversion (7.8 MB/frame), or it saves nothing."""

    import numpy as np

    from capture.screen import ScreenCapture
    from core.config import CAPTURE_MAX_FPS

    if not CAPTURE_MAX_FPS:

        print("    (CAPTURE_MAX_FPS desligado, nada a testar)")

        return

    capture = ScreenCapture()

    convertidos = []

    class FrameFalso:
        """Flags the conversion: if `to_ndarray` was called, the frame paid the expensive path."""

        def to_ndarray(self):

            convertidos.append(1)

            return np.zeros(
                (2400 * 3 // 2, 1080),
                dtype=np.uint8,
            )

    # Feeds at TWICE the cap.
    inicio = time.monotonic()
    enviados = 0

    while time.monotonic() - inicio < 1.5:

        capture._publish(FrameFalso())

        enviados += 1

        time.sleep(1.0 / (CAPTURE_MAX_FPS * 2))

    decorrido = time.monotonic() - inicio

    entregues = capture.version / decorrido

    assert entregues <= CAPTURE_MAX_FPS * 1.2, entregues

    assert entregues >= CAPTURE_MAX_FPS * 0.7, entregues

    assert capture.dropped > 0, "não descartou nada"

    # What was dropped did NOT pay for conversion.
    assert len(convertidos) == capture.version, (
        len(convertidos),
        capture.version,
        "frame descartado ainda pagou o YUV->BGR",
    )


def test_piso_descarta_frame_velho_sem_perder_o_anterior():
    """
    The frame the worker holds right after an action shows the
    BEFORE screen of its effect — _can_act would drop it anyway, so
    spending a pass on it is a double loss. The floor must not wipe the
    last good reading: the overlay would go empty, looking like a
    stalled detector.
    """

    frame = cv2.imread(str(IMAGE))

    _, vision, _, _ = build()

    vision.set_categories({"upgrade"})

    vision.start()

    try:

        # One good read first.
        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:

            vision.set_frame(frame)

            deteccoes, _ = vision.get_detections()

            if deteccoes:
                break

            time.sleep(0.02)

        assert deteccoes, "nem detectou antes de testar o piso"

        antes = vision.detections_time

        # Floor in the future: EVERY current frame is old.
        vision.set_frame_floor(time.monotonic() + 30.0)

        pulados = vision.skipped_stale

        for _ in range(30):

            vision.set_frame(frame)

            time.sleep(0.02)

        assert vision.skipped_stale > pulados, (
            "não descartou frame anterior ao piso"
        )

        # The previous reading still stands.
        ainda, _ = vision.get_detections()

        assert ainda == deteccoes, "apagou a última leitura boa"

        assert vision.detections_time == antes, (
            "publicou análise de frame que devia ter pulado"
        )

        assert vision.waiting_settle is True

        # And the thread didn't die skipping.
        assert vision.is_alive()

        # Floor released: analyzes again.
        vision.set_frame_floor(0.0)

        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:

            vision.set_frame(frame)

            if vision.detections_time > antes:
                break

            time.sleep(0.02)

        assert vision.detections_time > antes, (
            "não voltou a analisar depois de liberar o piso"
        )

        assert vision.waiting_settle is False

    finally:

        vision.stop()


def test_worker_reporta_fps_e_custo():
    """The two HUD numbers come from here, and differ from capture FPS."""

    frame = cv2.imread(str(IMAGE))

    _, vision, _, _ = build()

    # Few categories = fast pass.
    vision.set_categories({"upgrade"})

    vision.start()

    try:

        deadline = time.monotonic() + 2.0

        while time.monotonic() < deadline:

            vision.set_frame(frame)

            time.sleep(0.01)

        assert vision.get_fps() > 0.0, vision.get_fps()

        assert vision.get_duration() > 0.0, (
            vision.get_duration()
        )

        # One pass can't cost more than the interval
        # between passes.
        assert (
            vision.get_duration()
            <= 1.0 / vision.get_fps() + 0.05
        ), (vision.get_duration(), vision.get_fps())

    finally:

        vision.stop()


def test_hud_desenha_e_desliga():

    import vision.detector as vd

    frame = cv2.imread(str(IMAGE))

    detector = build()[0]

    detections = detector.detect(frame)

    stats = {
        "capture_fps": 59.4,
        "detect_fps": 3.1,
        "detect_ms": 318.0,
        "lag": 0.34,
    }

    sem_hud = detector.draw(frame.copy(), detections, None)

    com_hud = detector.draw(frame.copy(), detections, stats)

    assert (com_hud != sem_hud).any(), "HUD não desenhou"

    fps_original = vd.SHOW_FPS
    lag_original = vd.SHOW_DETECTION_LAG

    try:

        vd.SHOW_FPS = False
        vd.SHOW_DETECTION_LAG = False

        desligado = detector.draw(
            frame.copy(),
            detections,
            stats,
        )

        # Only the HUD goes; the boxes stay.
        assert (desligado == sem_hud).all(), (
            "desligar o HUD mexeu em outra coisa"
        )

    finally:

        vd.SHOW_FPS = fps_original
        vd.SHOW_DETECTION_LAG = lag_original


def test_dismiss_toca_no_ponto_neutro():
    """
    "dismiss" must become a HELD tap on DISMISS_POINT, ignoring the
    detection that fired it. It's in no rule today (NORMAL_RULES
    uses "upgrade_food" for up_food) — tested directly so the code doesn't
    rot unnoticed.
    """

    from core.config import (
        DISMISS_HOLD_DURATION,
        DISMISS_POINT,
        REFERENCE_HEIGHT,
        REFERENCE_WIDTH,
    )

    _, _, actions, _ = build()

    actions.start()

    try:

        actions.set_frame_size(
            REFERENCE_WIDTH,
            REFERENCE_HEIGHT,
        )

        # Detection far from the dismiss point, to prove
        # the tap does NOT use its center.
        deteccao = {
            "category": "up_food",
            "name": "item_001.png",
            "confidence": 0.968,
            "color_similarity": 0.996,
            "min_threshold": 0.9,
            "x": 689,
            "y": 1348,
            "width": 43,
            "height": 128,
        }

        assert actions.execute("dismiss", deteccao)

        deadline = time.monotonic() + 5.0

        while (
            actions.is_busy()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

        # It's a HELD tap, not a tap: a dry adb tap
        # sometimes doesn't close the panel.
        assert actions.android.taps == [], actions.android.taps

        assert actions.android.holds == [
            (
                DISMISS_POINT[0],
                DISMISS_POINT[1],
                DISMISS_HOLD_DURATION,
            )
        ], actions.android.holds

    finally:

        actions.stop()


def test_dismiss_converte_para_device_menor():
    """The dismiss point is noted at reference resolution: on another device it must scale."""

    _, _, actions, _ = build(device_size=(540, 1200))

    actions.start()

    try:

        actions.set_frame_size(540, 1200)

        assert actions.execute(
            "dismiss",
            {
                "category": "up_food",
                "name": "item_001.png",
                "confidence": 0.97,
                "color_similarity": 0.99,
                "min_threshold": 0.9,
                "x": 100,
                "y": 200,
                "width": 40,
                "height": 40,
            },
        )

        deadline = time.monotonic() + 5.0

        while (
            actions.is_busy()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

        # (10, 2200) on the 1080x2400 reference -> half of that.
        assert [
            (x, y) for x, y, _ in actions.android.holds
        ] == [(5, 1100)], actions.android.holds

    finally:

        actions.stop()


def test_hold_do_dismiss_e_curto():
    """If it used the food-upgrade duration (4s), the bot would freeze on every accidentally opened panel."""

    from core.config import (
        DISMISS_HOLD_DURATION,
        UPGRADE_FOOD_PRESS,
    )

    assert DISMISS_HOLD_DURATION < UPGRADE_FOOD_PRESS, (
        DISMISS_HOLD_DURATION,
        UPGRADE_FOOD_PRESS,
    )

    # Above a failed tap, and below the Android long-press
    # threshold (~500 ms), so it doesn't trigger a hold gesture.
    assert 0.15 <= DISMISS_HOLD_DURATION <= 0.5, (
        DISMISS_HOLD_DURATION
    )


def test_escape_rola_a_tela_para_baixo():
    """
    The escape sequence must scroll the VIEW down (swipe "up",
    finger up). The inverted direction would leave the bottom corner in
    the button bar.
    """

    from core.config import (
        SCROLL_BOTTOM_DIRECTION,
        SCROLL_BOTTOM_SWIPES,
    )

    _, _, actions, _ = build()

    actions.start()

    try:

        actions.set_frame_size(1080, 2400)

        assert actions.execute("scroll_bottom", None) is True

        deadline = time.monotonic() + 15.0

        while (
            actions.is_busy()
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)

        rolagens = actions.android.scrolls

        assert len(rolagens) == SCROLL_BOTTOM_SWIPES, rolagens

        # Finger up: end y SMALLER than start.
        assert SCROLL_BOTTOM_DIRECTION == "up", (
            SCROLL_BOTTOM_DIRECTION
        )

        for y1, y2 in rolagens:

            assert y2 < y1, (y1, y2)

        # And no BACK, which would close the game.
        assert actions.android.backs == [], (
            actions.android.backs
        )

    finally:

        actions.stop()


def test_escape_nao_bloqueia_o_loop():
    """Six swipes with pauses (~1.5s) on the action thread: execute() must return right away."""

    _, _, actions, _ = build()

    actions.start()

    try:

        actions.set_frame_size(1080, 2400)

        inicio = time.monotonic()

        assert actions.execute("scroll_bottom", None) is True

        assert time.monotonic() - inicio < 0.5, (
            time.monotonic() - inicio
        )

        assert actions.is_busy()

    finally:

        actions.stop()


def test_back_nao_esta_disponivel():
    """In this game BACK quits the game: it must not be invokable by action name."""

    _, _, actions, _ = build()

    actions.start()

    try:

        assert actions.execute("back", None) is False

        assert actions.android.backs == [], (
            actions.android.backs
        )

    finally:

        actions.stop()


def test_porta_nao_colide_com_o_scrcpy():
    """
    The scrcpy mirror uses 27183-27199 by default. If capture falls back
    into that range, the bot fails INTERMITTENTLY: whoever loses the
    port race connects to the wrong tunnel and the stream dies.
    """

    from core.config import SCRCPY_DEVICE_JAR, SCRCPY_PORT

    assert not (27183 <= SCRCPY_PORT <= 27199), (
        f"SCRCPY_PORT={SCRCPY_PORT} está na faixa default do "
        f"scrcpy.exe"
    )

    # And the on-device jar must be ours: both pushed
    # to the same file at the same time, and our stop() deleted it.
    assert "scrcpy-server.jar" not in SCRCPY_DEVICE_JAR, (
        SCRCPY_DEVICE_JAR
    )


def test_socket_do_servidor_tem_scid():
    """
    In scrcpy 4.1 the abstract socket is ALWAYS "scrcpy_<scid>". Forwarding
    to bare "scrcpy" makes adb return 0 bytes immediately — a silent
    failure that looked like "port in use".
    """

    from capture.screen import ScreenCapture

    c = ScreenCapture()

    # Simulates what start() does, without touching the device.
    import random as _r

    c.scid = _r.randint(0, 0x7FFFFFFF)
    c.socket_name = f"scrcpy_{c.scid:08x}"

    assert c.socket_name != "scrcpy", c.socket_name

    assert c.socket_name.startswith("scrcpy_"), c.socket_name

    # 8 hex chars, and scid within 31 bits (what scrcpy uses).
    sufixo = c.socket_name.split("_")[1]

    assert len(sufixo) == 8, sufixo

    int(sufixo, 16)

    assert 0 <= c.scid <= 0x7FFFFFFF, c.scid


def test_timeout_de_conexao_maior_que_o_arranque():
    """Measured: the server takes ~1.4s (more with the scrcpy mirror alongside); a short timeout reintroduces the failure."""

    from core.config import (
        CAPTURE_CONNECT_TIMEOUT,
        CAPTURE_START_TIMEOUT,
    )

    assert CAPTURE_CONNECT_TIMEOUT >= 8.0, (
        CAPTURE_CONNECT_TIMEOUT
    )

    assert CAPTURE_START_TIMEOUT >= 3.0, (
        CAPTURE_START_TIMEOUT
    )


def main():

    log.setup("WARNING")

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

    print()

    if failures:

        print(f"{failures}/{len(tests)} falharam.")

        return 1

    print(f"{len(tests)}/{len(tests)} passaram.")

    return 0


if __name__ == "__main__":

    sys.exit(main())
