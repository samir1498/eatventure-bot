"""EatVenture AI — entry point (python src/main.py)."""

import argparse
import subprocess
import sys
import time

import cv2

from actions.manager import ActionManager
from capture.screen import ScreenCapture
from core import devices, log
from core.battery import BatteryMonitor
from dataset.recorder import DatasetRecorder
from core.config import (
    AI_WINDOW_NAME,
    AI_WINDOW_POSITION,
    DATASET_DB_BATCH,
    DATASET_DB_DSN,
    DATASET_DB_ENABLED,
    DATASET_DB_SCHEMA,
    DEVICE_SERIAL,
    GAME_ENSURE_INTERVAL,
    GAME_PACKAGE,
    GAME_RELAUNCH_COOLDOWN,
    LOG_LEVEL,
    DATASET_SAVE,
    SCRCPY_EXTRA_ARGS,
    SCRCPY_PATH,
    SHOW_AI_VISION,
    SHOW_SCRCPY,
    STATUS_PANEL,
    STATUS_PANEL_INTERVAL,
    VISION_FILTER_BY_STATE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from core.state_machine import StateMachine
from core.status import StatusPanel, formata_bateria
from vision.detector import Detector
from vision.worker import VisionWorker

logger = log.get("main")


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="EatVenture AI",
    )

    parser.add_argument(
        "--device",
        metavar="SERIAL",
        help=(
            "serial do device (adb devices). Sem isto, usa "
            "DEVICE_SERIAL do config; sem os dois, pergunta "
            "quando houver mais de um conectado."
        ),
    )

    parser.add_argument(
        "--ai-collect",
        action="store_true",
        help="Ativa coleta de dados de IA para o dataset.",
    )

    parser.add_argument(
        "--layout-only",
        action="store_true",
        help=(
            "detect() busca só os overrides da resolução, sem os "
            "templates default (mais rápido, mas cego onde ainda não há "
            "override). Rode src/main_layouts.py em paralelo para manter "
            "os overrides atualizados."
        ),
    )

    return parser.parse_args(argv)


def start_scrcpy(device_id):

    window_title = f"EatVenture ({device_id})"

    logger.info("Starting %s...", window_title)

    return subprocess.Popen(
        [
            SCRCPY_PATH,

            "--serial",
            device_id,

            "--window-title",
            window_title,
        ]
        + list(SCRCPY_EXTRA_ARGS)
    )


def setup_window():

    cv2.namedWindow(AI_WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(
        AI_WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    cv2.moveWindow(AI_WINDOW_NAME, *AI_WINDOW_POSITION)


def build_recorder(ai_collect=None):
    """Dataset recorder, or None when disabled. The database is optional; without it samples.jsonl stays complete."""

    dataset_enabled = ai_collect if ai_collect is not None else DATASET_SAVE

    if not dataset_enabled:
        return None

    store = None

    if DATASET_DB_ENABLED:

        if not DATASET_DB_DSN:

            logger.error(
                "DATASET_DB_ENABLED on but "
                "DATASET_DB_DSN empty — writing to "
                "file only. See docs/dataset.md."
            )

        else:

            from dataset.store import PostgresStore

            try:

                store = PostgresStore(
                    DATASET_DB_DSN,
                    batch_size=DATASET_DB_BATCH,
                    schema=DATASET_DB_SCHEMA,
                ).connect()

            except Exception as error:

                # Database down must not block collection: images cannot be recovered later.
                logger.error(
                    "No database connection (%s) — writing "
                    "to file only. You can import later "
                    "with tools/dataset_import.py.",
                    error,
                )

                store = None

    return DatasetRecorder(store=store)


def read_detections(vision):
    """(detections, age, frame, frame instant). The frame is the one that GENERATED the detections, not the newest — keeps image and label consistent in the dataset."""

    frame, detections, frame_time = vision.get_input()

    lag = (
        max(0.0, time.monotonic() - frame_time)
        if frame_time
        else 0.0
    )

    return detections, lag, frame, frame_time


def run_loop(
    capture,
    vision,
    detector,
    state_machine,
    actions,
    battery,
    panel=None,
):

    last_version = 0

    # Avoid repeating the vision-down warning on every frame.
    vision_avisada = False

    # Zeros on purpose: the first pass already ensures the game is open.
    ultima_garantia_jogo = 0.0
    ultimo_reinicio_jogo = 0.0

    while True:

        # The game can leave the foreground on its own (ads, system): without
        # this the bot acts on the wrong screen or stalls. dumpsys
        # costs ~50ms, hence only every GAME_ENSURE_INTERVAL.
        agora = time.monotonic()

        if agora - ultima_garantia_jogo >= GAME_ENSURE_INTERVAL:

            ultima_garantia_jogo = agora

            pacote = actions.android.focused_package()

            # Feeds the state machine's foreground gate: while another
            # app is in front, _act holds every tap. None = unknown,
            # keep the previous value.
            if pacote is not None:
                state_machine.set_game_foreground(pacote == GAME_PACKAGE)

            if (
                pacote is not None
                and pacote != GAME_PACKAGE
                and agora - ultimo_reinicio_jogo >= GAME_RELAUNCH_COOLDOWN
            ):

                logger.warning(
                    "Game not in foreground (%s) — relaunching %s.",
                    pacote,
                    GAME_PACKAGE,
                )

                actions.android.collapse_shade()

                if actions.android.launch_package(GAME_PACKAGE):

                    ultimo_reinicio_jogo = agora

                    # Stays inhibited until the next ensure check
                    # sees the game package back in front.
                    state_machine.set_game_foreground(False)

                    # Discards the old-screen frame: the next
                    # pass already captures the game opening.
                    continue

        # Short timeout so the OpenCV window stays responsive
        # (and ESC works) even if the stream stalls.
        frame, version, timestamp = capture.get_frame(
            since_version=last_version,
            timeout=0.2,
        )

        if not capture.is_running():

            logger.warning("Stream ended.")

            break

        if frame is None:
            continue

        last_version = version

        # Real frame resolution: the VisionWorker returns detections
        # in this space, and that is what makes the frame->device conversion
        # exact at any device resolution (not just 1080x2400).
        altura_frame, largura_frame = frame.shape[:2]

        actions.set_frame_size(largura_frame, altura_frame)

        if VISION_FILTER_BY_STATE:

            vision.set_categories(
                state_machine.wanted_categories()
            )

        # A frame older than the last action cannot produce a valid detection
        # (_can_act discards it), so the worker does not even spend a pass on it.
        vision.set_frame_floor(state_machine.frame_floor())

        vision.set_frame(frame, timestamp)

        # get_input (not get_detections): also returns the frame that
        # PRODUCED the detections, so the dataset does not record image/label
        # from different passes.
        detections, lag, detect_frame, detect_time = (
            read_detections(vision)
        )

        # A dead detector thread does not stop the program on its own (capture and
        # window stay active, the bot just stops acting) — log once.
        if not vision.is_alive() and not vision_avisada:

            logger.error(
                "Vision thread is not running (%s) — "
                "there will be no detections and the bot "
                "will not act.",
                vision.last_error or "unknown reason",
            )

            vision_avisada = True

        # No minimum interval here: the cooldown for WHEN to act already lives
        # in the StateMachine, which accounts for frame age.
        state_machine.update(
            detections,
            lag,
            detect_frame,
            detect_time,
        )

        # After the update, to show the state with this pass's
        # effect already applied. The panel decides on its own whether to redraw, so calling
        # it every time costs nothing.
        if panel is not None:

            # Battery via BatteryMonitor (memory): dumpsys costs ~56ms
            # and cannot run in the loop.
            panel.update(
                state_machine.summary(),
                extra=(
                    f"{vision.get_fps():.1f} fps"
                    f" | {formata_bateria(battery.get())}"
                ),

                # Empty with DETECTOR_DEBUG_MISSES off.
                misses=detector.miss_report(),
            )

        # Without a window, there is no frame copy or drawing.
        if not SHOW_AI_VISION:
            continue

        # Downscale before drawing: `resize` already replaces the copy and the
        # drawing only touches the window resolution (much smaller than the frame).
        # Detections are converted by the same factor inside `draw`.
        escala_janela = min(
            WINDOW_WIDTH / largura_frame,
            WINDOW_HEIGHT / altura_frame,
            1.0,
        )

        if escala_janela < 1.0:

            ai_frame = cv2.resize(
                frame,
                (
                    int(round(largura_frame * escala_janela)),
                    int(round(altura_frame * escala_janela)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:

            # Frame already small: copy needed because `draw` writes
            # into the received array, and this is the capture buffer.
            ai_frame = frame.copy()

        ai_frame = detector.draw(
            ai_frame,
            detections,
            {
                "capture_fps": capture.get_fps(),
                "detect_fps": vision.get_fps(),
                "detect_ms": vision.get_duration() * 1000,
                "lag": lag,

                # In-memory read: dumpsys costs ~56 ms and cannot run here.
                "battery": battery.get(),

                "cycle": state_machine.cycle_stats(),

                # Without this, an empty overlay does not say whether the problem is
                # a template, threshold, or wrong state.
                "state": state_machine.state,
                "searched": detector.last_searched,
                "searchable": detector.last_total,
                "detections": len(detections),

                # Coverage of the resolution folder over the default.
                "coverage": detector.template_coverage(),

                # A frozen overlay after an action is EXPECTED (worker
                # skipping frames from before the effect settles), not a bug.
                "waiting_settle": vision.waiting_settle,
                "vision_error": (
                    None
                    if vision.is_alive()
                    else (
                        vision.last_error
                        or "vision thread stopped"
                    )
                ),
            },
            scale=escala_janela,
        )

        cv2.imshow(AI_WINDOW_NAME, ai_frame)

        # ESC only arrives here; this window receives keys.
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main(argv=None):

    log.setup(LOG_LEVEL)

    args = parse_args(argv)

    # --device overrides config, config overrides the prompt.
    # Ctrl+C here is user cancellation, not an error.
    try:

        device_id = devices.resolver(args.device or DEVICE_SERIAL)

    except KeyboardInterrupt:

        logger.info("Cancelled.")

        return 1

    windows = [
        name
        for name, enabled in (
            ("IA", SHOW_AI_VISION),
            ("scrcpy", SHOW_SCRCPY),
        )
        if enabled
    ]

    logger.info(
        "Windows: %s | quit with %s",
        ", ".join(windows) or "none (headless)",
        "ESC or Ctrl+C" if SHOW_AI_VISION else "Ctrl+C",
    )

    # The scrcpy mirror is optional: the bot's capture does not go through it.
    scrcpy_process = (
        start_scrcpy(device_id)
        if SHOW_SCRCPY
        else None
    )

    capture = ScreenCapture(device_id)

    detector = Detector(
        use_defaults=not args.layout_only,
        # Resolution-override learning does not depend on dataset
        # collection: it should keep going whenever a valid detection
        # shows up, even without AI-collect mode on.
        learn_only_actions=False,
    )

    vision = VisionWorker(detector)

    actions = ActionManager(device_id)

    recorder = build_recorder(ai_collect=args.ai_collect)

    state_machine = StateMachine(actions, recorder, detector)

    battery = BatteryMonitor(actions.android)

    # device_id already comes resolved from adb: it is the real serial in use even
    # when it came from the interactive prompt, not from config.
    panel = (
        StatusPanel(
            device_id,
            interval=STATUS_PANEL_INTERVAL,
        )
        if STATUS_PANEL
        else None
    )

    if panel is not None:

        # Without this, each action prints an INFO line and pushes the
        # panel, both fighting over the same terminal.
        log.set_console_level("WARNING")

    try:

        capture.start()

        vision.start()

        actions.start()

        battery.start()

        if recorder:

            recorder.start()

        if SHOW_AI_VISION:

            setup_window()

        run_loop(
            capture,
            vision,
            detector,
            state_machine,
            actions,
            battery,
            panel,
        )

    except KeyboardInterrupt:

        logger.info("Interrupted by user.")

    finally:

        # Before any log: moves the cursor below the panel,
        # otherwise the shutdown lines print over it.
        if panel is not None:

            panel.close()

            log.set_console_level(LOG_LEVEL)

        logger.info("Shutting down EatVenture AI...")

        # Before the rest: closes the pending sample and flushes the last batch to the database.
        if recorder:

            recorder.stop()

        battery.stop()

        vision.stop()

        actions.stop()

        capture.stop()

        cv2.destroyAllWindows()

        if scrcpy_process:

            scrcpy_process.terminate()

            try:

                scrcpy_process.wait(timeout=2)

            except subprocess.TimeoutExpired:

                scrcpy_process.kill()


if __name__ == "__main__":

    sys.exit(main())
