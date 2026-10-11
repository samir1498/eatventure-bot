"""EatVenture AI — ponto de entrada (python src/main.py)."""

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
    """Gravador do dataset, ou None se desligado. O banco é opcional; sem ele o samples.jsonl continua completo."""

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

                # Banco fora não pode impedir a coleta: imagens não se recuperam depois.
                logger.error(
                    "No database connection (%s) — writing "
                    "to file only. You can import later "
                    "with tools/dataset_import.py.",
                    error,
                )

                store = None

    return DatasetRecorder(store=store)


def read_detections(vision):
    """(detecções, idade, frame, instante_do_frame). O frame é o que GEROU as detecções, não o mais recente — mantém imagem e rótulo consistentes no dataset."""

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

    # Evita repetir o aviso de visão caída a cada quadro.
    vision_avisada = False

    # Zeros de propósito: a primeira volta já garante o jogo aberto.
    ultima_garantia_jogo = 0.0
    ultimo_reinicio_jogo = 0.0

    while True:

        # O jogo pode sair da frente sozinho (anúncio, sistema): sem
        # isto o bot age sobre a tela errada ou fica parado. dumpsys
        # custa ~50ms, por isso só a cada GAME_ENSURE_INTERVAL.
        agora = time.monotonic()

        if agora - ultima_garantia_jogo >= GAME_ENSURE_INTERVAL:

            ultima_garantia_jogo = agora

            pacote = actions.android.focused_package()

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

                    # Descarta o frame da tela antiga: a próxima
                    # volta já captura o jogo abrindo.
                    continue

        # Timeout curto para a janela do OpenCV continuar respondendo
        # (e o ESC funcionar) mesmo se o stream travar.
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

        # Resolução real do frame: o VisionWorker devolve as detecções
        # nesse espaço, e é isso que faz a conversão frame->device ser
        # exata em qualquer resolução de device (não só 1080x2400).
        altura_frame, largura_frame = frame.shape[:2]

        actions.set_frame_size(largura_frame, altura_frame)

        if VISION_FILTER_BY_STATE:

            vision.set_categories(
                state_machine.wanted_categories()
            )

        # Frame anterior à última ação não pode gerar detecção válida
        # (_can_act descarta), então o worker nem gasta uma passada nele.
        vision.set_frame_floor(state_machine.frame_floor())

        vision.set_frame(frame, timestamp)

        # get_input (não get_detections): devolve também o frame que
        # PRODUZIU as detecções, para dataset não gravar imagem/rótulo
        # de passadas diferentes.
        detections, lag, detect_frame, detect_time = (
            read_detections(vision)
        )

        # Thread do detector morta não para o programa sozinha (captura e
        # janela seguem ativas, o bot só some de agir) — loga uma vez.
        if not vision.is_alive() and not vision_avisada:

            logger.error(
                "Vision thread is not running (%s) — "
                "there will be no detections and the bot "
                "will not act.",
                vision.last_error or "unknown reason",
            )

            vision_avisada = True

        # Sem intervalo mínimo aqui: o cooldown de QUANDO agir já vive
        # na StateMachine, que considera a idade do frame.
        state_machine.update(
            detections,
            lag,
            detect_frame,
            detect_time,
        )

        # Depois do update, para mostrar o estado já com o efeito desta
        # volta. O panel decide sozinho se redesenha, então chamar
        # sempre não custa.
        if panel is not None:

            # Bateria via BatteryMonitor (memória): dumpsys custa ~56ms
            # e não pode entrar no loop.
            panel.update(
                state_machine.summary(),
                extra=(
                    f"{vision.get_fps():.1f} fps"
                    f" | {formata_bateria(battery.get())}"
                ),

                # Vazio com DETECTOR_DEBUG_MISSES desligado.
                misses=detector.miss_report(),
            )

        # Sem janela, não há cópia nem desenho do frame.
        if not SHOW_AI_VISION:
            continue

        # Reduz antes de desenhar: `resize` já substitui a cópia e o
        # desenho toca só a resolução da janela (bem menor que o frame).
        # As detecções são convertidas pelo mesmo fator dentro do `draw`.
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

            # Frame já pequeno: cópia necessária porque `draw` escreve
            # no array recebido, e este é o buffer da captura.
            ai_frame = frame.copy()

        ai_frame = detector.draw(
            ai_frame,
            detections,
            {
                "capture_fps": capture.get_fps(),
                "detect_fps": vision.get_fps(),
                "detect_ms": vision.get_duration() * 1000,
                "lag": lag,

                # Leitura em memória: dumpsys custa ~56 ms e não pode entrar aqui.
                "battery": battery.get(),

                "cycle": state_machine.cycle_stats(),

                # Sem isto, overlay vazio não diz se o problema é
                # template, threshold ou estado errado.
                "state": state_machine.state,
                "searched": detector.last_searched,
                "searchable": detector.last_total,
                "detections": len(detections),

                # Cobertura da pasta da resolução sobre o default.
                "coverage": detector.template_coverage(),

                # Overlay parado após uma ação é ESPERADO (worker
                # pulando frames de antes do efeito assentar), não defeito.
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

        # ESC só chega aqui, é esta janela que recebe teclas.
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main(argv=None):

    log.setup(LOG_LEVEL)

    args = parse_args(argv)

    # --device manda no config, o config manda na pergunta.
    # Ctrl+C aqui é desistência do usuário, não erro.
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

    # Espelho do scrcpy é opcional: a captura do bot não passa por ele.
    scrcpy_process = (
        start_scrcpy(device_id)
        if SHOW_SCRCPY
        else None
    )

    capture = ScreenCapture(device_id)

    detector = Detector(
        use_defaults=not args.layout_only,
        # O aprendizado de override por resolução não depende da coleta
        # do dataset: ele deve continuar sempre que uma detecção válida
        # aparecer, mesmo sem o modo de IA-collect ligado.
        learn_only_actions=False,
    )

    vision = VisionWorker(detector)

    actions = ActionManager(device_id)

    recorder = build_recorder(ai_collect=args.ai_collect)

    state_machine = StateMachine(actions, recorder, detector)

    battery = BatteryMonitor(actions.android)

    # device_id já vem resolvido do adb: é o serial real em uso mesmo
    # quando veio da pergunta interativa, não do config.
    panel = (
        StatusPanel(
            device_id,
            interval=STATUS_PANEL_INTERVAL,
        )
        if STATUS_PANEL
        else None
    )

    if panel is not None:

        # Sem isto, cada ação imprime uma linha de INFO e empurra o
        # painel, os dois brigando pelo mesmo terminal.
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

        # Antes de qualquer log: devolve o cursor para baixo do painel,
        # senão as linhas de encerramento escrevem em cima dele.
        if panel is not None:

            panel.close()

            log.set_console_level(LOG_LEVEL)

        logger.info("Shutting down EatVenture AI...")

        # Antes do resto: fecha a amostra pendente e grava o último lote no banco.
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
