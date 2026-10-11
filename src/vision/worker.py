"""Worker thread: detector assíncrono com filtro de categorias, sem letterbox, espera bloqueante."""

import threading
import time

from core import log
from core.config import VISION_INTERVAL
from core.metrics import DurationMeter, RateMeter

logger = log.get("vision")


class VisionWorker:

    def __init__(self, detector, interval=VISION_INTERVAL):

        self.detector = detector

        self.interval = interval

        self.running = False
        self.thread = None

        # Inicializando para evitar AttributeError na primeira volta de _run
        self.latest_raw_frame = None
        self.latest_frame_time = 0.0

        self.detections = []
        self.detections_time = 0.0

        # Frame que produziu as detecções acima. Não é o
        # latest_frame: aquele já foi substituído.
        self.detections_frame = None

        self.categories = None

        # Frame novo acorda o worker (substitui busy-wait)
        self.frame_lock = threading.Condition()

        self.detection_lock = threading.Lock()

        # Última falha da thread, para o loop principal e o
        # HUD poderem dizer POR QUE não há detecção.
        self.last_error = None
        self._error_kinds = set()

        # Frames anteriores a este timestamp são rejeitados (estabilização pós-ação)
        self.frame_floor = 0.0

        # Frames descartados por serem velhos demais para
        # decidir. Para o HUD explicar um overlay parado.
        self.skipped_stale = 0

        # Esperando a tela estabilizar depois de uma ação?
        self.waiting_settle = False

        # Custo real de uma passada, e quantas por segundo.
        #
        # É o número que diz se o bot está enxergando rápido
        # o bastante — não o FPS da captura.
        self.last_duration = 0.0

        self.duration = DurationMeter()

        self.rate = RateMeter()

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            name="vision-worker",
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread:

            self.thread.join(timeout=5)

            self.thread = None

    def set_frame(self, frame, timestamp=None):
        """Frame bruto sem processamento. Detector reescala templates por resolução."""

        with self.frame_lock:

            self.latest_raw_frame = frame

            self.latest_frame_time = (
                time.monotonic() if timestamp is None else timestamp
            )

            self.frame_lock.notify()

    def set_frame_floor(self, timestamp):
        """Rejeita frames anteriores a este timestamp (estabilização pós-ação)."""

        with self.frame_lock:

            self.frame_floor = timestamp or 0.0

    def set_categories(self, categories):
        """Restringe busca por categoria. Tupla preserva prioridade (ordem importa)."""

        with self.detection_lock:

            self.categories = (
                None
                if categories is None
                else tuple(categories)
            )

    def get_input(self):
        """Retorna (frame, detecções, timestamp) consistentes para dataset."""

        with self.detection_lock:

            return (
                self.detections_frame,
                list(self.detections),
                self.detections_time,
            )

    def get_detections(self):
        """Retorna (detecções, idade_em_segundos). Idade é do frame, não da consulta."""

        with self.detection_lock:

            detections = list(self.detections)

            frame_time = self.detections_time

        if not frame_time:
            return detections, 0.0

        return (
            detections,
            max(0.0, time.monotonic() - frame_time),
        )

    def get_fps(self):
        return self.rate.rate()

    def get_duration(self):
        return self.duration.average()

    def is_alive(self):
        """Thread da visão está viva? (detecção de morte silenciosa do worker)."""

        return bool(
            self.running
            and self.thread is not None
            and self.thread.is_alive()
        )

    def _fail(self, error):
        """Loga erro UMA vez por tipo (evita spam em laço rápido)."""

        self.last_error = f"{type(error).__name__}: {error}"

        if type(error).__name__ in self._error_kinds:
            return

        self._error_kinds.add(type(error).__name__)

        logger.exception(
            "Detector pass failed — %s. Thread "
            "continues; this message does not repeat.",
            self.last_error,
        )

    def _run(self):
        """Worker protegido: erros são capturados para evitar morte silenciosa."""

        while self.running:

            try:

                self._pass()

            except Exception as error:

                self._fail(error)

                # Sem isto, um erro imediato viraria laço
                # quente consumindo uma CPU inteira.
                time.sleep(0.1)

    def _pass(self):
        """Uma passada: aguarda frame novo, detecta, publica."""
        # Descarta frames atrasados (só o mais novo interessa)
        with self.frame_lock:

            if self.latest_raw_frame is None:

                # Dorme até chegar frame. O timeout é só para
                # o self.running voltar a ser consultado.
                self.frame_lock.wait(0.2)

            raw_frame = self.latest_raw_frame
            frame_time = self.latest_frame_time

            # Limpa para sinalizar que foi consumido.
            self.latest_raw_frame = None

            floor = self.frame_floor

        if raw_frame is None:
            return

        # Pula frame velho demais (mostra tela ANTES do efeito da ação anterior)
        if floor and frame_time and frame_time <= floor:

            self.skipped_stale += 1

            self.waiting_settle = True

            return

        self.waiting_settle = False

        with self.detection_lock:
            categories = (
                None if self.categories is None else self.categories
            )

        started = time.monotonic()

        detections = self.detector.detect(
            raw_frame,
            categories,
        )

        self.last_duration = time.monotonic() - started

        self.duration.add(self.last_duration)

        self.rate.tick()

        with self.detection_lock:

            self.detections = detections
            self.detections_time = frame_time

            self.detections_frame = raw_frame

        if self.interval:
            time.sleep(self.interval)
