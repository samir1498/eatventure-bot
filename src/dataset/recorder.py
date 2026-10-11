"""Grava dataset: frame, detecções (caixas+categoria), ação e resultado. Assíncrono com fila descartável."""

import hashlib
import json
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from core import log
from dataset import layout
from core.config import (
    DATASET_ACTIONS,
    DATASET_DIR,
    DATASET_HASH_SIZE,
    DATASET_IMAGE_FORMAT,
    DATASET_JPEG_QUALITY,
    DATASET_MAX_SAMPLES,
    DATASET_MAX_DISK_MB,
    DATASET_MAX_UNCHANGED_STREAK,
    DATASET_NEGATIVE_INTERVAL,
    DATASET_SCREEN_CHANGE,
    DATASET_OUTCOME_TIMEOUT,
    DATASET_QUEUE_SIZE,
)

logger = log.get("dataset")


# Resultados possíveis de uma ação.
CHANGED = "changed"        # o alvo saiu da tela: a ação pegou
UNCHANGED = "unchanged"    # o alvo continua lá: não pegou
UNKNOWN = "unknown"        # não chegou frame para julgar


class Sample:
    """Amostra em construção: fica pendente entre a ação e o frame que mostra o efeito dela."""

    __slots__ = (
        "id",
        "created",
        "action_time",
        "frame",
        "detections",
        "action",
        "action_kind",
        "target",
        "click",
        "state",
        "lag",
        "cycle_count",
        "outcome",
        "outcome_after",
        "thumb",
    )

    def __init__(self, **campos):

        for nome in self.__slots__:
            setattr(self, nome, campos.get(nome))


class DatasetRecorder:

    def __init__(
        self,
        directory=None,
        store=None,
        session=None,
        enabled=True,
    ):

        self.root = Path(
            DATASET_DIR if directory is None else directory
        )

        # Índice consultável (Postgres). None = só arquivos.
        self.store = store

        self.session = session or uuid.uuid4().hex

        self.enabled = enabled

        # Destino é por resolução e decidido por amostra: o device pode
        # trocar de resolução no meio da sessão (outro aparelho, rotação).
        self.data_dir = self.root / layout.DATA_DIRNAME

        # Fila limitada e descartável: o bot nunca espera o disco.
        self.queue = queue.Queue(maxsize=DATASET_QUEUE_SIZE)

        self.running = False
        self.thread = None

        self.pending = None
        self.lock = threading.Lock()

        # Corta amostragem de ação travada (medido: 35 amostras/30s repetindo a mesma tela).
        # Dedup por conteúdo não serve aqui pois a tela anima sozinha; o que separa
        # limpo é o RESULTADO (unchanged em sequência = travado).
        self.unchanged_action = None
        self.unchanged_streak = 0

        self.saved = 0
        self.dropped = 0
        self.duplicates = 0
        self.by_action = {}

        # Última negativa, para espaçá-las no TEMPO. Começa no
        # passado para a primeira sair logo.
        self.last_negative = 0.0

        # Bytes já gravados nesta execução, para o teto de
        # disco. Só conta o que esta sessão escreveu: varrer a
        # pasta a cada amostra custaria I/O à toa.
        self.bytes_written = 0

        self.disk_warned = False

    def start(self):

        if self.running or not self.enabled:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            name="dataset",
            daemon=True,
        )

        self.thread.start()

        logger.info(
            "Recording dataset to %s (session %s)",
            self.root,
            self.session[:8],
        )

    def stop(self):

        if not self.running:
            return

        # Fecha a pendente com veredito UNKNOWN em vez de perdê-la.
        self._flush_pending(UNKNOWN, None)

        self.running = False

        try:
            self.queue.put_nowait(None)

        except queue.Full:
            pass

        if self.thread:

            self.thread.join(timeout=10.0)

            self.thread = None

        logger.info(
            "Dataset: %d saved (%.1f MB), %d duplicates, "
            "%d dropped (queue full) %s",
            self.saved,
            self.bytes_written / 1_000_000,
            self.duplicates,
            self.dropped,
            dict(sorted(self.by_action.items())) or "",
        )

        if self.store:

            self.store.close()

    def on_action(
        self,
        action,
        action_kind,
        detection,
        click,
        state,
        frame,
        detections,
        lag,
        cycle_count=0,
    ):
        """Chamado quando uma ação SAIU de verdade. `click` está em coordenada do FRAME."""

        if not self.running:
            return

        if action not in DATASET_ACTIONS:
            return

        if frame is None:
            return

        if DATASET_MAX_SAMPLES and self.saved >= DATASET_MAX_SAMPLES:
            return

        # A pendente anterior nunca recebeu veredito.
        self._flush_pending(UNKNOWN, None)

        agora = time.monotonic()

        with self.lock:

            self.pending = Sample(
                id=uuid.uuid4().hex,
                created=datetime.now(timezone.utc),
                action_time=agora,

                # Cópia própria: o frame do ScreenCapture não sobrevive além do loop.
                frame=frame.copy(),

                detections=list(detections or []),

                # Miniatura do ANTES, para julgar ação sem alvo pela mudança de tela.
                thumb=self._thumb(frame),

                action=action,
                action_kind=action_kind,
                target=detection,
                click=click,
                state=state,
                lag=lag,
                cycle_count=cycle_count,
            )

    def observe(self, frame, detections, frame_time, state):
        """Julga a ação pendente e, de vez em quando, recolhe uma amostra NEGATIVA
        (tela sem alvo) — senão o detector aprende que sempre existe alvo."""

        if not self.running:
            return

        self._resolve(frame, detections, frame_time)

        self._maybe_negative(frame, detections, state)

    def _resolve(self, frame, detections, frame_time):

        with self.lock:

            pendente = self.pending

        if pendente is None:
            return

        # O frame precisa ser POSTERIOR à ação, senão não pode
        # mostrar o efeito dela.
        if frame_time is None or frame_time <= pendente.action_time:

            espera = time.monotonic() - pendente.action_time

            if espera > DATASET_OUTCOME_TIMEOUT:

                self._flush_pending(UNKNOWN, espera)

            return

        depois = frame_time - pendente.action_time

        self._flush_pending(
            self._julgar(pendente, frame, detections),
            depois,
        )

    def _julgar(self, pendente, frame, detections):
        """Com alvo, julga se ele saiu da tela; sem alvo (swipe/scroll/dismiss), se a TELA mudou."""

        if pendente.target is not None:

            return (
                UNCHANGED
                if self._alvo_continua(pendente, detections)
                else CHANGED
            )

        mudou = self._mudou(pendente.thumb, self._thumb(frame))

        if mudou is None:
            return UNKNOWN

        return CHANGED if mudou else UNCHANGED

    def _alvo_continua(self, pendente, detections):
        """Checa categoria E posição — só categoria daria falso "continua" com
        outra instância da mesma categoria em outro canto da tela (comum com comida)."""

        alvo = pendente.target

        folga = max(
            20,
            alvo.get("width", 0) // 2,
            alvo.get("height", 0) // 2,
        )

        for outra in detections or []:

            if outra["category"] != alvo["category"]:
                continue

            if (
                abs(outra["x"] - alvo["x"]) <= folga
                and abs(outra["y"] - alvo["y"]) <= folga
            ):
                return True

        return False

    def _flush_pending(self, outcome, depois):

        with self.lock:

            pendente = self.pending

            self.pending = None

        if pendente is None:
            return

        pendente.outcome = outcome
        pendente.outcome_after = depois

        self._submit(pendente)

    def _maybe_negative(self, frame, detections, state):

        if detections or frame is None:
            return

        if DATASET_NEGATIVE_INTERVAL <= 0:
            return

        agora = time.monotonic()

        # Espaçadas no TEMPO, não sorteadas: observe() é chamada
        # a cada passada do loop (~30x/s), e uma probabilidade
        # por passada enche o disco.
        if agora - self.last_negative < DATASET_NEGATIVE_INTERVAL:
            return

        if DATASET_MAX_SAMPLES and self.saved >= DATASET_MAX_SAMPLES:
            return

        self.last_negative = agora

        self._submit(
            Sample(
                id=uuid.uuid4().hex,
                created=datetime.now(timezone.utc),
                action_time=time.monotonic(),
                frame=frame.copy(),
                detections=[],
                thumb=self._thumb(frame),
                action=None,
                action_kind=None,
                target=None,
                click=None,
                state=state,
                lag=0.0,
                cycle_count=0,
                outcome=None,
                outcome_after=None,
            )
        )

    def _submit(self, sample):

        try:

            self.queue.put_nowait(sample)

        except queue.Full:

            self.dropped += 1

            # Um aviso por bloco, não um por amostra.
            if self.dropped % 20 == 1:

                logger.warning(
                    "Dataset queue full — %d sample(s) "
                    "dropped. Slow disk, or "
                    "DATASET_IMAGE_FORMAT in png with the bot "
                    "too fast?",
                    self.dropped,
                )

    def _run(self):

        while self.running:

            try:

                sample = self.queue.get(timeout=0.2)

            except queue.Empty:
                continue

            if sample is None:
                break

            try:

                self._write(sample)

            except Exception as error:

                logger.exception(
                    "Error writing sample: %s",
                    error,
                )

    def _write(self, sample):

        frame = sample.frame

        if frame is None:
            return

        # Teto de disco: o bot roda horas sem ninguém olhando.
        if (
            DATASET_MAX_DISK_MB
            and self.bytes_written >= DATASET_MAX_DISK_MB * 1_000_000
        ):

            if not self.disk_warned:

                logger.warning(
                    "Dataset reached the %d MB cap — stopping "
                    "writes. The bot keeps playing. Raise "
                    "DATASET_MAX_DISK_MB, switch "
                    "DATASET_IMAGE_FORMAT to jpg (4.3x smaller) "
                    "or move the folder.",
                    DATASET_MAX_DISK_MB,
                )

                self.disk_warned = True

            return

        # Mesma ação dando unchanged repetidamente = bot preso; para de gravar a mesma tela.
        if sample.action and sample.outcome == UNCHANGED:

            if sample.action == self.unchanged_action:

                self.unchanged_streak += 1

            else:

                self.unchanged_action = sample.action
                self.unchanged_streak = 1

            if self.unchanged_streak > DATASET_MAX_UNCHANGED_STREAK:

                self.duplicates += 1

                if self.unchanged_streak == (
                    DATASET_MAX_UNCHANGED_STREAK + 1
                ):

                    logger.info(
                        "'%s' stuck in unchanged — stopping "
                        "writes until the action changes or succeeds",
                        sample.action,
                    )

                return

        else:

            # Progresso (ou negativa): a contagem recomeça.
            self.unchanged_action = None
            self.unchanged_streak = 0

        chave = self._perceptual_hash(frame)

        dia = sample.created.strftime("%Y-%m-%d")

        altura_frame, largura_frame = sample.frame.shape[:2]

        shard = layout.shard_root(
            self.root,
            largura_frame,
            altura_frame,
        )

        pasta = layout.images_dir(shard) / dia

        pasta.mkdir(parents=True, exist_ok=True)

        nome = f"{sample.id}.{DATASET_IMAGE_FORMAT}"

        caminho = pasta / nome

        params = []

        if DATASET_IMAGE_FORMAT.lower() in ("jpg", "jpeg"):

            params = [
                cv2.IMWRITE_JPEG_QUALITY,
                int(DATASET_JPEG_QUALITY),
            ]

        if not cv2.imwrite(str(caminho), frame, params):

            logger.warning(
                "Could not write %s",
                caminho,
            )

            return

        registro = self._record(sample, caminho, chave, shard)

        with layout.index_path(shard).open(
            "a",
            encoding="utf-8",
        ) as arquivo:

            arquivo.write(
                json.dumps(registro, ensure_ascii=False) + "\n"
            )

        self.saved += 1

        self.bytes_written += caminho.stat().st_size

        rotulo = sample.action or "negative"

        self.by_action[rotulo] = self.by_action.get(rotulo, 0) + 1

        # Índice no banco é opcional; falha aqui não pode derrubar a gravação (JSONL é a fonte de verdade).
        if self.store:

            try:

                self.store.insert(registro)

            except Exception as error:

                logger.warning(
                    "Failed to index in database (file "
                    "was written): %s",
                    error,
                )

    def _record(self, sample, caminho, phash, shard):

        altura, largura = sample.frame.shape[:2]

        alvo = sample.target

        return {
            "id": sample.id,
            "session": self.session,
            "created_at": sample.created.isoformat(),

            # Relativo à raiz DA RESOLUÇÃO: a pasta daquela resolução
            # inteira pode ser movida, copiada ou apagada sem invalidar
            # o índice dela nem o das outras.
            "image": str(
                caminho.relative_to(shard).as_posix()
            ),

            "image_sha256": self._sha256(caminho),
            "phash": phash,

            "frame_width": largura,
            "frame_height": altura,

            "state": sample.state,

            "action": sample.action,
            "action_kind": sample.action_kind,

            "click_x": (
                None if sample.click is None else int(sample.click[0])
            ),
            "click_y": (
                None if sample.click is None else int(sample.click[1])
            ),

            "target_category": (
                None if alvo is None else alvo.get("category")
            ),
            "target_template": (
                None if alvo is None else alvo.get("template_path")
            ),
            "target_confidence": (
                None if alvo is None else float(alvo.get("confidence", 0.0))
            ),

            "detect_lag_ms": (
                None if sample.lag is None
                else round(sample.lag * 1000)
            ),

            "outcome": sample.outcome,
            "outcome_after_ms": (
                None if sample.outcome_after is None
                else round(sample.outcome_after * 1000)
            ),

            "cycle_count": sample.cycle_count,

            # TODAS as detecções do frame, não só a que virou
            # ação: as outras são rótulo grátis.
            "boxes": [
                {
                    "category": d["category"],
                    "template": d.get("name"),
                    "confidence": round(
                        float(d.get("confidence", 0.0)),
                        4,
                    ),
                    "color_similarity": round(
                        float(d.get("color_similarity", 0.0)),
                        4,
                    ),
                    "x": int(d["x"]),
                    "y": int(d["y"]),
                    "width": int(d["width"]),
                    "height": int(d["height"]),
                    "acted": bool(
                        alvo is not None
                        and d is alvo
                    ),
                }
                for d in sample.detections
            ],
        }

    def _thumb(self, frame):

        lado = DATASET_HASH_SIZE

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return cv2.resize(
            cinza,
            (lado, lado),
            interpolation=cv2.INTER_AREA,
        ).astype(np.int16)

    def _mudou(self, antes, depois):
        """Mudança GRANDE é confiável: rolagem move 8.9-59.8 (medido) vs ruído de
        animação abaixo de 5.8 — a distinção que a dedup por conteúdo não consegue fazer."""

        if antes is None or depois is None:
            return None

        return bool(
            np.abs(antes - depois).mean() > DATASET_SCREEN_CHANGE
        )

    def _perceptual_hash(self, frame):
        """Gravado só como metadado — não filtra aqui, é para dedup offline depois."""

        pequeno = self._thumb(frame)

        bits = (pequeno > pequeno.mean()).flatten()

        return "".join("1" if b else "0" for b in bits)

    def _sha256(self, caminho):
        """Identidade exata do arquivo, para detectar duplicata entre sessões."""

        digest = hashlib.sha256()

        with caminho.open("rb") as arquivo:

            for bloco in iter(lambda: arquivo.read(1 << 20), b""):

                digest.update(bloco)

        return digest.hexdigest()

    def stats(self):

        return {
            "saved": self.saved,
            "duplicates": self.duplicates,
            "dropped": self.dropped,
            "mb": round(self.bytes_written / 1_000_000, 1),
            "by_action": dict(self.by_action),
        }
