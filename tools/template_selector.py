import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

# Without output_dir, writes to tests/capture/ — never to
# tests/images/, which hold the regression fixtures.
from renumerar import proximo_numero, renumerar   # noqa: E402
import selector_layout as layout                  # noqa: E402
from capture.screen import ScreenCapture          # noqa: E402
from core import devices, log                     # noqa: E402
from core.config import (                         # noqa: E402
    DEVICE_SERIAL,
    LOG_LEVEL,
    SELECTOR_FALLBACK_HEIGHT,
    SELECTOR_FALLBACK_WIDTH,
    SELECTOR_HEIGHT_FRACTION,
    SELECTOR_MAX_HEIGHT,
    SELECTOR_MAX_WIDTH,
    SELECTOR_WIDTH_FRACTION,
)


WINDOW_NAME = "Template Selector"
TEMPLATES_DIR = ROOT / "src" / "vision" / "templates"
DEFAULT_TEMPLATES_DIR = TEMPLATES_DIR / "default"

# Manual crops ALWAYS go to default/, at any resolution: the
# resolution folder is written only by the bot. Hence the name carries the
# resolution the crop was taken at — provenance, not destination.
def _prefixo_resolucao(width, height):
    return f"{width}x{height}_"


class TemplateSelector:

    def __init__(self, serial=None):
        # Same source the bot uses at runtime: cropping from screencap produced a
        # lossless template losing ~0.13 confidence against the H.264
        # frame the detector actually compares.
        self.capture = ScreenCapture(serial=serial)
        self.image = None  # imagem original
        self.display = None  # downscaled view
        self.scale = 1.0
        self.start_x = None  # original-image coordinates, not window ones
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.selecting = False

        # Hidden root: only exists so the (category/name) dialogs have
        # somewhere to anchor — the selection window is still cv2's.
        self._tk_root = tk.Tk()
        self._tk_root.withdraw()

    def capture_screen(self):

        print()
        print("[SCREEN] Capturando nova tela...")

        if not self.capture.is_running():
            self.capture.start()

        frame, _, _ = self.capture.get_frame()

        if frame is None:

            raise RuntimeError(
                "Não veio frame do stream."
            )

        self.image = frame.copy()

        self._reset_selection()

        height, width = self.image.shape[:2]

        print(
            f"[SCREEN] Resolução original: "
            f"{width} x {height}"
        )

        print(
            f"[SCREEN] Escala visual: "
            f"{self.scale:.3f}"
        )

    def _janela(self):
        """Maximum window size, in screen pixels."""

        largura = SELECTOR_MAX_WIDTH
        altura = SELECTOR_MAX_HEIGHT

        if largura and altura:
            return largura, altura

        tela = layout.tela_disponivel(
            SELECTOR_HEIGHT_FRACTION,
            SELECTOR_WIDTH_FRACTION,
        )

        if tela is None:

            tela = (
                SELECTOR_FALLBACK_WIDTH,
                SELECTOR_FALLBACK_HEIGHT,
            )

        return (
            largura or tela[0],
            altura or tela[1],
        )

    def _prepare_display(self):
        """Computes the scale; does not alter the original image."""

        height, width = self.image.shape[:2]

        max_width, max_height = self._janela()

        self.scale = layout.escala(
            width,
            height,
            max_width,
            max_height,
        )

        janela = layout.tamanho_canvas(width, height, self.scale)

        print(
            f"[JANELA] {janela[0]}x{janela[1]} | "
            f"escala {self.scale:.3f} "
            f"(1 px na tela = "
            f"{1 / self.scale:.1f} px do device)"
        )

        self._render()

    def _render(self):
        """Builds the view: the whole image, downscaled."""

        height, width = self.image.shape[:2]

        if self.scale == 1.0:

            self.display = self.image.copy()

            return

        self.display = cv2.resize(
            self.image,
            layout.tamanho_canvas(width, height, self.scale),
            interpolation=cv2.INTER_AREA,
        )

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):
        """Stores original-image coordinates, not window ones."""

        if event == cv2.EVENT_LBUTTONDOWN:

            self.start_x, self.start_y = self._para_imagem(x, y)

            self.end_x = self.start_x
            self.end_y = self.start_y

            self.selecting = True

            self._draw_selection()

        elif event == cv2.EVENT_MOUSEMOVE:

            if self.selecting:

                self.end_x, self.end_y = self._para_imagem(x, y)

                self._draw_selection()

        elif event == cv2.EVENT_LBUTTONUP:

            self.end_x, self.end_y = self._para_imagem(x, y)

            self.selecting = False

            self._draw_selection()

    def _para_imagem(self, x, y):

        height, width = self.image.shape[:2]

        return layout.para_imagem(
            x,
            y,
            width,
            height,
            self.scale,
        )

    def _draw_selection(self):

        self._render()

        if (
            self.start_x is None
            or self.start_y is None
            or self.end_x is None
            or self.end_y is None
        ):
            return

        canto_a = layout.para_canvas(
            self.start_x,
            self.start_y,
            self.scale,
        )

        canto_b = layout.para_canvas(
            self.end_x,
            self.end_y,
            self.scale,
        )

        cv2.rectangle(
            self.display,
            canto_a,
            canto_b,
            (0, 255, 0),
            2
        )

        # real crop size (device pixels), not the on-screen size
        largura = abs(self.end_x - self.start_x)
        altura = abs(self.end_y - self.start_y)

        etiqueta = f"{largura}x{altura}"

        posicao = (
            min(canto_a[0], canto_b[0]),
            max(min(canto_a[1], canto_b[1]) - 8, 14),
        )

        cv2.putText(
            self.display,
            etiqueta,
            posicao,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            self.display,
            etiqueta,
            posicao,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    def select(self):

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self.mouse_callback
        )

        print()
        print("===================================")
        print("       TEMPLATE SELECTOR")
        print("===================================")
        print()
        print("A imagem original é mantida intacta.")
        print("Apenas a visualização é reduzida.")
        print()
        print("Arraste sobre o item.")
        print()
        print("ENTER = salvar")
        print("F5    = nova captura da tela")
        print("R     = nova captura da tela")
        print("ESC   = sair")
        print()

        while True:

            cv2.imshow(
                WINDOW_NAME,
                self.display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            # F5 on Windows returns 116; 'r' is the alternative
            if key == 116 or key == ord("r"):

                print()
                print(
                    "[SCREEN] Atualizando captura..."
                )

                self.capture_screen()

                print(
                    "[SCREEN] Nova imagem carregada."
                )

                continue

            if key == 13:

                if not self._valid_selection():

                    print(
                        "Seleção inválida."
                    )

                    continue

                self.save_selection()

                self._reset_selection()

        cv2.destroyWindow(
            WINDOW_NAME
        )

        self._tk_root.destroy()

        self.capture.stop()

    def _valid_selection(self):

        if self.start_x is None:
            return False

        if self.start_y is None:
            return False

        if self.end_x is None:
            return False

        if self.end_y is None:
            return False

        width = abs(
            self.end_x - self.start_x
        )

        height = abs(
            self.end_y - self.start_y
        )

        return (
            width > 5
            and height > 5
        )

    def save_selection(self):

        # coordinates already arrive converted from mouse_callback; dividing by the scale again would crop wrong
        x1 = min(self.start_x, self.end_x)
        x2 = max(self.start_x, self.end_x)

        y1 = min(self.start_y, self.end_y)
        y2 = max(self.start_y, self.end_y)

        height, width = self.image.shape[:2]

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        crop = self.image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            print(
                "Recorte vazio."
            )

            return

        # The crop is never rescaled, and never goes to a
        # resolution folder: only the bot writes there. The detector rescales.
        prefixo = _prefixo_resolucao(width, height)

        category = self._select_category(crop)

        if category is None:
            return

        category_dir = (
            DEFAULT_TEMPLATES_DIR
            / category
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if category == "food":

            output_path = self._food_output_path(
                category_dir,
                crop,
                prefixo,
            )

            if output_path is None:
                return

        else:

            # compact the sequence before saving: len(existing)+1 would silently
            # overwrite on gaps (e.g. food with 17 files up to item_020)
            renomeados = renumerar(
                category_dir,
                aplicar=True
            )

            if renomeados:

                print()
                print(
                    f"[NUMERAÇÃO] {len(renomeados)} arquivo(s) "
                    f"compactado(s) em {category}:"
                )

                for origem, destino in renomeados:

                    print(
                        f"  {origem.name} -> {destino.name}"
                    )

            # Numbering is unique per category, not per prefix: two
            # crops from different screens must not both become item_001.
            number = proximo_numero(category_dir)

            output_path = (
                category_dir
                / f"{prefixo}item_{number:03d}.png"
            )

            while any(
                candidato.stem.endswith(f"item_{number:03d}")
                for candidato in category_dir.glob("*.png")
            ):

                number += 1

                output_path = (
                    category_dir
                    / f"{prefixo}item_{number:03d}.png"
                )

        success = cv2.imwrite(
            str(output_path),
            crop
        )

        if not success:

            raise RuntimeError(
                "Não foi possível salvar "
                "o template."
            )

        print()
        print("===================================")
        print("Template salvo!")
        print("===================================")
        print(
            f"Pasta:     {DEFAULT_TEMPLATES_DIR.name}"
        )
        print(
            f"Categoria: {category}"
        )
        print(
            f"Arquivo:   {output_path}"
        )
        print(
            f"Original:  "
            f"{x1},{y1} → {x2},{y2}"
        )
        print(
            f"Recorte:   "
            f"{x2 - x1} x {y2 - y1} (device)"
        )
        print("===================================")
        print()

    def _food_output_path(self, category_dir, crop, prefixo):
        """Food is named by item, not by number — several bots/stages share a template and the filename helps tell which is which."""

        name = self._ask_name("Nome do food", "Nome do item:", crop)

        if name is None:
            return None

        name = name.replace(" ", "_")

        output_path = category_dir / f"{prefixo}{name}.png"

        numero = 2

        while output_path.exists():

            output_path = (
                category_dir
                / f"{prefixo}{name}_{numero}.png"
            )

            numero += 1

        return output_path

    def _preview_image(self, crop, max_side=420):

        altura, largura = crop.shape[:2]

        escala = max_side / max(altura, largura)

        interpolacao = (
            cv2.INTER_AREA
            if escala < 1.0
            else cv2.INTER_NEAREST
        )

        redimensionado = cv2.resize(
            crop,
            (round(largura * escala), round(altura * escala)),
            interpolation=interpolacao,
        )

        rgb = cv2.cvtColor(redimensionado, cv2.COLOR_BGR2RGB)

        altura, largura = rgb.shape[:2]

        cabecalho = f"P6\n{largura} {altura}\n255\n".encode("ascii")

        return tk.PhotoImage(data=cabecalho + rgb.tobytes())

    def _ask_name(self, title, prompt, crop=None):

        invalid_chars = '<>:"/\\|?*'

        while True:

            resultado = {"value": None}

            janela = tk.Toplevel(self._tk_root)
            janela.title(title)
            janela.attributes("-topmost", True)

            corpo = tk.Frame(janela)
            corpo.pack(padx=12, pady=12)

            if crop is not None:

                preview = self._preview_image(crop)

                preview_label = tk.Label(corpo, image=preview)
                preview_label.image = preview
                preview_label.pack(side="left", padx=(0, 12))

            lado_direito = tk.Frame(corpo)
            lado_direito.pack(side="left")

            tk.Label(lado_direito, text=prompt).pack(pady=(0, 6))

            entrada = tk.Entry(lado_direito, width=30)
            entrada.pack()
            entrada.focus_set()

            def confirmar(event=None):
                resultado["value"] = entrada.get()
                janela.destroy()

            def cancelar():
                janela.destroy()

            entrada.bind("<Return>", confirmar)

            botoes = tk.Frame(lado_direito)
            botoes.pack(pady=(6, 0))

            tk.Button(botoes, text="OK", width=10, command=confirmar).pack(side="left", padx=4)
            tk.Button(botoes, text="Cancelar", width=10, command=cancelar).pack(side="left", padx=4)

            janela.protocol("WM_DELETE_WINDOW", cancelar)

            self._tk_root.wait_window(janela)

            name = resultado["value"]

            if name is None:
                return None

            name = name.strip()

            if not name:
                messagebox.showwarning(title, "Nome inválido.")
                continue

            if any(char in name for char in invalid_chars):
                messagebox.showwarning(title, "Nome contém caracteres inválidos.")
                continue

            return name

    def _reset_selection(self):

        self.start_x = None
        self.start_y = None

        self.end_x = None
        self.end_y = None

        self.selecting = False

        if self.image is not None:

            self._prepare_display()

    def _get_categories(self):
        """Default categories: every resolution folder starts with this same set as default."""

        if not DEFAULT_TEMPLATES_DIR.exists():
            return []

        categories = [
            path.name
            for path in DEFAULT_TEMPLATES_DIR.iterdir()
            if path.is_dir()
        ]

        return sorted(
            categories,
            key=str.lower
        )

    def _select_category(self, crop):

        categories = self._get_categories()

        resultado = {"value": None}

        janela = tk.Toplevel(self._tk_root)
        janela.title("Categoria")
        janela.attributes("-topmost", True)

        corpo = tk.Frame(janela)
        corpo.pack(padx=12, pady=12)

        preview = self._preview_image(crop)

        preview_label = tk.Label(corpo, image=preview)
        preview_label.image = preview
        preview_label.pack(side="left", padx=(0, 12))

        lado_direito = tk.Frame(corpo)
        lado_direito.pack(side="left")

        tk.Label(lado_direito, text="Selecione a categoria:").pack(pady=(0, 6))

        grade = tk.Frame(lado_direito)
        grade.pack()

        def escolher(categoria):
            resultado["value"] = categoria
            janela.destroy()

        colunas = 3

        for indice, categoria in enumerate(categories):

            tk.Button(
                grade,
                text=categoria,
                width=14,
                command=lambda c=categoria: escolher(c),
            ).grid(
                row=indice // colunas,
                column=indice % colunas,
                padx=4,
                pady=4,
            )

        def nova_categoria():

            categoria = self._ask_name("Nova categoria", "Nome da categoria:", crop)

            if categoria:
                escolher(categoria)

        tk.Button(
            lado_direito,
            text="+ Nova categoria",
            command=nova_categoria,
        ).pack(pady=(6, 0))

        janela.protocol("WM_DELETE_WINDOW", janela.destroy)

        self._tk_root.wait_window(janela)

        return resultado["value"]


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Recorta templates da tela do device",
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

    args = parser.parse_args(argv)

    log.setup(LOG_LEVEL)

    try:

        serial = devices.resolver(args.device or DEVICE_SERIAL)

    except KeyboardInterrupt:

        print("Cancelado.")

        return 1

    selector = TemplateSelector(serial)

    selector.capture_screen()
    selector.select()

    return 0


if __name__ == "__main__":

    sys.exit(main())
